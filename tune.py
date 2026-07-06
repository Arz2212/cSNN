
import math, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from spikingjelly.activation_based import neuron, layer, encoding, functional
import numpy as np

from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from hyperopt import hp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(1)

# ══════════════════════════════════════════════════════════════════════════════
#  Модель (из main.py)
# ══════════════════════════════════════════════════════════════════════════════

class CSDPLearner:
    """ΔW = -α * (z_hub / z_max - phase) * z_post ⊗ z_pre"""
    def __init__(self, synapse, tau_z=5.0, gamma_z=1.0, alpha=0.001,
                 w_decay=0.0, w_clip=1.0, dt=1.0):
        self.synapse = synapse
        self.alpha = alpha
        self.tau_z = tau_z
        self.gamma_z = gamma_z
        self.dt = dt
        self.w_decay = w_decay
        self.w_clip = w_clip
        self.z_pre = None
        self.z_post = None
        self.zmax = 1.0
        self._decay = self.dt / self.tau_z

    def step(self, pre_spikes, post_spikes, phase, zhub):
        if self.z_pre is None:
            self.z_pre  = torch.zeros_like(pre_spikes, dtype=torch.float32)
            self.z_post = torch.zeros_like(post_spikes, dtype=torch.float32)
        batch_max = zhub.max().item()
        self.zmax = max(self.zmax, batch_max)
        self.z_pre  += self._decay * (-self.z_pre  + self.gamma_z * pre_spikes.float())
        self.z_post += self._decay * (-self.z_post + self.gamma_z * post_spikes.float())
        factor = zhub / self.zmax - phase
        B = pre_spikes.size(0)
        dw = -self.alpha * ((self.z_post * factor).t() @ self.z_pre) / B
        wd = -self.w_decay * self.synapse.weight.data
        with torch.no_grad():
            self.synapse.weight.data += dw + wd
            if self.w_clip > 0:
                self.synapse.weight.data.clamp_(-self.w_clip, self.w_clip)

    def reset(self):
        self.z_pre = self.z_post = None


class SupervisedLearner:
    def __init__(self, synapse, alpha=0.01, w_clip=1.0):
        self.synapse = synapse
        self.alpha = alpha
        self.w_clip = w_clip

    def step(self, pre_spikes, post_spikes, target):
        B = pre_spikes.size(0)
        error = target - post_spikes.float()
        dw = self.alpha * torch.einsum('bi,bj->ij', error, pre_spikes.float()) / B
        with torch.no_grad():
            self.synapse.weight.data += dw
            if self.w_clip > 0:
                self.synapse.weight.data.clamp_(-self.w_clip, self.w_clip)

    def reset(self):
        pass


class XOR_SNN(nn.Module):
    """3 → 5 → 3 → 2 LIF + хабы + контекст."""
    def __init__(self, n_in=3, n_h1=5, n_h2=3, n_out=2,
                 tau_m=2.0, v_thr=0.3, tau_m_hab=4.0, v_thr_hab=0.15, tau_hub=3.0):
        super().__init__()
        self.n_out   = n_out
        self.tau_hub = tau_hub
        self.encoder = encoding.PoissonEncoder()

        self.fc1  = layer.Linear(n_in,  n_h1, bias=False)
        self.lif1 = neuron.LIFNode(tau=tau_m, v_threshold=v_thr)
        self.fc2  = layer.Linear(n_h1,  n_h2, bias=False)
        self.lif2 = neuron.LIFNode(tau=tau_m, v_threshold=v_thr)
        self.fc3  = layer.Linear(n_h2,  n_out, bias=False)
        self.lif3 = neuron.LIFNode(tau=tau_m, v_threshold=v_thr)

        self.ctx1 = layer.Linear(n_out, n_h1, bias=False)
        self.ctx2 = layer.Linear(n_out, n_h2, bias=False)

        self.hub_fc1_in = layer.Linear(n_h1, 1, bias=False)
        self.hub_lif1    = neuron.LIFNode(tau=tau_m_hab, v_threshold=v_thr_hab)
        self.hub_fc2_in = layer.Linear(n_h2, 1, bias=False)
        self.hub_lif2    = neuron.LIFNode(tau=tau_m_hab, v_threshold=v_thr_hab)

        self._init_weights()

    def _init_weights(self):
        """Детерминированная инициализация — linspace чуть выше середины."""
        for name, m in self.named_modules():
            if isinstance(m, layer.Linear):
                fan_in, fan_out = m.weight.shape[1], m.weight.shape[0]
                n = fan_in * fan_out
                if 'hub' in name:
                    lo, hi = -0.2, 0.5   
                else:
                    lo, hi = -0.1, 0.3   
                vals = torch.linspace(lo, hi, n).reshape(fan_out, fan_in)
                with torch.no_grad():
                    m.weight.copy_(vals)

    def forward(self, x, T, csdp_learners=None, sup_learner=None,
                phase=1.0, labels=None, ctx_scale=1.5):
        functional.reset_net(self)
        B = x.size(0)
        spikes_out = []

        label_1hot = None
        if labels is not None:
            label_1hot = F.one_hot(labels, num_classes=self.n_out).float()
            ctx1_in = self.ctx1(label_1hot) * ctx_scale
            ctx2_in = self.ctx2(label_1hot) * ctx_scale

        y_target = label_1hot if (labels is not None and sup_learner is not None and phase == 1.0) else None

        hub1 = torch.zeros(B, 1, device=x.device)
        hub2 = torch.zeros(B, 1, device=x.device)
        decay_hub = 1.0 / self.tau_hub

        for t in range(T):
            s0 = self.encoder(x)
            ff1 = self.fc1(s0)
            if label_1hot is not None:
                ff1 = ff1 + ctx1_in
            s1 = self.lif1(ff1)
            ff2 = self.fc2(s1)
            if label_1hot is not None:
                ff2 = ff2 + ctx2_in
            s2 = self.lif2(ff2)
            s3 = self.lif3(self.fc3(s2))
            spikes_out.append(s3)

            h1_spk = self.hub_lif1(self.hub_fc1_in(s1))
            h2_spk = self.hub_lif2(self.hub_fc2_in(s2))
            hub1 = hub1 + decay_hub * (-hub1 + h1_spk.float())
            hub2 = hub2 + decay_hub * (-hub2 + h2_spk.float())

            if csdp_learners is not None and label_1hot is not None:
                csdp_learners[0].step(s0, s1, phase, hub1)
                csdp_learners[1].step(s1, s2, phase, hub2)
                csdp_learners[2].step(label_1hot, s1, phase, hub1)
                csdp_learners[3].step(label_1hot, s2, phase, hub2)
            if sup_learner is not None and phase == 1.0 and y_target is not None:
                sup_learner.step(s2, s3, y_target)

        return torch.stack(spikes_out)


# ══════════════════════════════════════════════════════════════════════════════
#  Данные
# ══════════════════════════════════════════════════════════════════════════════

def make_xor_dataset(n_samples=200, rate_low=0.15, rate_high=0.75, rate_bias=0.5):
    patterns = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    X, y = [], []
    for (a, b), label in patterns:
        inp = [rate_high if a else rate_low, rate_high if b else rate_low, rate_bias]
        for _ in range(n_samples):
            X.append(inp); y.append(label)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


@torch.no_grad()
def accuracy(net, loader, T):
    """Точность на одном тестовом прогоне."""
    net.eval()
    ok, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = net(x, T, labels=y)
        ok += (out.sum(0).argmax(1).cpu() == y.cpu()).sum().item()
        total += y.size(0)
    return ok / total


@torch.no_grad()
def avg_accuracy(net, loader, T, n_evals=10):
    """Средняя точность по n_evals тестовым прогонам."""
    accs = [accuracy(net, loader, T) for _ in range(n_evals)]
    return float(np.mean(accs))


TRAIN_ONLY_EPOCHS = 100   # эпохи только обучения (без замера точности)
TRAIN_EVAL_EPOCHS = 30    # эпохи обучения + замер точности (усредняется)
T                 = 100
BATCH             = 64


def objective(config):
    tau_m     = config["tau_m"]
    v_thr     = config["v_thr"]
    tau_hub   = config["tau_hub"]
    v_thr_hab = config["v_thr_hab"]
    tau_m_hab = config["tau_m_hab"]
    tau_z     = config["tau_z"]
    w_clip    = config["w_clip"]

    alpha_csdp = 10 ** config["alpha_csdp_log"]
    alpha_ctx  = 10 ** config["alpha_ctx_log"]
    alpha_sup  = 10 ** config["alpha_sup_log"]
    w_decay    = 10 ** config["w_decay_log"]

    X_train, y_train = make_xor_dataset(n_samples=200)
    X_test,  y_test  = make_xor_dataset(n_samples=50)
    train_ldr = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH, shuffle=True, drop_last=True)
    test_ldr  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=BATCH, shuffle=False)

    net = XOR_SNN(
        n_in=3, n_h1=5, n_h2=3, n_out=2,
        tau_m=tau_m, v_thr=v_thr,
        tau_m_hab=tau_m_hab, v_thr_hab=v_thr_hab, tau_hub=tau_hub,
    )

    csdp = [
        CSDPLearner(net.fc1,  alpha=alpha_csdp, tau_z=tau_z, w_decay=w_decay, w_clip=w_clip),
        CSDPLearner(net.fc2,  alpha=alpha_csdp, tau_z=tau_z, w_decay=w_decay, w_clip=w_clip),
        CSDPLearner(net.ctx1, alpha=alpha_ctx,  tau_z=tau_z, w_decay=w_decay, w_clip=w_clip),
        CSDPLearner(net.ctx2, alpha=alpha_ctx,  tau_z=tau_z, w_decay=w_decay, w_clip=w_clip),
    ]
    sup = SupervisedLearner(net.fc3, alpha=alpha_sup, w_clip=w_clip)

    net.to(DEVICE)

    # ── Фаза 1: 100 эпох чистого обучения ──
    for epoch in range(TRAIN_ONLY_EPOCHS):
        net.train()
        for xb, yb in train_ldr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            net(xb, T, csdp_learners=csdp, sup_learner=sup, phase=1.0, labels=yb)
            for lrn in csdp:
                lrn.reset()
            wrong = 1 - yb
            net(xb, T, csdp_learners=csdp, sup_learner=None, phase=0.0, labels=wrong.to(DEVICE))
            for lrn in csdp:
                lrn.reset()

    # ── Фаза 2: 30 эпох обучения + замера точности ──
    eval_accs = []
    for epoch in range(TRAIN_EVAL_EPOCHS):
        # обучение (одна эпоха)
        net.train()
        for xb, yb in train_ldr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            net(xb, T, csdp_learners=csdp, sup_learner=sup, phase=1.0, labels=yb)
            for lrn in csdp:
                lrn.reset()
            wrong = 1 - yb
            net(xb, T, csdp_learners=csdp, sup_learner=None, phase=0.0, labels=wrong.to(DEVICE))
            for lrn in csdp:
                lrn.reset()
        # замер точности (без градиентов, обучение не идёт во время замера)
        acc = accuracy(net, test_ldr, T)
        eval_accs.append(acc)

    final_acc = float(np.mean(eval_accs))
    tune.report({"accuracy": final_acc})


search_space = {
    # Непрерывные параметры (обычная шкала)
    "tau_m":     hp.uniform("tau_m",       1.0, 5.0),   # постоянная времени LIF
    "v_thr":     hp.uniform("v_thr",       0.1, 0.8),   # порог спайка
    "tau_hub":   hp.uniform("tau_hub",     2.0, 6.0),   # постоянная хаба
    "v_thr_hab": hp.uniform("v_thr_hab",   0.05, 0.4),  # порог хаба
    "tau_m_hab": hp.uniform("tau_m_hab",   2.0, 8.0),   # постоянная нейронов хаба
    "tau_z":     hp.uniform("tau_z",       2.0, 8.0),   # постоянная trace CSDP
    "w_clip":    hp.uniform("w_clip",      0.5, 3.0),   


    "alpha_csdp_log": hp.uniform("alpha_csdp_log", math.log10(0.001), math.log10(0.2)),
    "alpha_ctx_log":  hp.uniform("alpha_ctx_log",  math.log10(0.001), math.log10(0.3)),
    "alpha_sup_log":  hp.uniform("alpha_sup_log",  math.log10(0.001), math.log10(0.2)),
    "w_decay_log":    hp.uniform("w_decay_log",    math.log10(1e-6),  math.log10(1e-2)),
}


if __name__ == "__main__":
    import ray

    N_CPUS = int(os.environ.get("RAY_NUM_CPUS", 10))
    N_GPUS = 1 if torch.cuda.is_available() else 0
    ray.init(num_cpus=N_CPUS, num_gpus=N_GPUS, ignore_reinit_error=True)

    algo = HyperOptSearch(
        search_space,
        metric="accuracy",
        mode="max",
        n_initial_points=1000,
    )

    tuner = tune.Tuner(
        tune.with_resources(objective, {"cpu": 1, "gpu": 0.11}),
        tune_config=tune.TuneConfig(
            search_alg=algo,
            num_samples=10000,
            max_concurrent_trials=9,
        ),
        run_config=tune.RunConfig(
            name="xor_snn_hyperopt",
            storage_path=os.path.abspath("./ray_results"),
            verbose=1,
        ),
    )

    results = tuner.fit()
    best = results.get_best_result(metric="accuracy", mode="max")
    cfg = best.config

    print("\n" + "=" * 60)
    print("Лучшие гиперпараметры (100 эпох обучения + 30 эпох обучения-с-замером):")
    print(f"  {'T':<18s} = {T}  (зафиксировано)")
    print(f"  {'batch_size':<18s} = {BATCH}  (зафиксировано)")
    for name in ["tau_m", "v_thr", "tau_hub", "v_thr_hab", "tau_m_hab", "tau_z", "w_clip"]:
        print(f"  {name:<18s} = {cfg[name]:.4f}")
    print(f"  {'alpha_csdp':<18s} = {10 ** cfg['alpha_csdp_log']:.6f}")
    print(f"  {'alpha_ctx':<18s} = {10 ** cfg['alpha_ctx_log']:.6f}")
    print(f"  {'alpha_sup':<18s} = {10 ** cfg['alpha_sup_log']:.6f}")
    print(f"  {'w_decay':<18s} = {10 ** cfg['w_decay_log']:.8f}")
    print(f"\n  Средняя точность (30 эпох обучения-с-замером) = {best.metrics['accuracy']:.3f}")
    print("=" * 60)
    ray.shutdown()


"""f __name__ == "__main__":
    CFG = {
    "tau_m":     4.2722,
    "v_thr":     0.2892,
    "tau_hub":   4.6653,
    "v_thr_hab": 0.1850,
    "tau_m_hab": 6.2350,
    "tau_z":     5.7368,
    "w_clip":    1.3220,
    "alpha_csdp_log": 0.004626,
    "alpha_ctx_log":  0.006283,
    "alpha_sup_log":  0.002530,
    "w_decay_log":    0.00024425,
}

    print(objective(CFG))"""