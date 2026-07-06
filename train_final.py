"""
Обучение XOR-сети 1000 эпох → 100 тестовых прогонов.
Каждые 10 эпох — промежуточный тест (без обучения).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from spikingjelly.activation_based import neuron, layer, encoding, functional
import numpy as np

DEVICE = torch.device("cpu")
torch.set_num_threads(8)


T            = 100
BATCH        = 64
TRAIN_EPOCHS = 1000
TEST_EVAL    = 10    
FINAL_EVALS  = 100   

CFG = {
    "tau_m":     1.9629,
    "v_thr":     0.5628,
    "tau_hub":   5.4804,
    "v_thr_hab": 0.1971,
    "tau_m_hab": 2.7790,
    "tau_z":     7.0780,
    "w_clip":    1.6154,
    "alpha_csdp": 0.008309,
    "alpha_ctx":  0.008208,
    "alpha_sup":  0.019609,
    "w_decay":    0.00007597,
}



class CSDPLearner:
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

        hub1, hub2 = torch.zeros(B, 1), torch.zeros(B, 1)
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
def test_accuracy(net, loader, T, n_evals):
    """Средняя точность по n_evals тестовым прогонам (без обучения)."""
    net.eval()
    accs = []
    for _ in range(n_evals):
        ok, total = 0, 0
        for x, y in loader:
            out = net(x, T, labels=y)
            ok += (out.sum(0).argmax(1) == y).sum().item()
            total += y.size(0)
        accs.append(ok / total)
    return float(np.mean(accs)), float(np.std(accs))



if __name__ == "__main__":
    print("=" * 60)
    print(f"XOR SNN — обучение {TRAIN_EPOCHS} эпох, T={T}, batch={BATCH}")
    print("=" * 60)

    X_train, y_train = make_xor_dataset(n_samples=200)
    X_test,  y_test  = make_xor_dataset(n_samples=50)
    train_ldr = DataLoader(TensorDataset(X_train, y_train), BATCH, shuffle=True, drop_last=True)
    test_ldr  = DataLoader(TensorDataset(X_test,  y_test),  BATCH, shuffle=False)

    net = XOR_SNN(
        n_in=3, n_h1=5, n_h2=3, n_out=2,
        tau_m=CFG["tau_m"], v_thr=CFG["v_thr"],
        tau_m_hab=CFG["tau_m_hab"], v_thr_hab=CFG["v_thr_hab"], tau_hub=CFG["tau_hub"],
    )

    csdp = [
        CSDPLearner(net.fc1,  alpha=CFG["alpha_csdp"], tau_z=CFG["tau_z"], w_decay=CFG["w_decay"], w_clip=CFG["w_clip"]),
        CSDPLearner(net.fc2,  alpha=CFG["alpha_csdp"], tau_z=CFG["tau_z"], w_decay=CFG["w_decay"], w_clip=CFG["w_clip"]),
        CSDPLearner(net.ctx1, alpha=CFG["alpha_ctx"],  tau_z=CFG["tau_z"], w_decay=CFG["w_decay"], w_clip=CFG["w_clip"]),
        CSDPLearner(net.ctx2, alpha=CFG["alpha_ctx"],  tau_z=CFG["tau_z"], w_decay=CFG["w_decay"], w_clip=CFG["w_clip"]),
    ]
    sup = SupervisedLearner(net.fc3, alpha=CFG["alpha_sup"], w_clip=CFG["w_clip"])

    print(f"\nОбучение ({TRAIN_EPOCHS} эпох, тест каждые {TEST_EVAL})...\n")

    for epoch in range(1, TRAIN_EPOCHS + 1):
        net.train()
        for xb, yb in train_ldr:
            net(xb, T, csdp_learners=csdp, sup_learner=sup, phase=1.0, labels=yb)
            for lrn in csdp:
                lrn.reset()
            wrong = 1 - yb
            net(xb, T, csdp_learners=csdp, sup_learner=None, phase=0.0, labels=wrong)
            for lrn in csdp:
                lrn.reset()

        if epoch % 10 == 0:
            acc, std = test_accuracy(net, test_ldr, T, TEST_EVAL)
            print(f"  эпоха {epoch:5d}/{TRAIN_EPOCHS}  |  точность = {acc:.4f} ± {std:.4f}")

    # ── финал: 100 тестовых прогонов ──
    print(f"\nФинальный тест ({FINAL_EVALS} прогонов)...")
    final_acc, final_std = test_accuracy(net, test_ldr, T, FINAL_EVALS)
    print(f"\n{'=' * 60}")
    print(f"  Средняя точность: {final_acc:.4f} ± {final_std:.4f}")
    print(f"{'=' * 60}")
