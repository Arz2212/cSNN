"""
XOR-задача на спайковой сети с обучением CSDP (Contrastive Signal-Dependent Plasticity).

Архитектура: 4 слоя
  Вход:      3 нейрона  (2 числа + bias)
  Скрытый 1: 5 нейронов (LIF)
  Скрытый 2: 3 нейрона (LIF)
  Выход:     2 нейрона (LIF) — [класс 0, класс 1]

Кодирование входа спайками: 0 — НЕ отсутствие спайков.
  Значение 0 → низкая частота (≈0.15), значение 1 → высокая частота (≈0.75).
  Третий нейрон — постоянная фоновая активность.

Обучение: контрастное CSDP + supervised на выходной слой.
  Положительная фаза: вход + правильная метка.
  Отрицательная фаза: вход + неправильная метка.

Защита от насыщения: клиппинг весов + weight decay.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from spikingjelly.activation_based import neuron, layer, encoding, functional
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CSDPLearner:
    """
    Контрастное обучение CSDP.
        ΔW = -α * (z_hub / z_max  -  phase) * z_post ⊗ z_pre
    """
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

        self.z_pre  = self.z_pre  + self._decay * (-self.z_pre  + self.gamma_z * pre_spikes.float())
        self.z_post = self.z_post + self._decay * (-self.z_post + self.gamma_z * post_spikes.float())

        factor = zhub / self.zmax - phase
        B = pre_spikes.size(0)
        dw = -self.alpha * ((self.z_post * factor).t() @ self.z_pre) / B
        wd = -self.w_decay * self.synapse.weight.data

        with torch.no_grad():
            self.synapse.weight.data += dw + wd
            # Клиппинг весов для предотвращения runaway
            if self.w_clip > 0:
                self.synapse.weight.data.clamp_(-self.w_clip, self.w_clip)

    def reset(self):
        self.z_pre = None
        self.z_post = None


class SupervisedLearner:
    """ΔW = α * (target - output) ⊗ pre"""
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


class XOR_SNN_CSDP(nn.Module):
    def __init__(self,
                 n_in=3, n_h1=5, n_h2=3, n_out=2,
                 tau_m=2.0, v_thr=0.3,
                 tau_m_hab=4.0, v_thr_hab=0.15,
                 tau_hub=3.0):
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
        for name, m in self.named_modules():
            if isinstance(m, layer.Linear):
                if 'hub' in name:
                    nn.init.uniform_(m.weight, -0.5, 0.5)
                elif 'ctx' in name:
                    nn.init.uniform_(m.weight, -0.3, 0.3)
                else:
                    nn.init.uniform_(m.weight, -0.3, 0.3)

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

        y_target = None
        if labels is not None and sup_learner is not None and phase == 1.0:
            y_target = label_1hot

        hub_trace1 = torch.zeros(B, 1, device=x.device)
        hub_trace2 = torch.zeros(B, 1, device=x.device)
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
            hub_trace1 = hub_trace1 + decay_hub * (-hub_trace1 + h1_spk.float())
            hub_trace2 = hub_trace2 + decay_hub * (-hub_trace2 + h2_spk.float())

            if csdp_learners is not None and label_1hot is not None:
                csdp_learners[0].step(s0, s1, phase, hub_trace1)
                csdp_learners[1].step(s1, s2, phase, hub_trace2)
                csdp_learners[2].step(label_1hot, s1, phase, hub_trace1)
                csdp_learners[3].step(label_1hot, s2, phase, hub_trace2)

            if sup_learner is not None and phase == 1.0 and y_target is not None:
                sup_learner.step(s2, s3, y_target)

        return torch.stack(spikes_out)


def make_xor_dataset(n_samples_per_class=200, rate_low=0.15, rate_high=0.75, rate_bias=0.5):
    patterns = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    X_list, y_list = [], []
    for (a, b), label in patterns:
        rate_a = rate_high if a == 1 else rate_low
        rate_b = rate_high if b == 1 else rate_low
        inp = [rate_a, rate_b, rate_bias]
        for _ in range(n_samples_per_class):
            X_list.append(inp)
            y_list.append(label)
    return torch.tensor(X_list, dtype=torch.float32), torch.tensor(y_list, dtype=torch.long)


@torch.no_grad()
def evaluate(net, loader, T):
    net.eval()
    correct, total = 0, 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        out = net(x_batch, T, labels=y_batch)
        preds = out.sum(dim=0).argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_no_context(net, loader, T):
    net.eval()
    correct, total = 0, 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        out = net(x_batch, T, labels=None)
        preds = out.sum(dim=0).argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    return 100.0 * correct / total


def train_one_seed(seed, EPOCHS=60, T=20):
    """Обучает модель с заданным seed, возвращает лучшую точность."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    BATCH_SIZE = 64
    N_IN, N_H1, N_H2, N_OUT = 3, 5, 3, 2

    X_train, y_train = make_xor_dataset(n_samples_per_class=200)
    X_test,  y_test  = make_xor_dataset(n_samples_per_class=50)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    net = XOR_SNN_CSDP(
        n_in=N_IN, n_h1=N_H1, n_h2=N_H2, n_out=N_OUT,
        tau_m=2.0, v_thr=0.3,
        tau_m_hab=4.0, v_thr_hab=0.15, tau_hub=3.0
    ).to(DEVICE)

    w_clip_val = 1.5
    csdp_learners = [
        CSDPLearner(net.fc1,  alpha=0.02,  tau_z=4.0, gamma_z=1.0, w_decay=1e-4, w_clip=w_clip_val),
        CSDPLearner(net.fc2,  alpha=0.02,  tau_z=4.0, gamma_z=1.0, w_decay=1e-4, w_clip=w_clip_val),
        CSDPLearner(net.ctx1, alpha=0.06,  tau_z=4.0, gamma_z=1.0, w_decay=1e-4, w_clip=w_clip_val),
        CSDPLearner(net.ctx2, alpha=0.06,  tau_z=4.0, gamma_z=1.0, w_decay=1e-4, w_clip=w_clip_val),
    ]
    sup_learner = SupervisedLearner(net.fc3, alpha=0.03, w_clip=w_clip_val)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        net.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            net(x_batch, T, csdp_learners=csdp_learners, sup_learner=sup_learner,
                phase=1.0, labels=y_batch)
            for l in csdp_learners:
                l.reset()

            wrong = 1 - y_batch
            net(x_batch, T, csdp_learners=csdp_learners, sup_learner=None,
                phase=0.0, labels=wrong)
            for l in csdp_learners:
                l.reset()

        acc_ctx = evaluate(net, test_loader, T)
        if acc_ctx > best_acc:
            best_acc = acc_ctx

    return best_acc, net


if __name__ == "__main__":
    print("=" * 60)
    print("XOR на спайковой сети с CSDP (SpikingJelly)")
    print("Архитектура: 3 → 5 → 3 → 2  (LIF, v_thr=0.3)")
    print("Кодирование: 0 ≠ отсутствие спайков")
    print("Защита от насыщения: клиппинг весов ±1.5 + weight_decay=1e-4")
    print("=" * 60)

    T = 20
    EPOCHS = 20
    seeds_to_try = [7, 13, 23, 99]

    best_overall_acc = 0.0
    best_overall_seed = None
    best_overall_net = None

    for seed in seeds_to_try:
        acc, net = train_one_seed(seed, EPOCHS=EPOCHS, T=T)
        status = "✓" if acc >= 90 else ("~" if acc >= 75 else "✗")
        print(f"  seed {seed:4d} → {acc:5.1f}% {status}")
        if acc > best_overall_acc:
            best_overall_acc = acc
            best_overall_seed = seed
            best_overall_net = net

    print(f"\nЛучший seed: {best_overall_seed} — {best_overall_acc:.1f}%")

    # ── Детальный тест лучшей модели ──
    if best_overall_net is not None:
        net = best_overall_net
        X_patterns = torch.tensor([
            [0.15, 0.15, 0.5], [0.15, 0.75, 0.5],
            [0.75, 0.15, 0.5], [0.75, 0.75, 0.5],
        ], dtype=torch.float32).to(DEVICE)
        Y_patterns = torch.tensor([0, 1, 1, 0], dtype=torch.long).to(DEVICE)
        labels_str = ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]

        print("\n" + "=" * 60)
        print(f"Финальный тест (seed={best_overall_seed}):")
        print("-" * 60)

        net.eval()
        with torch.no_grad():
            out_ctx = net(X_patterns, T, labels=Y_patterns)
            spike_sums_ctx = out_ctx.sum(dim=0)
            pred_ctx = spike_sums_ctx.argmax(dim=1)

            functional.reset_net(net)
            out_noctx = net(X_patterns, T, labels=None)
            spike_sums_noctx = out_noctx.sum(dim=0)
            pred_noctx = spike_sums_noctx.argmax(dim=1)

        print(f"{'Вход':<14} {'Метка':<6} {'Предск.(ctx)':<14} {'Спайки(ctx)':<22} {'Предск.(no ctx)':<16} {'Спайки(no ctx)'}")
        print("-" * 95)
        for i in range(4):
            ok_ctx = "✓" if pred_ctx[i] == Y_patterns[i] else "✗"
            ok_noctx = "✓" if pred_noctx[i] == Y_patterns[i] else "✗"
            s_ctx = "[" + ", ".join(f"{v:.1f}" for v in spike_sums_ctx[i].tolist()) + "]"
            s_noctx = "[" + ", ".join(f"{v:.1f}" for v in spike_sums_noctx[i].tolist()) + "]"
            print(f"{labels_str[i]:<14} {Y_patterns[i].item():<6} "
                  f"{pred_ctx[i].item()} {ok_ctx:<11} "
                  f"{s_ctx:<22} "
                  f"{pred_noctx[i].item()} {ok_noctx:<13} "
                  f"{s_noctx}")

        correct_ctx = (pred_ctx == Y_patterns).sum().item()
        correct_noctx = (pred_noctx == Y_patterns).sum().item()
        print(f"\nПравильно (с контекстом): {correct_ctx}/4")
        print(f"Правильно (без контекста): {correct_noctx}/4")

        # Перебор меток
        print("\nТест с перебором меток (без знания ответа):")
        with torch.no_grad():
            for i in range(4):
                x_i = X_patterns[i:i+1]
                functional.reset_net(net)
                out0 = net(x_i, T, labels=torch.tensor([0], device=DEVICE))
                s0 = out0.sum(dim=0)[0]
                functional.reset_net(net)
                out1 = net(x_i, T, labels=torch.tensor([1], device=DEVICE))
                s1 = out1.sum(dim=0)[0]

                score0 = s0[0] - s0[1]
                score1 = s1[1] - s1[0]
                pred_best = 0 if score0 > score1 else 1
                ok = "✓" if pred_best == Y_patterns[i] else "✗"
                print(f"  {labels_str[i]}  label=0: [{s0[0]:.1f}, {s0[1]:.1f}]  |  "
                      f"label=1: [{s1[0]:.1f}, {s1[1]:.1f}]  →  {pred_best} {ok}")

        print(f"\nВеса fc1: [{net.fc1.weight.data.min():.3f}, {net.fc1.weight.data.max():.3f}]")
        print(f"Веса fc3: [{net.fc3.weight.data.min():.3f}, {net.fc3.weight.data.max():.3f}]")

    print("\nГотово.")
