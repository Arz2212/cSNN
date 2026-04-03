from ngclearn.utils.io_utils import makedir
from ngclearn.utils.viz.raster import create_raster_plot
from ngclearn.utils.viz.synapse_plot import visualize
from jax import numpy as jnp, random, jit, nn
from ngclearn.utils.model_utils import scanner
from ngcsimlib.compilers import compile_command, wrap_command
from ngcsimlib.context import Context
from ngcsimlib.operations import summation
from ngclearn.components import (VarTrace, BernoulliCell, SLIFCell, RateCell,
                                 StaticSynapse, HebbianSynapse, TraceSTDPSynapse)

from custom.CSDPSynapse import CSDPSynapse
from custom.goodnessModCell import GoodnessModCell
from custom.maskedErrorCell import MaskedErrorCell as ErrorCell
from ngclearn.utils.model_utils import softmax
from img_utils import csdp_deform
import ngclearn.utils.weight_distribution as dist


def reset_synapse(syn, batch_size, synapse_type="hebb"):
    pad = jnp.zeros((batch_size, syn.shape[0]))
    syn.inputs.set(pad)
    pad = jnp.zeros((batch_size, syn.shape[1]))
    syn.outputs.set(pad)
    if synapse_type == "hebb":
        syn.pre.set(pad)
        syn.post.set(pad)
    elif synapse_type == "csdp":
        syn.preSpike.set(pad)
        syn.postSpike.set(pad)
        syn.preTrace.set(pad)
        syn.postTrace.set(pad)


def reset_bernoulli(bern, batch_size):
    pad = jnp.zeros((batch_size, bern.n_units))
    bern.inputs.set(pad)
    bern.outputs.set(pad)
    bern.tols.set(pad)


def reset_errcell(ecell, batch_size):
    pad = jnp.zeros((batch_size, ecell.n_units))
    ecell.mu.set(pad)
    ecell.dmu.set(pad)
    ecell.target.set(pad)
    ecell.dtarget.set(pad)
    ecell.modulator.set(pad + 1.)
    ecell.mask.set(pad + 1.)


def reset_ratecell(rcell, batch_size):
    pad = jnp.zeros((batch_size, rcell.n_units))
    rcell.j.set(pad)
    rcell.j_td.set(pad)
    rcell.z.set(pad)
    rcell.zF.set(pad)


def reset_lif(lif, batch_size):
    pad = jnp.zeros((batch_size, lif.n_units))
    lif.j.set(pad)
    lif.v.set(pad)
    lif.s.set(pad)
    lif.tols.set(pad)
    lif.rfr.set(pad + lif.refract_T)
    lif.surrogate.set(pad + 1.)


def reset_trace(trace, batch_size):
    pad = jnp.zeros((batch_size, trace.n_units))
    trace.outputs.set(pad)
    trace.inputs.set(pad)
    trace.trace.set(pad)


class CSDP_SNN():
    def __init__(self, dkey, in_dim=1, out_dim=1, hid_dim=1024, hid_dim2=1024,
                 batch_size=1, eta=0.002, T=40, dt=3., learn_recon=False,
                 algo_type="supervised", exp_dir="exp", model_name="snn_single_layer",
                 load_model_dir=None, load_param_subdir=None, **kwargs):
        self.exp_dir = exp_dir
        self.model_name = model_name
        if load_model_dir is None:
            makedir(exp_dir)
            makedir(exp_dir + "/filters")
            makedir(exp_dir + "/raster")

        dkey, *subkeys = random.split(dkey, 10)
        self.T = T
        self.dt = dt
        self.algo_type = algo_type
        self.learn_recon = learn_recon

        tau_m = 100.
        vThr = 0.055
        R_m = 0.1
        rho_b = 0.001
        tau_tr = 13.

        weightInit = dist.uniform(amin=-1., amax=1.)
        biasInit = None

        optim_type = "adam"
        nonneg_w = False

        eta_w = eta
        if batch_size >= 200:
            eta_w = 0.002
        else:
            eta_w = 0.001

        if load_model_dir is not None:
            self.load_from_disk(load_model_dir, load_param_subdir)
        else:
            batch_size = 1
            with Context("Circuit") as self.circuit:
                # Входной слой
                self.z0 = BernoulliCell("z0", n_units=in_dim, batch_size=batch_size, key=subkeys[0])

                # Выходной (классифицирующий) слой
                self.zy = SLIFCell(
                    name="zy", n_units=out_dim, tau_m=tau_m, resist_m=1.,
                    thr=vThr, resist_inh=0., refract_time=0., thr_gain=0.,
                    thr_leak=0., rho_b=rho_b, sticky_spikes=False,
                    thr_jitter=0.025, batch_size=batch_size, key=subkeys[1]
                )

                # Контекст / Метки
                self.z3 = BernoulliCell("z3", n_units=out_dim, batch_size=batch_size, key=subkeys[2])

                # Ошибка выхода
                self.ey = ErrorCell(name="ey", n_units=out_dim)

                # Прямой синапс от входа к выходу
                self.W_out = HebbianSynapse(
                    name="W_out", shape=(in_dim, out_dim), eta=eta_w,
                    weight_init=weightInit, bias_init=biasInit, resist_scale=1.,
                    w_bound=1., w_decay=0., sign_value=-1., optim_type=optim_type,
                    pre_wght=1., post_wght=R_m, is_nonnegative=nonneg_w,
                    key=subkeys[3]
                )

                # Статические записи
                self.z0_prev = RateCell("z0_prev", n_units=in_dim, tau_m=0., prior=("gaussian", 0.),
                                        batch_size=batch_size)
                self.z3_prev = RateCell("z3_prev", n_units=out_dim, tau_m=0., prior=("gaussian", 0.),
                                        batch_size=batch_size)

                # Следы (traces)
                self.tr0 = VarTrace("tr0", n_units=in_dim, tau_tr=tau_tr, decay_type="lin", batch_size=batch_size,
                                    a_delta=0., key=subkeys[4])
                self.tr3 = VarTrace("tr3", n_units=out_dim, tau_tr=tau_tr, decay_type="lin", a_delta=0.,
                                    batch_size=batch_size, key=subkeys[5])

                # Подключение
                self.tr0.inputs << self.z0.outputs
                self.tr3.inputs << self.zy.s

                self.z0_prev.j << self.z0.outputs
                self.z3_prev.j << self.z3.outputs

                self.ey.mu << self.tr3.outputs
                self.ey.target << self.z3.outputs

                # Прямая передача сигнала
                self.W_out.inputs << self.z0.outputs
                self.zy.j << self.W_out.outputs

                # Настройка пластичности HebbianSynapse
                self.W_out.pre << self.z0_prev.zF
                self.W_out.post << self.ey.dmu

                # Пути выполнения
                exec_path = [self.z0_prev, self.z3_prev, self.W_out,
                             self.z0, self.z3, self.zy, self.tr0, self.tr3, self.ey]
                evolve_path = [self.W_out]
                save_path = [self.W_out, self.zy]

                reset_cmd, reset_args = self.circuit.compile_by_key(*exec_path, compile_key="reset")
                advance_cmd, advance_args = self.circuit.compile_by_key(*exec_path, compile_key="advance_state")
                evolve_cmd, evolve_args = self.circuit.compile_by_key(*evolve_path, compile_key="evolve")
                self.dynamic()
        self.traces = ["tr0", "tr3"]
        self.input_cells = ["z0", "z3"]
        self.lifs = ["zy"]
        self.ratecells = ["z0_prev", "z3_prev"]
        self.hebb_synapses = ["W_out"]
        self.ecells = ["ey"]
        self.csdp_synapses = []
        self.gcells = []

        self.saveable_comps = (self.input_cells + self.lifs + self.ratecells +
                               self.ecells + self.traces + self.hebb_synapses)

    def dynamic(self):
        z0, z3, zy, ey = self.circuit.get_components("z0", "z3", "zy", "ey")

        self.circuit.add_command(wrap_command(jit(self.circuit.reset)), name="reset")
        self.circuit.add_command(wrap_command(jit(self.circuit.advance_state)), name="advance")
        self.circuit.add_command(wrap_command(jit(self.circuit.evolve)), name="evolve")

        @Context.dynamicCommand
        def clamp_input(x):
            z0.inputs.set(x)

        @Context.dynamicCommand
        def clamp_target(y):
            z3.inputs.set(y)

        @Context.dynamicCommand
        def clamp_mod_labels(labs):
            ey.mask.set(labs)

    def save_to_disk(self, save_dir, params_only=False):
        if params_only:
            model_dir = "{}/{}/{}".format(self.exp_dir, self.model_name, save_dir)
            makedir(model_dir)
            for comp_name in self.saveable_comps:
                comp = self.circuit.components.get(comp_name)
                comp.save(model_dir)
        else:
            self.circuit.save_to_json(self.exp_dir, self.model_name)

    def load_from_disk(self, model_directory, param_subdir="/custom"):
        with Context("Circuit") as circuit:
            self.circuit = circuit
            self.circuit.load_from_dir(model_directory, custom_folder=param_subdir)
            self.dynamic()
        self.W_out = self.circuit.components.get("W_out")
        self.z0_prev = self.circuit.components.get("z0_prev")
        self.z3_prev = self.circuit.components.get("z3_prev")
        self.z0 = self.circuit.components.get("z0")
        self.z3 = self.circuit.components.get("z3")
        self.zy = self.circuit.components.get("zy")
        self.tr0 = self.circuit.components.get("tr0")
        self.tr3 = self.circuit.components.get("tr3")

    def get_synapse_stats(self, param_name="W_out"):
        _W = self.circuit.components.get(param_name).weights.value
        msg = "{}:\n  min {} ;  max {} \n  mu {} ;  norm {}".format(
            param_name, jnp.amin(_W), jnp.amax(_W), jnp.mean(_W), jnp.linalg.norm(_W)
        )
        return msg

    def process(self, Xb, Yb, dkey, adapt_synapses=False, collect_spikes=False,
                collect_rate_codes=False, lab_estimator="softmax",
                collect_recon=True, Xb_neg=None, Yb_neg=None):
        dkey, *subkeys = random.split(dkey, 2)
        if adapt_synapses:
            if self.algo_type == "supervised":
                Yb_neg = random.uniform(subkeys[0], Yb.shape, minval=0., maxval=1.) * (1. - Yb)
                Yb_neg = nn.one_hot(jnp.argmax(Yb_neg, axis=1), num_classes=Yb.shape[1], dtype=jnp.float32)
                Xb_neg = Xb
            else:
                bsize = Xb.shape[0]
                _Xb = jnp.expand_dims(jnp.reshape(Xb, (bsize, 28, 28)), axis=3)
                # Предполагается наличие csdp_deform в img_utils
                Xb_neg, Yb_neg = csdp_deform(subkeys[0], _Xb, Yb, alpha=0.5, use_rot=False)
                Xb_neg = jnp.reshape(jnp.squeeze(Xb_neg, axis=3), (bsize, 28 * 28))

            _Xb = jnp.concatenate((Xb, Xb_neg), axis=0)
            _Yb = jnp.concatenate((Yb, Yb_neg), axis=0)
            mod_signal = jnp.concatenate((jnp.ones((Xb.shape[0], 1)), jnp.zeros((Xb_neg.shape[0], 1))), axis=0)
        else:
            _Yb = Yb * 0
            _Xb = Xb
            mod_signal = jnp.ones((Xb.shape[0], 1))

        self.circuit.reset()
        batch_size = _Xb.shape[0]

        for name in self.traces:
            reset_trace(self.circuit.components.get(name), batch_size)
        for name in self.ecells:
            reset_errcell(self.circuit.components.get(name), batch_size)
        for name in self.input_cells:
            reset_bernoulli(self.circuit.components.get(name), batch_size)
        for name in self.lifs:
            reset_lif(self.circuit.components.get(name), batch_size)
        for name in self.ratecells:
            reset_ratecell(self.circuit.components.get(name), batch_size)

        if adapt_synapses:
            reset_synapse(self.circuit.components.get("W_out"), batch_size, synapse_type="hebb")

        s0_mu = _Xb * 0
        y_count = 0.
        self.z3.inputs.set(_Yb)
        self.z3_prev.z.set(_Yb)
        T = self.T + 1

        for ts in range(1, T):
            self.circuit.clamp_input(_Xb)
            self.circuit.clamp_target(_Yb)
            self.circuit.clamp_mod_labels(mod_signal)
            self.circuit.advance(t=ts * self.dt, dt=self.dt)
            if adapt_synapses:
                self.circuit.evolve(t=ts * self.dt, dt=self.dt)

            y_count = self.zy.s.value + y_count
            s0_mu = self.tr0.outputs.value + s0_mu

        s0_mu = s0_mu / T

        if lab_estimator == "softmax":
            y_hat = softmax(y_count)
        else:
            y_hat = y_count

        # Возвращаем нули вместо R1 и R2, чтобы train_csdp.py корректно распаковал 6 переменных
        R1 = 0.
        R2 = 0.
        R3 = y_hat

        return y_hat, y_count, R1, R2, R3, s0_mu