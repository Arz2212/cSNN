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


def reset_goodnesscell(gcell, batch_size):
    pad = jnp.zeros((batch_size, gcell.n_units))
    gcell.inputs.set(pad)
    gcell.modulator.set(pad + 1.)
    gcell.contrastLabels.set(jnp.zeros((batch_size, 1)))


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
                 algo_type="supervised", exp_dir="exp", model_name="snn_csdp",
                 load_model_dir=None, load_param_subdir=None, A_p=0.05, A_m=0.05, t_w=5., **kwargs):
        self.exp_dir = exp_dir
        self.model_name = model_name
        if load_model_dir is None:
            makedir(exp_dir)
            makedir(exp_dir + "/filters")
            makedir(exp_dir + "/raster")

        dkey, *subkeys = random.split(dkey, 20)
        self.T = T
        self.dt = dt

        self.algo_type = algo_type
        self.learn_recon = learn_recon
        tau_m = 100.
        vThr = 0.055
        R_m = 0.1
        inh_R = 0.01
        rho_b = 0.001
        tau_tr = 13.

        weightInit = dist.uniform(amin=-1., amax=1.)
        biasInit = None

        self.use_rot = False
        self.alpha = 0.5
        optim_type = "adam"
        goodnessThr1 = goodnessThr2 = 10.
        use_dyn_threshold = False
        nonneg_w = False

        eta_w = eta
        if batch_size >= 200:
            eta_w = 0.002
            w_decay = 0.00005
        elif batch_size >= 100:
            eta_w = 0.001
            w_decay = 0.00006
        elif batch_size >= 50:
            eta_w = 0.001
            w_decay = 0.00007
        elif batch_size >= 20:
            eta_w = 0.00075
            w_decay = 0.00008
        elif batch_size >= 10:
            eta_w = 0.00055
            w_decay = 0.00009
        else:
            eta_w = 0.0004
            w_decay = 0.0001
        soft_bound = False

        if algo_type == "unsupervised":
            goodnessThr1 = goodnessThr2 = 10.
            use_dyn_threshold = False
            soft_bound = False
            self.use_rot = False

        if load_model_dir is not None:
            self.load_from_disk(load_model_dir, load_param_subdir)
        else:
            batch_size = 1
            with Context("Circuit") as self.circuit:
                self.z0 = BernoulliCell("z0", n_units=in_dim,
                                        batch_size=batch_size, key=subkeys[0])

                # W2 теперь соединяет in_dim напрямую с hid_dim2
                self.W2 = CSDPSynapse(
                    name="W2", shape=(in_dim, hid_dim2), eta   =eta_w,
                    weight_init=weightInit, bias_init=biasInit, w_bound=1.,
                    is_nonnegative=nonneg_w, w_decay=w_decay, resist_scale=R_m,
                    optim_type=optim_type, soft_bound=soft_bound, key=subkeys[3]
                )
                self.z2 = SLIFCell(
                    name="z2", n_units=hid_dim2, tau_m=tau_m, resist_m=1.,
                    thr=vThr, resist_inh=0., refract_time=0., thr_gain=0.,
                    thr_leak=0., rho_b=rho_b, sticky_spikes=False,
                    thr_jitter=0.025, batch_size=batch_size, key=subkeys[4]
                )
                self.M2 = CSDPSynapse(
                    name="M2", shape=(hid_dim2, hid_dim2), eta=eta_w,
                    weight_init=weightInit, bias_init=biasInit, w_bound=1.,
                    is_nonnegative=True, w_decay=w_decay, resist_scale=inh_R,
                    is_hollow=True, w_sign=-1.,
                    optim_type=optim_type, soft_bound=soft_bound, key=subkeys[7]
                )
                self.zy = SLIFCell(
                    name="zy", n_units=out_dim, tau_m=tau_m, resist_m=1.,
                    thr=vThr, resist_inh=0.,
                    refract_time=0., thr_gain=0., thr_leak=0., rho_b=rho_b,
                    sticky_spikes=False, thr_jitter=0.025,
                    batch_size=batch_size, key=subkeys[8]
                )
                self.ey = ErrorCell(name="ey", n_units=out_dim)

                self.C3 = HebbianSynapse(
                    name="C3", shape=(hid_dim2, out_dim), eta=eta_w,
                    weight_init=weightInit, bias_init=biasInit, resist_scale=1.,
                    w_bound=1., w_decay=0., sign_value=-1., optim_type=optim_type,
                    pre_wght=1., post_wght=R_m, is_nonnegative=nonneg_w,
                    key=subkeys[10]
                )
                if self.learn_recon:
                    self.zR = SLIFCell(
                        name="zR", n_units=in_dim, tau_m=tau_m, resist_m=1.,
                        thr=vThr, resist_inh=0.,
                        refract_time=0., thr_gain=0., thr_leak=0., rho_b=0.,
                        sticky_spikes=False, thr_jitter=0.025,
                        batch_size=batch_size, key=subkeys[11]
                    )
                    self.eR = ErrorCell(name="eR", n_units=in_dim)
                    self.R1 = HebbianSynapse(
                        name="R1", shape=(hid_dim2, in_dim), eta=eta_w,  # shape updated
                        weight_init=weightInit, bias_init=biasInit,
                        resist_scale=1., w_bound=1., w_decay=0., sign_value=-1.,
                        optim_type=optim_type, pre_wght=1., post_wght=R_m,
                        is_nonnegative=nonneg_w, key=subkeys[12]
                    )
                self.z3 = BernoulliCell(
                    "z3", n_units=out_dim, batch_size=batch_size, key=subkeys[13]
                )
                if self.algo_type == "supervised":
                    self.V3y = CSDPSynapse(
                        name="V3y", shape=(out_dim, hid_dim2),
                        eta=eta_w,
                        weight_init=weightInit, bias_init=biasInit, w_bound=1.,
                        is_nonnegative=nonneg_w, w_decay=w_decay,
                        resist_scale=R_m,
                        optim_type=optim_type, soft_bound=soft_bound,
                        key=subkeys[13]
                    )

                self.g2 = GoodnessModCell(name="g2", n_units=hid_dim2, threshold=goodnessThr2,
                                          use_dyn_threshold=use_dyn_threshold)

                self.z0_prev = RateCell(
                    name="z0_prev", n_units=in_dim, tau_m=0.,
                    prior=("gaussian", 0.), batch_size=batch_size
                )
                self.z2_prev = RateCell(
                    name="z2_prev", n_units=hid_dim2, tau_m=0.,
                    prior=("gaussian", 0.), batch_size=batch_size
                )
                self.z3_prev = RateCell(
                    name="z3_prev", n_units=out_dim, tau_m=0.,
                    prior=("gaussian", 0.), batch_size=batch_size
                )

                self.tr0 = VarTrace(
                    "tr0", n_units=in_dim, tau_tr=tau_tr, decay_type="lin",
                    batch_size=batch_size, a_delta=0., key=subkeys[15]
                )
                self.tr2 = VarTrace(
                    "tr2", n_units=hid_dim2, tau_tr=tau_tr, decay_type="lin",
                    a_delta=0., batch_size=batch_size, key=subkeys[15]
                )
                self.tr3 = VarTrace(
                    "tr3", n_units=out_dim, tau_tr=tau_tr, decay_type="lin",
                    a_delta=0., batch_size=batch_size, key=subkeys[15]
                )

                self.tr2.inputs << self.z2.s
                self.tr3.inputs << self.zy.s
                self.g2.inputs << self.tr2.trace

                self.z0_prev.j << self.z0.outputs
                self.z2_prev.j << self.z2.s
                self.z3_prev.j << self.z3.outputs

                self.ey.mu << self.tr3.outputs
                self.ey.target << self.z3.outputs
                self.C3.inputs << self.z2.s
                self.zy.j << self.C3.outputs

                if self.learn_recon:
                    self.R1.inputs << self.z2.s
                    self.zR.j << self.R1.outputs
                    self.tr0.inputs << self.zR.s
                    self.eR.mu << self.zR.s
                    self.eR.target << self.z0.outputs
                    self.R1.pre << self.z2.s
                    self.R1.post << self.eR.dmu

                self.W2.inputs << self.z0.outputs
                self.M2.inputs << self.z2_prev.zF
                if self.algo_type == "supervised":
                    self.V3y.inputs << self.z3_prev.zF
                    self.z2.j << summation(self.W2.outputs, self.M2.outputs,
                                           self.V3y.outputs)
                else:
                    self.z2.j << summation(self.W2.outputs, self.M2.outputs)

                self.C3.pre << self.z2_prev.zF
                self.C3.post << self.ey.dmu

                self.W2.preSpike << self.z0_prev.zF
                self.W2.postSpike << self.z2.s
                self.W2.postTrace << self.g2.modulator

                self.M2.preSpike << self.z2_prev.zF
                self.M2.postSpike << self.z2.s
                self.M2.preTrace << self.z2_prev.zF
                self.M2.postTrace << self.g2.modulator

                if self.algo_type == "supervised":
                    self.V3y.preSpike << self.z3_prev.zF
                    self.V3y.postSpike << self.z2.s
                    self.V3y.preTrace << self.z3_prev.zF
                    self.V3y.postTrace << self.g2.modulator

                if self.algo_type == "supervised":
                    exec_path = [self.z0_prev, self.z2_prev, self.z3_prev,
                                 self.W2, self.V3y, self.M2, self.C3,
                                 self.z0, self.z2, self.z3, self.zy,
                                 self.tr2, self.tr3, self.g2, self.ey]
                    evolve_path = [self.W2, self.V3y, self.M2, self.C3]
                    save_path = [self.W2, self.V3y, self.M2, self.C3, self.z2, self.zy]
                else:
                    exec_path = [self.z0_prev, self.z2_prev, self.z3_prev,
                                 self.W2, self.M2, self.C3,
                                 self.z0, self.z2, self.z3, self.zy,
                                 self.tr2, self.tr3, self.g2, self.ey]
                    evolve_path = [self.W2, self.M2, self.C3]
                    save_path = [self.W2, self.M2, self.C3, self.z2, self.zy]

                if self.learn_recon:
                    recon_path = [self.R1, self.zR, self.tr0, self.eR]
                    exec_path = exec_path + recon_path
                    evolve_path = evolve_path + [self.R1]
                    save_path = save_path + [self.zR, self.R1]

                reset_cmd, reset_args = self.circuit.compile_by_key(
                    *exec_path, compile_key="reset")
                advance_cmd, advance_args = self.circuit.compile_by_key(
                    *exec_path, compile_key="advance_state")
                evolve_cmd, evolve_args = self.circuit.compile_by_key(
                    *evolve_path, compile_key="evolve")
                self.dynamic()

        self.traces = ["tr0", "tr2", "tr3"]
        self.gcells = ["g2"]
        self.input_cells = ["z0", "z3"]
        self.lifs = ["z2", "zy"]
        self.ratecells = ["z0_prev", "z2_prev", "z3_prev"]
        if self.algo_type == "supervised":
            self.csdp_synapses = ["W2", "V3y", "M2"]
        else:
            self.csdp_synapses = ["W2", "M2"]
        self.hebb_synapses = ["C3"]
        self.ecells = ["ey"]
        if self.learn_recon:
            self.ecells = self.ecells + ["eR"]
            self.lifs = self.lifs + ["zR"]
            self.hebb_synapses = self.hebb_synapses + ["R1"]
        self.saveable_comps = (self.input_cells + self.lifs + self.ratecells +
                               self.ecells + self.gcells + self.traces +
                               self.hebb_synapses + self.csdp_synapses)

    def dynamic(self):
        z0, z2, z3, zy = self.circuit.get_components("z0", "z2", "z3", "zy")
        g2, ey, eR = self.circuit.get_components("g2", "ey", "eR")

        self.circuit.add_command(
            wrap_command(jit(self.circuit.reset)), name="reset")
        self.circuit.add_command(
            wrap_command(jit(self.circuit.advance_state)), name="advance")
        self.circuit.add_command(
            wrap_command(jit(self.circuit.evolve)), name="evolve")

        @Context.dynamicCommand
        def clamp_input(x):
            z0.inputs.set(x)

        @Context.dynamicCommand
        def clamp_target(y):
            z3.inputs.set(y)

        @Context.dynamicCommand
        def clamp_mod_labels(labs):
            g2.contrastLabels.set(labs)
            ey.mask.set(labs)
            if self.learn_recon:
                eR.mask.set(labs)

    def save_to_disk(self, save_dir, params_only=False):
        if params_only:
            model_dir = "{}/{}/{}".format(self.exp_dir, self.model_name,
                                          save_dir)
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
        self.W2 = self.circuit.components.get("W2")
        self.M2 = self.circuit.components.get("M2")
        self.C3 = self.circuit.components.get("C3")
        self.V3y = self.circuit.components.get("V3y")
        self.R1 = self.circuit.components.get("R1")
        self.z0_prev = self.circuit.components.get("z0_prev")
        self.z2_prev = self.circuit.components.get("z2_prev")
        self.z3_prev = self.circuit.components.get("z3_prev")
        self.z0 = self.circuit.components.get("z0")
        self.z2 = self.circuit.components.get("z2")
        self.z3 = self.circuit.components.get("z3")
        self.zy = self.circuit.components.get("zy")
        self.zR = self.circuit.components.get("zR")
        self.tr0 = self.circuit.components.get("tr0")
        self.tr2 = self.circuit.components.get("tr2")

    def get_synapse_stats(self, param_name="W2"):
        _W = self.circuit.components.get(param_name).weights.value
        msg = "{}:\n  min {} ;  max {} \n  mu {} ;  norm {}".format(
            param_name, jnp.amin(_W), jnp.amax(_W), jnp.mean(_W),
            jnp.linalg.norm(_W)
        )
        return msg

    def viz_receptive_fields(
            self, param_name, field_shape, fname, transpose_params=False,
            n_fields_to_view=-1
    ):
        _W = self.circuit.components.get(param_name).weights.value
        if 0 < n_fields_to_view < _W.shape[1]:
            _W = _W[:, 0:n_fields_to_view]
        if transpose_params:
            _W = _W.T
        visualize([_W], [field_shape], fname)

    def process(self, Xb, Yb, dkey, adapt_synapses=False, collect_spikes=False,
                collect_rate_codes=False, lab_estimator="softmax",
                collect_recon=True, Xb_neg=None, Yb_neg=None):
        dkey, *subkeys = random.split(dkey, 2)
        if adapt_synapses:
            if self.algo_type == "supervised":
                Yb_neg = random.uniform(subkeys[0], Yb.shape, minval=0.,
                                        maxval=1.) * (1. - Yb)
                Yb_neg = nn.one_hot(jnp.argmax(Yb_neg, axis=1),
                                    num_classes=Yb.shape[1], dtype=jnp.float32)
                Xb_neg = Xb
            else:
                if Xb_neg is None:
                    bsize = Xb.shape[0]
                    _Xb = jnp.expand_dims(jnp.reshape(Xb, (bsize, 28, 28)), axis=3)
                    Xb_neg, Yb_neg = csdp_deform(subkeys[0], _Xb, Yb,
                                                 alpha=self.alpha,
                                                 use_rot=self.use_rot)
                    Xb_neg = jnp.reshape(jnp.squeeze(Xb_neg, axis=3),
                                         (bsize, 28 * 28))
            _Xb = jnp.concatenate((Xb, Xb_neg), axis=0)
            _Yb = jnp.concatenate((Yb, Yb_neg), axis=0)
            mod_signal = jnp.concatenate((jnp.ones((Xb.shape[0], 1)),
                                          jnp.zeros((Xb_neg.shape[0], 1))),
                                         axis=0)
        else:
            _Yb = Yb * 0
            _Xb = Xb
            mod_signal = jnp.ones((Xb.shape[0], 1))

        self.circuit.reset()

        batch_size = Xb.shape[0]
        if adapt_synapses:
            batch_size = batch_size * 2
        for name in self.traces:
            reset_trace(self.circuit.components.get(name), batch_size)
        for name in self.ecells:
            reset_errcell(self.circuit.components.get(name), batch_size)
        for name in self.gcells:
            reset_goodnesscell(self.circuit.components.get(name), batch_size)
        for name in self.input_cells:
            reset_bernoulli(self.circuit.components.get(name), batch_size)
        for name in self.lifs:
            reset_lif(self.circuit.components.get(name), batch_size)
        for name in self.ratecells:
            reset_ratecell(self.circuit.components.get(name), batch_size)
        if adapt_synapses:
            for name in self.hebb_synapses:
                reset_synapse(self.circuit.components.get(name), batch_size,
                              synapse_type="hebb")
            for name in self.csdp_synapses:
                reset_synapse(self.circuit.components.get(name), batch_size,
                              synapse_type="csdp")

        s0_mu = _Xb * 0
        y_count = 0.
        self.z3.inputs.set(_Yb)
        self.z3_prev.z.set(_Yb)
        T = self.T + 1
        R2 = 0.
        R3 = 0.
        for ts in range(1, T):
            self.circuit.clamp_input(_Xb)
            self.circuit.clamp_target(_Yb)
            self.circuit.clamp_mod_labels(mod_signal)
            self.circuit.advance(t=ts * self.dt, dt=self.dt)
            if adapt_synapses:
                self.circuit.evolve(t=ts * self.dt, dt=self.dt)

            y_count = self.zy.s.value + y_count
            if self.learn_recon or collect_recon:
                s0_mu = self.tr0.outputs.value + s0_mu
            if collect_rate_codes:
                R2 = self.z2.s.value + R2

        s0_mu = s0_mu / T

        if lab_estimator == "softmax":
            y_hat = softmax(y_count)
        else:
            y_hat = y_count
        if collect_rate_codes:
            R2 = R2 / T
            R3 = y_hat

        # возвращаю 0. вместо R1 чтобы сохранить количество элементов для распаковки
        return y_hat, y_count, 0., R2, R3, s0_mu