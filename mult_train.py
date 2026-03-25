import os

# ЗАПРЕЩАЕМ JAX ЗАБИРАТЬ ВСЮ ПАМЯТЬ СРАЗУ (Обязательно до импорта JAX!)
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"


from jax import numpy as jnp, random, nn, jit
import sys, getopt as gopt, optparse, time

from csdp_model import CSDP_SNN as Model
## bring in ngc-learn analysis tools
from ngclearn.utils.metric_utils import measure_ACC, measure_CatNLL
import ray
from ray import tune
from ray import train


def measure_BCE(p, x, offset=1e-7, preserve_batch=False):  ## binary cross-entropy
    p_ = jnp.clip(p, offset, 1 - offset)
    bce = -jnp.sum(x * jnp.log(p_) + (1.0 - x) * jnp.log(1.0 - p_), axis=1, keepdims=True)
    if preserve_batch is False:
        bce = jnp.mean(bce)
    return bce


def measure_MSE(mu, x, preserve_batch=False):  ## mean squared error
    diff = mu - x
    se = jnp.square(diff)  ## squared error
    mse = jnp.sum(se, axis=1, keepdims=True)  # technically se at this point
    if preserve_batch is False:
        mse = jnp.mean(mse)  # this is proper mse
    return mse


options, remainder = gopt.getopt(
    sys.argv[1:], '',
    ["dataX=", "dataY=", "devX=", "devY=", "algo_type=", "num_iter=",
     "verbosity=", "seed=", "exp_dir=", "nZ1=", "nZ2="]
)

# external dataset arguments
nZ1 = 1024
nZ2 = 128
algo_type = "supervised"  # "unsupervised"
seed = 1234
num_iter = 30  ## epochs
batch_size = 500
dev_batch_size = 1000
dataX = "data/mnist/trainX.npy"
dataY = "data/mnist/trainY.npy"
devX = "data/mnist/validX.npy"
devY = "data/mnist/validY.npy"
exp_dir = "exp"
verbosity = 0  ## verbosity level (0 - fairly minimal, 1 - prints multiple lines on I/O)

for opt, arg in options:
    if opt in ("--dataX"):
        dataX = arg.strip()
    elif opt in ("--dataY"):
        dataY = arg.strip()
    elif opt in ("--devX"):
        devX = arg.strip()
    elif opt in ("--devY"):
        devY = arg.strip()
    elif opt in ("--algo_type"):
        algo_type = arg.strip()
    elif opt in ("--num_iter"):
        num_iter = int(arg.strip())
    elif opt in ("--seed"):
        seed = int(arg.strip())
    elif opt in ("--verbosity"):
        verbosity = int(arg.strip())
    elif opt in ("--nZ1"):
        nZ1 = int(arg.strip())
    elif opt in ("--nZ2"):
        nZ2 = int(arg.strip())
    elif opt in ("--exp_dir"):
        exp_dir = arg.strip()

# Загружаем данные
_X_local = jnp.load(dataX)
_Y_local = jnp.load(dataY)
Xdev_local = jnp.load(devX)
Ydev_local = jnp.load(devY)

x_dim = _X_local.shape[1]
patch_shape = (int(jnp.sqrt(x_dim)), int(jnp.sqrt(x_dim)))
y_dim = _Y_local.shape[1]

n_batches = int(_X_local.shape[0] / batch_size)
save_point = 5

## set up JAX seeding
dkey = random.PRNGKey(seed)
dkey, *subkeys = random.split(dkey, 10)

########################################################################
## configure model
hid_dim = nZ1
hid_dim2 = nZ2
out_dim = y_dim
learn_recon = True
T = 50
eta_w = 0.002
dt = 3.


@jit
def measure_acc_nll(yMu, Yb):
    mask = jnp.concatenate((jnp.ones((Yb.shape[0], 1)), jnp.zeros((Yb.shape[0], 1))), axis=0)
    N = jnp.sum(mask)
    _Yb = jnp.concatenate((Yb, Yb), axis=0) * mask
    offset = 1e-6
    _yMu = jnp.clip(yMu * mask, offset, 1.0 - offset)
    loss = -(_yMu * jnp.log(_yMu))
    nll = jnp.sum(jnp.sum(loss, axis=1, keepdims=True) * mask) * (1. / N)

    guess = jnp.argmax(yMu, axis=1, keepdims=True)
    lab = jnp.argmax(_Yb, axis=1, keepdims=True)
    acc = jnp.sum(jnp.equal(guess, lab) * mask) / (N)
    return acc, nll


def eval_model(model, Xdev, Ydev, batch_size, dkey, verbosity=0):
    n_batches = int(Xdev.shape[0] / batch_size)
    n_samp_seen = 0
    nll = 0.
    acc = 0.
    bce = 0.
    mse = 0.
    for j in range(n_batches):
        idx = j * batch_size
        Xb = Xdev[idx: idx + batch_size, :]
        Yb = Ydev[idx: idx + batch_size, :]

        yMu, yCnt, _, _, _, xMu = model.process(
            Xb, Yb, dkey=dkey, adapt_synapses=False)

        _nll = measure_CatNLL(yMu, Yb) * Xb.shape[0]
        _acc = measure_ACC(yMu, Yb) * Yb.shape[0]
        _bce = measure_BCE(xMu, Xb, preserve_batch=False) * Xb.shape[0]
        _mse = measure_MSE(xMu, Xb, preserve_batch=False) * Xb.shape[0]
        nll += _nll
        acc += _acc
        bce += _bce
        mse += _mse

        n_samp_seen += Yb.shape[0]
        if verbosity > 0:
            print("\r Eval.Step:  Acc = {:.3f}; NLL = {:.5f}; CE = {:.5f}; MSE = {:.5f} ".format(
                (acc / n_samp_seen) * 100., nll / n_samp_seen, bce / n_samp_seen,
                mse / n_samp_seen), end=""
            )
    if verbosity > 0:
        print()

    return nll / Xdev.shape[0], acc / Xdev.shape[0], bce / Xdev.shape[0], mse / Xdev.shape[0]


# --- ИЗМЕНЕНИЕ: Обновленная функция обучения, принимающая ссылки на данные ---
def epoh(conf):
    # Извлекаем ссылки на данные из конфига и получаем сами данные (Ray сделает это моментально из общей памяти)
    _X = ray.get(conf["_X_ref"])
    _Y = ray.get(conf["_Y_ref"])
    Xdev = ray.get(conf["Xdev_ref"])
    Ydev = ray.get(conf["Ydev_ref"])

    # Инициализация ключей внутри процесса
    dkey = random.PRNGKey(conf['seed'])
    dkey, *subkeys = random.split(dkey, 10)
    dkey, *subkeys = random.split(dkey, 3)
    ptrs = random.permutation(subkeys[0], _X.shape[0])
    X = _X[ptrs, :]
    Y = _Y[ptrs, :]

    n_samp_seen = 0
    tr_nll = 0.
    tr_acc = 0.
    model = Model(subkeys[1], in_dim=x_dim, out_dim=y_dim, hid_dim=hid_dim, hid_dim2=hid_dim2,
                  batch_size=batch_size, eta_w=eta_w, T=T, dt=dt, algo_type=algo_type,
                  exp_dir=exp_dir, learn_recon=learn_recon, A_m=conf['A_minus'], A_p=conf['A_plus'])

    for j in range(n_batches):
        dkey, *subkeys = random.split(dkey, 2)
        idx = j * batch_size
        s_ptr = idx
        e_ptr = idx + batch_size
        if e_ptr > X.shape[0]:
            e_ptr = X.shape[0]
        Xb = X[s_ptr: e_ptr, :]
        Yb = Y[s_ptr: e_ptr, :]

        yMu, yCnt, _, _, _, x_mu = model.process(
            Xb, Yb, dkey=dkey, adapt_synapses=True, collect_rate_codes=True
        )
        _tr_acc, _tr_nll = measure_acc_nll(yMu, Yb)
        tr_nll += _tr_nll * Yb.shape[0]
        tr_acc += _tr_acc * Yb.shape[0]
        n_samp_seen += Yb.shape[0]

    # --- ИЗМЕНЕНИЕ: Добавили обязательный параметр dkey=dkey ---
    nll, acc, bce, mse = eval_model(
        model, Xdev, Ydev, batch_size=dev_batch_size, dkey=dkey, verbosity=verbosity
    )

    tune.report({"score": 1 - float(acc)})


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    # --- ИЗМЕНЕНИЕ: Кладем тяжелые массивы в хранилище объектов Ray ---
    _X_ref = ray.put(_X_local)
    _Y_ref = ray.put(_Y_local)
    Xdev_ref = ray.put(Xdev_local)
    Ydev_ref = ray.put(Ydev_local)

    # Передаем ссылки на объекты в search_space
    search_space = {
        "A_minus": tune.uniform(0, 1),
        "A_plus": tune.uniform(0, 1),
        # Статичные параметры передаются так же, как и гиперпараметры
        "_X_ref": _X_ref,
        "_Y_ref": _Y_ref,
        "Xdev_ref": Xdev_ref,
        "Ydev_ref": Ydev_ref,
        "seed": seed  # Передаем сид для правильной рандомизации в каждом процессе
    }

    trainable_with_resources = tune.with_resources(
        epoh,
        resources={
            "cpu": 3,
            "gpu": 0.2
        }
    )

    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="score",
            mode="min",
            num_samples=100,
            max_concurrent_trials=5
        ),
    )

    results = tuner.fit()

    best_result = results.get_best_result("score", "min")

    print("\n" + "=" * 40)
    print("Лучшие найденные параметры:")
    print(f"A_minus: {best_result.config['A_minus']:.4f}")
    print(f"A_plus: {best_result.config['A_plus']:.4f}")
    print(f"Минимальный loss (score): {best_result.metrics['score']:.4f}")
    print("=" * 40)