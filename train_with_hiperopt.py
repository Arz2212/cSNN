import os

# КРИТИЧЕСКИ ВАЖНО: Отключаем жадное резервирование памяти JAX.
# Иначе главный процесс займет весь GPU, и дочерние процессы упадут.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import multiprocessing as mp
import sys, getopt as gopt, time, gc
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

import jax
from jax import numpy as jnp, random, nn, jit
from csdp_model import CSDP_SNN as Model

## bring in ngc-learn analysis tools
from ngclearn.utils.metric_utils import measure_ACC, measure_CatNLL


def measure_BCE(p, x, offset=1e-7, preserve_batch=False):  ## binary cross-entropy
    p_ = jnp.clip(p, offset, 1 - offset)
    bce = -jnp.sum(x * jnp.log(p_) + (1.0 - x) * jnp.log(1.0 - p_), axis=1, keepdims=True)
    if preserve_batch is False:
        bce = jnp.mean(bce)
    return bce


def measure_MSE(mu, x, preserve_batch=False):  ## mean squared error
    diff = mu - x
    se = jnp.square(diff)  ## squared error
    mse = jnp.sum(se, axis=1, keepdims=True)
    if preserve_batch is False:
        mse = jnp.mean(mse)
    return mse


@jit
def measure_acc_nll(yMu, Yb):  ## this is just a fast compound accuracy/NLL function
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


# read in general program arguments
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
verbosity = 0  ## verbosity level

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
        print(nZ1)
    elif opt in ("--nZ2"):
        nZ2 = int(arg.strip())
    elif opt in ("--exp_dir"):
        exp_dir = arg.strip()

print("#####################################################")
print("Train-set: X: {} | Y: {}".format(dataX, dataY))
print("  Dev-set: X: {} | Y: {}".format(devX, devY))
print("#####################################################")

# Загрузка данных происходит на глобальном уровне,
# каждый процесс будет иметь к ним доступ (в spawn они подгрузятся заново, что безопасно для JAX)
_X = jnp.load(dataX)
_Y = jnp.load(dataY)
print(_X, _X.shape)
Xdev = jnp.load(devX)
Ydev = jnp.load(devY)
x_dim = _X.shape[1]
patch_shape = (int(jnp.sqrt(x_dim)), int(jnp.sqrt(x_dim)))
y_dim = _Y.shape[1]

n_batches = int(_X.shape[0] / batch_size)
save_point = 5

## configure model
hid_dim = nZ1
hid_dim2 = nZ2
out_dim = y_dim
learn_recon = True
T = 50
eta_w = 0.002
dt = 3.


def eval_model(model, Xdev, Ydev, batch_size, dkey, verbosity=1):
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

    nll = nll / (Xdev.shape[0])
    acc = acc / (Xdev.shape[0])
    bce = bce / (Xdev.shape[0])
    mse = mse / (Xdev.shape[0])
    return nll, acc, bce, mse


# ========================================================================
# 1. Функция-рабочий, которая будет выполняться в изолированном процессе
# ========================================================================
def run_trial(param, return_dict):
    dkey = random.PRNGKey(seed)
    dkey, *subkeys = random.split(dkey, 3)
    ptrs = random.permutation(subkeys[0], _X.shape[0])
    X = _X[ptrs, :]
    Y = _Y[ptrs, :]

    n_samp_seen = 0
    tr_nll = 0.
    tr_acc = 0.

    model = Model(subkeys[1], in_dim=x_dim, out_dim=y_dim, hid_dim=hid_dim, hid_dim2=hid_dim2,
                  batch_size=batch_size, eta_w=eta_w, T=T, dt=dt, algo_type=algo_type,
                  exp_dir=exp_dir, learn_recon=learn_recon, A_m=param['A_minus'], A_p=param['A_plus'])

    for j in range(n_batches // 2000):
        print(1)
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

    # Передаем dkey в eval_model (раньше он брался из глобальной области)
    nll, acc, bce, mse = eval_model(
        model, Xdev, Ydev, batch_size=dev_batch_size, dkey=dkey, verbosity=verbosity
    )

    loss = 1.0 - float(acc)
    print(f"Trial done: Loss={loss:.4f}, Acc={acc:.4f}")

    # Записываем результат в разделяемый словарь
    return_dict['loss'] = loss

    # Процесс сам завершится и очистит за собой всю память


# ========================================================================
# 2. Обертка для Hyperopt, запускающая рабочий процесс
# ========================================================================
def epoh(param):
    manager = mp.Manager()
    return_dict = manager.dict()

    # Создаем и запускаем изолированный процесс
    p = mp.Process(target=run_trial, args=(param, return_dict))
    p.start()
    p.join()  # Ждем, пока процесс закончит работу

    # Если процесс упал (например, баг внутри ngcsimlib или OOM), Hyperopt не сломается
    if p.exitcode != 0:
        print(f"\n[ОШИБКА] Процесс упал на параметрах: {param}")
        return {'loss': 1.0, 'status': STATUS_OK}  # Возвращаем плохой loss, чтобы алгоритм искал дальше

    # Возвращаем найденный loss
    return {'loss': return_dict.get('loss', 1.0), 'status': STATUS_OK}


space = {
    'A_plus': hp.uniform('A_plus', 0.001, 0.1),
    'A_minus': hp.uniform('A_minus', 0.001, 0.1)
}

if __name__ == "__main__":
    # Устанавливаем метод запуска процессов 'spawn' (надежнее всего работает с CUDA/JAX)
    mp.set_start_method('spawn', force=True)

    sim_start_time = time.time()

    best_params = fmin(
        fn=epoh,
        space=space,
        algo=tpe.suggest,
        max_evals=10,
    )

    print("\nЛучшие найденные параметры STDP:")
    for key, value in best_params.items():
        print(f"{key}: {value:.4f}")

    print(f"\nОбщее время: {time.time() - sim_start_time:.2f} сек.")