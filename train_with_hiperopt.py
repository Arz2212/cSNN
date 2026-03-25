import os
import sys, getopt as gopt, time

# КРИТИЧЕСКИ ВАЖНО: Отключаем предварительное выделение всей памяти GPU
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import jax
from jax import numpy as jnp, random, nn, jit
from csdp_model import CSDP_SNN as Model
from ngclearn.utils.metric_utils import measure_ACC, measure_CatNLL

import ray
from ray import train, tune
from ray.tune.search.hyperopt import HyperOptSearch


def measure_BCE(p, x, offset=1e-7, preserve_batch=False):
    p_ = jnp.clip(p, offset, 1 - offset)
    bce = -jnp.sum(x * jnp.log(p_) + (1.0 - x) * jnp.log(1.0 - p_), axis=1, keepdims=True)
    if preserve_batch is False:
        bce = jnp.mean(bce)
    return bce


def measure_MSE(mu, x, preserve_batch=False):
    diff = mu - x
    se = jnp.square(diff)
    mse = jnp.sum(se, axis=1, keepdims=True)
    if preserve_batch is False:
        mse = jnp.mean(mse)
    return mse


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


def train_model(config):
    _X = jnp.load(config["dataX"])
    _Y = jnp.load(config["dataY"])
    Xdev = jnp.load(config["devX"])
    Ydev = jnp.load(config["devY"])

    x_dim = _X.shape[1]
    y_dim = _Y.shape[1]
    n_batches = int(_X.shape[0] / config["batch_size"])

    dkey = random.PRNGKey(config["seed"])
    dkey, *subkeys = random.split(dkey, 3)
    ptrs = random.permutation(subkeys[0], _X.shape[0])
    X = _X[ptrs, :]
    Y = _Y[ptrs, :]

    model = Model(
        subkeys[1], in_dim=x_dim, out_dim=y_dim,
        hid_dim=config["nZ1"], hid_dim2=config["nZ2"],
        batch_size=config["batch_size"], eta_w=config["eta_w"],
        T=config["T"], dt=config["dt"], algo_type=config["algo_type"],
        exp_dir=config["exp_dir"], learn_recon=config["learn_recon"],
        A_m=config["A_minus"], A_p=config["A_plus"]
    )

    tr_nll, tr_acc, n_samp_seen = 0., 0., 0

    # Обучение
    for j in range(n_batches):
        dkey, *subkeys = random.split(dkey, 2)
        idx = j * config["batch_size"]
        s_ptr = idx
        e_ptr = min(idx + config["batch_size"], X.shape[0])

        Xb = X[s_ptr: e_ptr, :]
        Yb = Y[s_ptr: e_ptr, :]

        yMu, yCnt, _, _, _, x_mu = model.process(
            Xb, Yb, dkey=dkey, adapt_synapses=True, collect_rate_codes=True
        )
        _tr_acc, _tr_nll = measure_acc_nll(yMu, Yb)
        tr_nll += _tr_nll * Yb.shape[0]
        tr_acc += _tr_acc * Yb.shape[0]
        n_samp_seen += Yb.shape[0]

    # Эвалюация
    nll, acc, bce, mse = eval_model(
        model, Xdev, Ydev,
        batch_size=config["dev_batch_size"],
        dkey=dkey,
        verbosity=config["verbosity"]
    )

    loss = 1.0 - float(acc)

    train.report({"loss": loss, "acc": float(acc), "nll": float(nll)})



if __name__ == "__main__":
    options, remainder = gopt.getopt(
        sys.argv[1:], '',
        ["dataX=", "dataY=", "devX=", "devY=", "algo_type=", "num_iter=",
         "verbosity=", "seed=", "exp_dir=", "nZ1=", "nZ2="]
    )

    # 1. Получаем АБСОЛЮТНЫЙ путь к папке проекта
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

    # 2. Собираем полные пути к данным
    dataX_path = os.path.join(PROJECT_DIR, "data", "mnist", "trainX.npy")
    dataY_path = os.path.join(PROJECT_DIR, "data", "mnist", "trainY.npy")
    devX_path = os.path.join(PROJECT_DIR, "data", "mnist", "validX.npy")
    devY_path = os.path.join(PROJECT_DIR, "data", "mnist", "validY.npy")

    # 3. ПРОВЕРКА: Убеждаемся, что файлы реально существуют по этим путям
    if not os.path.exists(dataX_path):
        print(f"\n[КРИТИЧЕСКАЯ ОШИБКА] Файл не найден: {dataX_path}")
        print("Проверьте, что папка 'data' лежит в той же директории, что и этот скрипт.")
        sys.exit(1)

    # Базовые константы с жестко прописанными абсолютными путями
    base_config = {
        "nZ1": 3000, "nZ2": 600, "algo_type": "supervised", "seed": 1234,
        "num_iter": 10, "batch_size": 500, "dev_batch_size": 1000,
        "dataX": dataX_path,
        "dataY": dataY_path,
        "devX": devX_path,
        "devY": devY_path,
        "exp_dir": "exp_supervised_mnist", "verbosity": 0, "learn_recon": True,
        "T": 50, "eta_w": 0.002, "dt": 3.
    }

    for opt, arg in options:
        opt_name = opt.replace("--", "")
        if opt_name in ["num_iter", "seed", "verbosity", "nZ1", "nZ2"]:
            base_config[opt_name] = int(arg.strip())
        else:
            base_config[opt_name] = arg.strip()

    # Убираем working_dir, чтобы не мучить Ray копированием гигабайтов данных
    runtime_env = {
        "env_vars": {
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "JAX_TRACEBACK_FILTERING": "off"
        }
    }
    ray.init(runtime_env=runtime_env)

    search_space = {
        **base_config,
        "A_plus": tune.uniform(0.001, 1.0),
        "A_minus": tune.uniform(0.001, 1.0)
    }

    from ray.tune.search.hyperopt import HyperOptSearch

    algo = HyperOptSearch(metric="loss", mode="min")
    sim_start_time = time.time()

    tuner = tune.Tuner(
        # Используем мощь вашего Ryzen и RTX 3090
        tune.with_resources(train_model, resources={"cpu": 3, "gpu": 0.25}),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            search_alg=algo,
            num_samples=100,
            max_concurrent_trials=4,  # 4 воркера одновременно
        ),
        param_space=search_space,
    )

    results = tuner.fit()
    best_result = results.get_best_result()

    print("\nЛучшие найденные параметры STDP:")
    print(f"A_plus: {best_result.config['A_plus']:.4f}")
    print(f"A_minus: {best_result.config['A_minus']:.4f}")
    print(f"Лучший Loss: {best_result.metrics['loss']:.4f}")
    print(f"\nОбщее время: {time.time() - sim_start_time:.2f} сек.")

    ray.shutdown()