from jax import numpy as jnp, random, nn, jit
import sys, getopt as gopt, optparse, time

from csdp_model import CSDP_SNN as Model
## bring in ngc-learn analysis tools
from ngclearn.utils.metric_utils import measure_ACC, measure_CatNLL
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def measure_BCE(p, x, offset=1e-7, preserve_batch=False): ## binary cross-entropy
    p_ = jnp.clip(p, offset, 1 - offset)
    bce = -jnp.sum(x * jnp.log(p_) + (1.0 - x) * jnp.log(1.0 - p_),axis=1, keepdims=True)
    if preserve_batch is False:
        bce = jnp.mean(bce)
    return bce

def measure_MSE(mu, x, preserve_batch=False): ## mean squared error
    diff = mu - x
    se = jnp.square(diff) ## squared error
    mse = jnp.sum(se, axis=1, keepdims=True) # technically se at this point
    if preserve_batch is False:
        mse = jnp.mean(mse) # this is proper mse
    return mse
################################################################################

# read in general program arguments
options, remainder = gopt.getopt(
    sys.argv[1:], '',
    ["dataX=", "dataY=", "devX=", "devY=", "algo_type=", "num_iter=",
     "verbosity=", "seed=", "exp_dir=", "nZ1=", "nZ2="]
)
# external dataset arguments
nZ1 = 1024
nZ2 = 128
algo_type = "supervised" #"unsupervised"
seed = 1234
num_iter = 30 ## epochs
batch_size = 500
dev_batch_size = 1000
dataX = "data/mnist/trainX.npy"
dataY = "data/mnist/trainY.npy"
devX = "data/mnist/validX.npy"
devY = "data/mnist/validY.npy"
exp_dir = "exp"
verbosity = 0 ## verbosity level (0 - fairly minimal, 1 - prints multiple lines on I/O)
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

_X = jnp.load(dataX)
_Y = jnp.load(dataY)
print(_X, _X.shape)
Xdev = jnp.load(devX)
Ydev = jnp.load(devY)
x_dim = _X.shape[1]
patch_shape = (int(jnp.sqrt(x_dim)), int(jnp.sqrt(x_dim)))
y_dim = _Y.shape[1]

n_batches = int(_X.shape[0]/batch_size) ## get number batches (for progress reporting)
save_point = 5 ## save model params every epoch/iteration modulo "save_point"

## set up JAX seeding
dkey = random.PRNGKey(seed)
dkey, *subkeys = random.split(dkey, 10)

########################################################################
## configure model
hid_dim = nZ1
hid_dim2 = nZ2
out_dim = y_dim ## output dimensionality
learn_recon = True
T = 50
eta_w = 0.002  ## learning rate -- CSDP/hebbian update modulation
dt = 3. # ms ## integration time constant
########################################################################

########################################################################
## declare a simple test-time evaluation co-routine
def eval_model(model, Xdev, Ydev, batch_size, verbosity=1):
    ## evals model's test-time inference performance
    n_batches = int(Xdev.shape[0]/batch_size)

    n_samp_seen = 0
    nll = 0. ## negative Categorical log liklihood
    acc = 0. ## accuracy
    bce = 0. ## bin cross-entropy
    mse = 0. ## mean-squared error
    for j in range(n_batches):
        ## extract data block/batch
        idx = j * batch_size
        Xb = Xdev[idx: idx + batch_size, :]
        Yb = Ydev[idx: idx + batch_size, :]

        ## run model inference
        yMu, yCnt, _, _, _, xMu = model.process(
            Xb, Yb, dkey=dkey, adapt_synapses=False)
        ## record metric measurements (note: must also un-normalizing them here)
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
                (acc/n_samp_seen) * 100., nll/n_samp_seen, bce/n_samp_seen,
                mse/n_samp_seen), end=""
            )
    if verbosity > 0:
        print()
    ## produce final measurements
    nll = nll/(Xdev.shape[0])
    acc = acc/(Xdev.shape[0])
    bce = bce/(Xdev.shape[0])
    mse = mse/(Xdev.shape[0])
    return nll, acc, bce, mse
########################################################################

########################################################################

trAcc_set = []
trNll_set = []
acc_set = []
nll_set = []
bce_set = []
mse_set = []

sim_start_time = time.time() ## start time profiling




@jit
def measure_acc_nll(yMu, Yb): ## this is just a fast compound accuracy/NLL function
    mask = jnp.concatenate((jnp.ones((Yb.shape[0],1)),jnp.zeros((Yb.shape[0],1))), axis=0)
    N = jnp.sum(mask)
    _Yb = jnp.concatenate((Yb,Yb), axis=0) * mask
    offset = 1e-6
    _yMu = jnp.clip(yMu * mask, offset, 1.0 - offset)
    loss = -(_yMu * jnp.log(_yMu))
    nll = jnp.sum(jnp.sum(loss, axis=1, keepdims=True) * mask) * (1./N)

    guess = jnp.argmax(yMu, axis=1, keepdims=True)
    lab = jnp.argmax(_Yb, axis=1, keepdims=True)
    acc = jnp.sum( jnp.equal(guess, lab) * mask )/(N)
    return acc, nll

dkey, *subkeys = random.split(dkey, 3)
ptrs = random.permutation(subkeys[0],_X.shape[0])
X = _X[ptrs, :]
Y = _Y[ptrs, :]

    ## begin a single epoch/iteration
n_samp_seen = 0
tr_nll = 0.
tr_acc = 0.
space = {
    'A_plus': hp.uniform('A_plus', 0.001, 0.1),       # Скорость обучения при потенцировании (LTP)
    'A_minus': hp.uniform('A_minus', 0.001, 0.1),     # Скорость обучения при депрессии (LTD)# Временное окно для LTD (в миллисекундах)
}

n_samp_seen = 0
tr_nll = 0.
tr_acc = 0.
def epoh(param):
    n_samp_seen = 0
    tr_nll = 0.
    tr_acc = 0.
    dkey = random.PRNGKey(seed)
    dkey, *subkeys = random.split(dkey, 10)

    model = Model(subkeys[1], in_dim=x_dim, out_dim=y_dim, hid_dim=hid_dim, hid_dim2=hid_dim2,
                  batch_size=batch_size, eta_w=eta_w, T=T, dt=dt, algo_type=algo_type,
                  exp_dir=exp_dir, learn_recon=learn_recon, A_minus=param['A_minus'], A_plus=param['A_plus'])
    for j in range(n_batches):
        dkey, *subkeys = random.split(dkey, 2)
        ## sample mini-batch of patterns
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
        _tr_acc, _tr_nll = measure_acc_nll(yMu, Yb)  # compute masked scores
        tr_nll += _tr_nll * Yb.shape[0]  ## un-normalize score
        tr_acc += _tr_acc * Yb.shape[0]  ## un-normalize score
        n_samp_seen += Yb.shape[0]
    nll, acc, bce, mse = eval_model(
        model, Xdev, Ydev, batch_size=dev_batch_size, verbosity=verbosity
    )
    loss = 1 - acc
    return loss

best_params = fmin(
    fn=epoh, # Наша функция с SNN
    space=space,               # Пространство гиперпараметров
    algo=tpe.suggest,          # Алгоритм (Tree-structured Parzen Estimator)
    max_evals=10,             # Количество экспериментов (сколько раз запустится SNN)
)
print("\nЛучшие найденные параметры STDP:")
for key, value in best_params.items():
    print(f"{key}: {value:.4f}")