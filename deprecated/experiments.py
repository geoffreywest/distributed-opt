import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from scipy.special import expit
import pandas as pd
import multiprocessing
import concurrent.futures


## Generating data

def generate_features(n):
    # Dimension of features
    p = 25

    # Mean for each feature
    mean = np.zeros(p - 1)

    # Covariance matrix
    cov = np.asmatrix(np.diag(10 / np.array(range(1, p)) ** 2))

    # Generate mulivariate normal data
    X = np.random.multivariate_normal(mean, cov, n)
    X = np.concatenate((X, np.ones((n, 1))), axis=1)
    return X


def generate_distribution_params():
    # Dimension of features
    p = 25

    # Mean for the linear weight vector
    mean = np.zeros(p)

    # Covariance matrix for entries in the weight vector
    cov = np.eye(p)

    # Generate weight vectors
    w1 = np.random.multivariate_normal(mean, cov)
    w2 = np.random.multivariate_normal(mean, cov)

    return w1, w2


def generate_labels(X):
    # Generate the parameters of the label-generating distribution
    w1, w2 = generate_distribution_params()

    # Obtain minimum of two linear scores
    scores1 = X * np.asmatrix(w1).T
    scores2 = X * np.asmatrix(w2).T
    min_scores = np.min([scores1, scores2], axis=0)

    # Apply sigmoid function
    prob = expit(min_scores)

    # Generate Bernoulli random labels
    labels = np.random.binomial(1, prob)

    return labels


def generate_data(n):
    # Generate features
    X = generate_features(n)

    # Generate labels
    y = generate_labels(X)

    return X, y







#############################################################################################


## Logistic Regression


def log_likelihood(X, w, y):
    # Compute softmax predictions
    scores = np.dot(X, w)
    prob = expit(scores)

    # Log likelihood = 1/N * sum(log p(y_i | x_i))
    log_lik = (y.T * np.log(prob) + (1 - y).T * np.log(1 - prob)) / X.shape[0]

    return np.mean(log_lik)


def objective(X, w, y):
    log_loss = -log_likelihood(X, w, y)
    l2_norm = np.linalg.norm(w)
    return log_loss, l2_norm


def gradient(X, w, y):
    # Compute softmax predictions
    scores = np.dot(X, w)
    prob = np.asmatrix(expit(scores))

    # Compute the gradient of the log loss
    grad = np.dot(np.transpose(X), prob - y) / X.shape[0]

    # Compute the gradient of the l2 norm
    l2_grad = 2 * w

    return grad, l2_grad





#############################################################################################


## Federated algorithms

class DataSource:
    """
    Acts like a "button" for random sample data. Reads in the entire sample dataset and stores it locally.
    Then returns data randomly to each machine to prevent re-use of the same data.
    """

    def __init__(self, randX, randY):
        self._X = randX
        self._Y = randY
        self._num_used = 0

    def generate_observations(self, count):
        X = self._X[self._num_used:self._num_used + count]
        Y = self._Y[self._num_used:self._num_used + count]
        self._num_used += count
        return (X, Y)


class Machine:
    """
    Represents an invididual machine in the network. We will instantiate it with some data,
    ask it to perform some gradient updates and return the results, then forget about this
    machine and the data that was given to it.
    """

    def __init__(self, source, data_count):
        X_local, Y_local = source.generate_observations(data_count)
        self._X = X_local  # Locally stored predictor data
        self._Y = Y_local  # Locally stored responder data
        self._m = data_count

    def execute_inner_sgd(self, eta_inner, w_cur):
        w = np.copy(w_cur)
        grads = []
        for i in range(self._m):
            # Execute the local gradient step:
            grad, _ = gradient(self._X[[i]], w, self._Y[[i]])
            w = w - eta_inner * grad

            grads.append(grad)

        # Return the sum of observed gradients
        return np.sum(grads, axis=0)

    def execute_pipelined_DSVRG(self, w_cur, w_prev, g_prev, eta):
        # Iterate m steps of variance-reduced SGD on the local data
        w = np.copy(w_cur)
        for i in range(self._m):
            # Compute the gradient adjustment for the current iterate:
            # g_(t-1) - nabla f_(k,i)(w_(t-1))
            grad_adjustment = g_prev - gradient(self._X[[i]], w_prev, self._Y[[i]])[0]

            # Execute the variance reduced gradient step:
            w = w - eta * (gradient(self._X[[i]], w, self._Y[[i]])[0] + grad_adjustment)

            # TODO: possibly add gamma shrinkage term

        # Compute the local gradient at the iterate where we started the round
        g_local, _ = gradient(self._X, w_cur, self._Y)
        return w, g_local


#############################################################################################

## Experiments

def run_rounds_counting(arg):
    global results_round_1

    eta = arg['eta']
    q = arg['q']
    eps = arg['eps']
    id = arg['id']

    randX, randY = rand_data[id]

    # Create a DataSource class to avoid repeated use of data
    source = DataSource(randX, randY)

    eta_outer = eta
    eta_inner = eta * q if arg['opt'] == 'inner/outer' else eta

    w0 = np.asmatrix(np.zeros(X.shape[1])).T
    ws = [w0]
    g_prev = w0

    if arg['opt'] == 'pdsvrg':
        ws.append(w0)

    log_losses = []
    l2_norms = []

    log_loss, l2_norm = objective(X, w0, y)
    log_losses.append(log_loss)
    l2_norms.append(l2_norm)

    R = 0
    while log_losses[-1] - best_val > eps:
        if R % 100 == 0:
            print(R)
            print(log_losses[-1] - best_val)
        if R >= 500:
            break
        w_cur = ws[-1]

        # Generate M random machines to perform the Local SGD steps
        inner_results = []
        if arg['opt'] == 'inner/outer':
            inner_results = [
                Machine(source, data_count=K).execute_inner_sgd(eta_inner, w_cur)
                for _ in range(M)]
            w_next = w_cur - eta_outer / M * np.sum(inner_results, axis=0)
        elif arg['opt'] == 'pdsvrg':
            k = int(K/3) if q else K
            inner_results = [
                Machine(source, data_count=k).execute_pipelined_DSVRG(w_cur, ws[-2], g_prev, eta)
                for _ in range(M)]

            w_next = np.asmatrix(np.mean([w for (w, _) in inner_results], axis=0))
            g_prev = np.asmatrix(np.mean([g for (_, g) in inner_results], axis=0))

        # Perform the outer gradient step
        ws.append(w_next)

        # Record progress on the objective
        in_mean = min(len(ws), 10)
        w_mean = np.asmatrix(np.mean(ws[-in_mean:], axis=0))
        log_loss, l2_norm = objective(X, w_mean, y)
        log_losses.append(log_loss)
        l2_norms.append(l2_norm)

        R += 1
    print(f'-- {R}')
    key = gen_key(arg)
    results_round_1[key] = R
    return R


def optimize_param(X, y, lmbda=0, eta=5e-2):
    M = 100  # Number of machines in each round
    K = 10   # Number of stochastic gradients to calculate on each machine
    R = 5000 # Number of rounds

    w0 = np.asmatrix(np.zeros(X.shape[1])).T

    w = w0
    next_idx = 0
    log_losses = []
    l2_norms = []

    log_loss, l2_norm = objective(X, w, y)
    log_losses.append(log_loss)
    l2_norms.append(l2_norm)

    for r in range(R):
        if r % (R / 10) == 0:
            print(f'{log_losses[-1]}')
            pass

        next_idx = next_idx % X.shape[0]
        X_batch = X[next_idx:next_idx + K * M]
        y_batch = y[next_idx:next_idx + K * M]
        next_idx += K * M
        next_idx = next_idx % X.shape[0]

        grad, l2_grad = gradient(X_batch, w, y_batch)
        w = w - eta * K * (grad + lmbda / 2 * l2_grad)

        log_loss, l2_norm = objective(X, w, y)
        log_losses.append(log_loss)
        l2_norms.append(l2_norm)

    return log_losses[-1]



#############################################################################################


def callback(args):
    print('Beginning callback...')
    res = run_rounds_counting(args)
    print('Completed callback...')
    return res


def get_data():
    df = pd.read_csv('data.csv', index_col=0)
    y = np.asmatrix(df['y']).T
    cols = [c for c in df.columns if c != 'y']
    X = np.asmatrix(df[cols].values)
    return X, y

M = 10
K = 90
N_random = 10

rand_data = {}
results_round_1 = {}
results_round_2 = {}

X = None
y = None
best_val = 0


def pipelined_dsvrg_routine():
    global X
    global y
    global rand_data
    global results_round_1
    global results_round_2
    global M
    global K
    global best_val
    global N_random

    start = time.time()

    n = 50000
    #X, y = generate_data(n)
    print(f'Data distribution: {np.sum(y)}/{n}')

    for i in range(2 * N_random):
        indices = np.random.randint(0, n, 500_000)
        randX = X[indices]
        randY = y[indices]
        rand_data[i] = randX, randY

    suboptimalities = .01 / np.arange(1, 11)

    grids = {
        True: np.linspace(.3, .5, 11),
        False: np.linspace(.3, .4, 6),
    }
    equate_gradients = [True, False]

    processes = []
    args_list = []

    for id in range(N_random):
        for eq in equate_gradients:
            for eps in suboptimalities:
                for eta in grids[eq]:
                    args = {
                        'opt': 'pdsvrg',
                        'q': eq,
                        'eta': eta,
                        'id': id,
                        'eps': eps,
                        'M': M,
                        'K': K,
                    }
                    #p = multiprocessing.Process(target=callback, args=[args])
                    #processes.append(p)
                    args_list.append(args)


    print(f'Processes to start: {len(processes)}')
    N_to_run = len(args_list)

    for arg in args_list[:N_to_run]:
        #callback(args)
        f = concurrent.futures.ProcessPoolExecutor(max_workers=4).submit(callback, arg)
        processes.append(f)

    for arg, f in list(zip(args_list, processes))[:N_to_run]:
        key = gen_key(arg)
        results_round_1[key] = f.result()

    optimal_steps = {}

    for eq in equate_gradients:
        for eps in suboptimalities:
            a = len(grids[eq])
            b = N_random
            counts = np.zeros((a,b))
            for i in range(a):
                for j in range(b):
                    key = gen_key({
                        'opt': 'pdsvrg',
                        'q': eq,
                        'eta': grids[eq][i],
                        'id': j,
                        'eps': eps,
                        'M': M,
                        'K': K,
                    })
                    counts[i][j] = results_round_1[key]
            best_steps = np.mean(counts, axis=1)
            best_step = np.argmin(best_steps)
            optimal_steps[eq,eps] = grids[eq][best_step]

    mid = time.time()

    args_list_2 = []
    processes_2 = []

    for i in range(N_random):
        for eq in equate_gradients:
            for eps in suboptimalities:
                eta = optimal_steps[eq,eps]
                args_list_2.append({
                    'opt': 'pdsvrg',
                    'q': eq,
                    'eta': eta,
                    'id': N_random + i,
                    'eps': eps,
                    'M': M,
                    'K': K,
                })

    for arg in args_list_2:
        f = concurrent.futures.ProcessPoolExecutor().submit(callback, arg)
        processes_2.append(f)

    for arg, f in zip(args_list_2, processes_2):
        key = gen_key(arg)
        results_round_2[key] = f.result()

    aggregate_results = {}

    for eq in equate_gradients:
        for eps in suboptimalities:
            vals = []
            for i in range(N_random):
                eta = optimal_steps[eq,eps]
                key = gen_key({
                    'opt': 'pdsvrg',
                    'q': eq,
                    'eta': eta,
                    'id': N_random + i,
                    'eps': eps,
                    'M': M,
                    'K': K,
                })
                vals.append(results_round_2[key])
            aggregate_results[eq,eps] = np.mean(vals)

    finish = time.time()
    print(len(results_round_1))
    for k, v in results_round_1.items():
        print(k,v)

    steps = []
    for k, v in optimal_steps.items():
        print(k, v)
        steps.append(k + (v,))

    print('---')
    counts = []
    for k, v in aggregate_results.items():
        print(k, v)
        counts.append(k + (v,))

    df1 = pd.DataFrame.from_records(steps)
    df2 = pd.DataFrame.from_records(counts)
    df1.to_csv(f'steps_pipelined_{M}_{K}.csv', header=True)
    df2.to_csv(f'counts_pipelined_{M}_{K}.csv', header=True)

    print(f'Time elapsed phase 1: {mid-start}')
    print(f'Time elapsed phase 2: {finish-mid}')


def inner_outer_routine():
    global X
    global y
    global rand_data
    global results_round_1
    global results_round_2
    global M
    global K
    global best_val
    global N_random

    start = time.time()

    q_params = [0, .25, .5, 1]

    suboptimalities = .01 / np.arange(1, 11)

    grids = {
        0: np.linspace(.02, .05, 8),
        .25: np.linspace(.04, .12, 9),
        .5: np.linspace(.1, .22, 11),
        1: np.linspace(.05, .16, 9)
    }

    #for q in grids.keys():
    #    grids[q] = grids[q][:5]

    processes = []
    args_list = []

    for id in range(N_random):
        for q in q_params:
            for eps in suboptimalities:
                for eta in grids[q]:
                    args = {
                        'opt': 'inner/outer',
                        'q': q,
                        'eta': eta,
                        'id': id,
                        'eps': eps,
                        'M': M,
                        'K': K,
                    }
                    #p = multiprocessing.Process(target=callback, args=[args])
                    #processes.append(p)
                    args_list.append(args)

    print(f'Processes to start: {len(args_list)}')
    #for p in processes[:2]:
        #p.start()

    #for p in processes[:2]:
        #p.join()

    N_to_run = len(args_list)

    #print(len(args_list))
    #return

    for arg in args_list[:N_to_run]:
        #callback(args)
        f = concurrent.futures.ProcessPoolExecutor(max_workers=4).submit(callback, arg)
        processes.append(f)

    for arg, f in list(zip(args_list, processes))[:N_to_run]:
        key = gen_key(arg)
        results_round_1[key] = f.result()

    optimal_steps = {}

    for q in q_params:
        for eps in suboptimalities:
            a = len(grids[q])
            b = N_random
            counts = np.zeros((a,b))
            for i in range(a):
                for j in range(b):
                    key = gen_key({
                        'opt': 'inner/outer',
                        'q': q,
                        'eta': grids[q][i],
                        'id': j,
                        'eps': eps,
                        'M': M,
                        'K': K,
                    })
                    counts[i][j] = results_round_1[key]
            best_steps = np.mean(counts, axis=1)
            best_step = np.argmin(best_steps)
            optimal_steps[q,eps] = grids[q][best_step]

    mid = time.time()

    args_list_2 = []
    processes_2 = []

    for i in range(N_random):
        for q in q_params:
            for eps in suboptimalities:
                eta = optimal_steps[q,eps]
                args_list_2.append({
                    'opt': 'inner/outer',
                    'q': q,
                    'eta': eta,
                    'id': N_random + i,
                    'eps': eps,
                    'M': M,
                    'K': K,
                })

    for arg in args_list_2:
        f = concurrent.futures.ProcessPoolExecutor().submit(callback, arg)
        processes_2.append(f)

    for arg, f in zip(args_list_2, processes_2):
        key = gen_key(arg)
        results_round_2[key] = f.result()

    aggregate_results = {}

    for q in q_params:
        for eps in suboptimalities:
            vals = []
            for i in range(N_random):
                eta = optimal_steps[q,eps]
                key = gen_key({
                    'opt': 'inner/outer',
                    'q': q,
                    'eta': eta,
                    'id': N_random + i,
                    'eps': eps,
                    'M': M,
                    'K': K,
                })
                vals.append(results_round_2[key])
            aggregate_results[q,eps] = np.mean(vals)

    finish = time.time()
    print(len(results_round_1))
    for k, v in results_round_1.items():
        print(k,v)

    steps = []
    for k, v in optimal_steps.items():
        print(k, v)
        steps.append(k + (v,))

    print('---')
    counts = []
    for k, v in aggregate_results.items():
        print(k, v)
        counts.append(k + (v,))

    df1 = pd.DataFrame.from_records(steps)
    df2 = pd.DataFrame.from_records(counts)
    df1.to_csv(f'steps_{M}_{K}.csv', header=True)
    df2.to_csv(f'counts_{M}_{K}.csv', header=True)

    print(f'Time elapsed phase 1: {mid-start}')
    print(f'Time elapsed phase 2: {finish-mid}')


def gen_key(arg):
    return (arg['opt'],
            arg['q'],
            arg['eta'],
            arg['id'],
            arg['eps'],
            arg['M'],
            arg['K'])


def gen_args_from_key(key):
    return {
        'opt': key[0],
        'q': key[1],
        'eta': key[2],
        'id': key[3],
        'eps': key[4],
        'M': key[5],
        'K': key[6],
    }


def main():
    global X
    global y
    global rand_data
    global best_val

    n = 50000
    X, y = get_data()
    print(f'Data distribution: {np.sum(y)}/{n}')

    best_val = optimize_param(X, y)
    print(f'----- Best val: {best_val} -----')

    for i in range(2 * N_random):
        indices = np.random.randint(0, n, 500_000)
        randX = X[indices]
        randY = y[indices]
        rand_data[i] = randX, randY

    pipelined_dsvrg_routine()
    #inner_outer_routine()


if __name__ == '__main__':
    main()


'''
experiment_args = {
    'opt': 'inner/outer',
    'q': 1,
    'eps': 1,
    'M': 10,
    'K': 10,
    'id': 1,
}

'K': key[5]
    }


experiment_results = {
    'optimal_step',
    'num_rounds',
}

q_params = [0, .25, .5, 1]
M_params = [10, 50, 250]
K_params = [10, 50, 250]
eps_params = [e]


M = 10
K = 10

random_data = []

for r in random_data:
    for eps in suboptimalities:
        for q in q_params:
            for step_size in grids[q]:

10 * 10 * 4 * 10

'''