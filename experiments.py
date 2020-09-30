import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from scipy.special import expit
import pandas as pd
import multiprocessing


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

            # Execute the variace reduced gradient step:
            w = w - eta * (gradient(self._X[[i]], w, self._Y[[i]])[0] + grad_adjustment)

            # TODO: possibly add gamma shrinkage term

        # Compute the local gradient at the iterate where we started the round
        g_local, _ = gradient(self._X, w_cur, self._Y)
        return w, g_local


#############################################################################################

## Experiments

def run_rounds_counting(arg):
    eta = arg['eta']
    q = arg['q']
    eps = arg['eps']
    id = arg['id']

    randX, randY = rand_data[id]

    best_val = .2
    # Create a DataSource class to avoid repeated use of data
    source = DataSource(randX, randY)

    eta_outer = eta
    eta_inner = eta * q

    w0 = np.asmatrix(np.zeros(X.shape[1])).T
    ws = [w0]

    log_losses = []
    l2_norms = []

    log_loss, l2_norm = objective(X, w0, y)
    log_losses.append(log_loss)
    l2_norms.append(l2_norm)

    R = 0
    while log_losses[-1] - best_val > eps:
        if R % 100 == 0:
            print(R)
        if R >= 1000:
            break
        w_cur = ws[-1]

        # Generate M random machines to perform the Local SGD steps
        inner_results = [
            Machine(source, data_count=K).execute_inner_sgd(eta_inner, w_cur)
            for _ in range(M)]

        # Perform the outer gradient step
        w_next = w_cur - eta_outer / M * np.sum(inner_results, axis=0)
        ws.append(w_next)

        # Record progress on the objective
        w_mean = np.asmatrix(np.mean(ws, axis=0))
        log_loss, l2_norm = objective(X, w_mean, y)
        log_losses.append(log_loss)
        l2_norms.append(l2_norm)

        R += 1
    print(f'-- {R}')
    key = gen_key(arg)
    results_round_1[key] = R
    return R


def alternative_tune_outer_step_size(X, y, grid, M, K, q, eps):
    num_rounds = []
    for eta in grid:
        print(f'step size: {eta}')
        num_rounds.append(run_rounds_counting(X, y, eta, M, K, q, eps))
    idx = np.argmin(num_rounds)
    return grid[idx]


def run_experiment(X, y, M, K, q, grid):
    n = 50_000
    suboptimalities = .05 / np.arange(1, 11)
    optimal_steps = []
    for eps in suboptimalities:
        cur_steps = []
        for i in range(10):
            print(f'tune {eps}, {i}')
            indices = np.random.randint(0, n, 500_000)
            randX = X[indices]
            randY = y[indices]

            cur_steps.append(
                alternative_tune_outer_step_size(randX, randY, grid, M, K, q, eps)
            )
        optimal_steps.append(np.mean(cur_steps))

    num_rounds = []
    for (eps, eta) in zip(suboptimalities, optimal_steps):
        cur_num_rounds = []
        for i in range(10):
            print(f'exp {eps}, {i}')
            indices = np.random.randint(0, n, 500_000)
            randX = X[indices]
            randY = y[indices]
            cur_num_rounds.append(run_rounds_counting(randX, randY, eta, M, K, q, eps))
        num_rounds.append(np.mean(cur_num_rounds))

    return suboptimalities, num_rounds

def run(args):
    M = args['M']
    K = args['K']
    eps = args['eps']
    q = args['q']



#############################################################################################

def callback(args):
    print('In callback...')
    run_rounds_counting(args)
    print(args)
    return

M = 10
K = 10

rand_data = {}
results_round_1 = {}
results_round_2 = {}

X = None
y = None

def main():
    global X
    global y
    global rand_data
    global results_round_1
    global results_round_2
    global M
    global K

    start = time.time()

    n = 50000
    X, y = generate_data(n)
    print(f'Data distribution: {np.sum(y)}/{n}')

    q_params = [0, .25, .5, 1]

    N_random = 2
    for i in range(2 * N_random):
        indices = np.random.randint(0, n, 500_000)
        randX = X[indices]
        randY = y[indices]
        rand_data[i] = randX, randY

    suboptimalities = .05 / np.arange(1, 11)

    grids = {
        0: [.12, .14, .16, .18, .2],
        .25: [.12, .14, .16, .18, .2],
        .5: [.12, .14, .16, .18, .2],
        1: [.12, .14, .16, .18, .2]
    }

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
                    p = multiprocessing.Process(target=callback, args=[args])
                    processes.append(p)
                    args_list.append(args)

    print(f'Processes to start: {len(processes)}')
    for p in processes[:10]:
        p.start()

    for p in processes[:10]:
        p.join()

    #for arg in args_list[:10]:
    #    callback(args)

    finish = time.time()
    print(results_round_1)
    print(f'Time elapsed: {finish-start}')


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