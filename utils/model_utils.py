import numpy as np
import json
import os
import torch

class Metrics(object):

    def __init__(self, clients, params):
        self.params = params
        num_rounds = params['num_rounds']
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}

        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}
        self.accuracies = []
        self.train_accuracies = []

    def update(self, rnd, cid, stats):
        bytes_w, comp, bytes_r = stats
        self.bytes_written[cid][rnd] += bytes_w
        self.client_computations[cid][rnd] += comp
        self.bytes_read[cid][rnd] += bytes_r

    def write(self):
        metrics = {}
        metrics['dataset'] = self.params['dataset']
        metrics['num_rounds'] = self.params['num_rounds']
        metrics['eval_every'] = self.params['eval_every']
        metrics['clients_per_round'] = self.params['clients_per_round']
        metrics['learning_rate'] = self.params['learning_rate']
        metrics['mu'] = self.params['mu']
        metrics['rho'] = self.params['rho']
        metrics['num_epochs'] = self.params['num_epochs']
        metrics['batch_size'] = self.params['batch_size']
        metrics['accuracies'] = self.accuracies
        metrics['train_accuracies'] = self.train_accuracies
        metrics['client_computations'] = self.client_computations
        metrics['bytes_written'] = self.bytes_written
        metrics['bytes_read'] = self.bytes_read

        if self.params['optimizer'] == 'fedio':
            format_keys = ['seed', 'optimizer', 'learning_rate', 'num_epochs', 'rho']
        elif self.params['optimizer'] == 'fedpdsvrg':
            format_keys = ['seed', 'optimizer', 'learning_rate', 'num_epochs']
        elif self.params['optimizer'] == 'fedprox':
            format_keys = ['seed', 'optimizer', 'learning_rate', 'num_epochs', 'mu']
        else:
            format_keys = ['seed', 'optimizer', 'learning_rate', 'num_epochs', 'mu']

        format_params = [self.params[k] for k in format_keys]

        metrics_file = os.path.join('out', self.params['dataset'],
            ('metrics'+'_{}'*len(format_params)+'.json').format(*format_params))

        if not os.path.exists(os.path.join('out', self.params['dataset'])):
            os.mkdir(os.path.join('out', self.params['dataset']))
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)

def read_data(train_data_file, test_data_file):
    '''
    Parses data in given train and test data directories

    TODO change assumptions?
    Assumes:
        - Data are in .json files with keys 'users' and 'user_data'
        - The set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    with open(train_data_file, 'r') as f:
        train_data = json.load(f)
    with open(test_data_file, 'r') as f:
        test_data = json.load(f)

    clients = list(sorted(train_data['user_data'].keys()))
    groups = [] # TODO fix this for general case
    return clients, groups, train_data['user_data'], test_data['user_data']

def batch_data(data, batch_size):
    '''
    Input data is a dict {'X': np.array, 'y': np.array} on one client
    Returns:
        X, y which are a randomly shuffled np.array of length batch_size
    '''
    X = data['X']
    y = data['y']

    # Shuffle the data
    shuffle = np.random.permutation(len(X))
    X = X[shuffle]
    y = y[shuffle]

    # Loop through mini-batches
    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        yield X_batch, y_batch


def batch_data_multiple_iters(data, batch_size, num_iters):
    '''
    Input data is a dict {'X': np.array, 'y': np.array} on one client
    Returns:
        X, y which are a randomly shuffled np.array of length batch_size
    '''
    X = data['X']
    y = data['y']

    idx = 0
    for i in range(num_iters):
        if idx + batch_size > len(X):
            # Shuffle the data
            shuffle = np.random.permutation(len(X))
            X = X[shuffle]
            y = y[shuffle]
            idx = 0

        X_batch = X[idx:idx+batch_size]
        y_batch = y[idx:idx+batch_size]
        yield X_batch, y_batch
