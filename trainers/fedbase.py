import numpy as np
import torch
from tqdm import tqdm

from models.client import Client
from utils.model_utils import Metrics

class BaseFederated(object):
    def __init__(self, options, model, dataset):
        # Transfer params to self
        for key, val in options.items():
            setattr(self, key, val)

        # Create worker nodes
        self.clients = self.setup_clients(dataset)
        if self.verbosity > 0:
            print('{} Clients in Total'.format(len(self.clients)))
        self.latest_params = self.client_model.get_params()

        # Initialize metrics
        self.metrics = Metrics(self.clients, options)

    def setup_clients(self, dataset):
        '''
        Instantiate a list of clients
        '''
        users, groups, train_data, test_data = dataset

        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [Client(u, g, train_data[u], test_data[u], self.client_model, self.inner_opt) for u, g in zip(users, groups)]
        return all_clients

    def train_error_and_loss(self):
        '''
        Returns performance of all clients on local training data as a tuple of lists
        - ids: [client ids]
        - groups: [client groups]
        - num_samples: [number of local samples on each client]
        - num_correct: [number of correct predictions on the training data]
        - losses: [local training losses of each client]
        '''
        losses, tot_correct, num_samples = zip(*[c.train_error_and_loss() for c in self.clients])

        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        losses = list(map(float, losses))

        return ids, groups, num_samples, tot_correct, losses

    def show_grads(self):
        '''
        Return: gradients on all workers and the global gradient
        '''

        model_len = process_grad(self.latest_model).size
        global_grads = np.zeros(model_len)

        intermediate_grads = []
        samples = []

        self.client_model.set_params(self.latest_model)
        for c in self.clients:
            num_samples, client_grads = c.get_grads(self.latest_model)
            samples.append(num_samples)
            global_grads = np.add(global_grads, client_grads * num_samples)
            intermediate_grads.append(global_grads)

        global_grads = global_grads * 1.0 / np.sum(np.asarray(samples))
        intermediate_grads.append(global_grads)

        return  intermediate_grads

    def test(self):
        '''
        Evaluate model performance on all clients

        Return:
            1. ids: client ids tested
            2. groups: client groups
            3. num_samples: number of evaluation samples on each client
            4. tot_correct: number evaluation samples correct
        '''
        losses, num_correct, num_samples = zip(*[c.test() for c in self.clients])
        ids = [c.id for c in self.clients]
        groups = [c.group for c in self.clients]
        losses = list(map(float, losses))
        return ids, groups, num_samples, num_correct, losses

    def save(self):
        self.client_model.save()

    def select_clients(self, round, num_clients):
        '''
        Select num_clients clients and return as a list
        '''
        num_clients = min(num_clients, len(self.clients))
        return np.random.choice(self.clients, num_clients, replace=False)

    def aggregate(self, solns):
        '''
        Returns the mean of the local solutions
        '''
        #if len(wsolns) == 0: return []
        #_, solns = zip(*wsolns)
        res = [torch.zeros_like(p) for p in solns[0]]
        for pgroup in solns:
            for cur, new in zip(res, pgroup): cur += new / len(solns)
        return res
