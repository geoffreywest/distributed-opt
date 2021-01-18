import numpy as np
from tqdm import trange, tqdm
import torch
import importlib
import sys

from .fedbase import BaseFederated
sys.path.append('/Users/geoffreywest/Desktop/Research/Srebro/Code/distributed-opt/research')
from utils.model_utils import batch_data

class ProxOptimizer(torch.optim.SGD):
    '''
    '''
    def __init__(self, client_model, lr, mu):
        super(ProxOptimizer, self).__init__(client_model.parameters(), lr)
        self.client_model = client_model    # the model to optimize
        self.lr = lr                        # the outer learning rate
        self.mu = mu                        # the prox term coefficient
        self.grad_sum = None                # sum of observed gradients
        self.orig = None                    # starting parameter of an inner optimization
        self.flops = 0 # TODO               # amount of computation in the inner opt

    def reset_meta(self):
        '''
        Reset the origin and set the gradient sum to zero
        '''
        self.orig = [p.clone() for p in self.client_model.parameters()]

    def global_step(self):
        '''
        Return the global parameter update based on the running sum of gradients:
            = orig - lr * sum(grads)
        '''
        with torch.no_grad():
            soln = [op - gp * self.lr for op, gp in zip(self.orig, self.grad_sum)]
        return soln

    def prox_term(self):
        '''
        Calculate the prox term using the current and original parameters
        '''
        cur_params = list(self.client_model.parameters())
        res = torch.tensor(.0)
        for param, orig in zip(cur_params, self.orig):
            res += torch.norm(param - orig)
        return self.mu / 2 * res


    def solve_inner(self, data, num_epochs=1, batch_size=32):
        '''
        Perform the inner optimization routine
        '''
        self.reset_meta()
        for _ in range(num_epochs):
            for X, y in batch_data(data, batch_size):
                # Perform the gradient step
                self.zero_grad()
                loss = self.client_model.loss_fn(self.client_model(X), y) + self.prox_term()
                loss.backward()
                self.step()
        soln = [p for p in self.client_model.parameters()]
        comp = num_epochs * (len(data['y'])//batch_size) * batch_size * self.flops
        return soln, comp


class Server(BaseFederated):
    '''
    Server for distributed optimization with proximal SGD
    '''
    def __init__(self, options, model, dataset):
        print('Using proximal method to optimize.')
        self.client_model = model(*options['model_params'])
        self.inner_opt = ProxOptimizer(self.client_model, options['learning_rate'], options['mu'])
        super(Server, self).__init__(options, model, dataset)

    def run(self):
        '''
        Train using inner/outer method.
        '''
        print('Training with {} workers ---'.format(self.clients_per_round))

        latest_params = [p.clone() for p in self.client_model.get_params()]
        for i in trange(self.num_rounds):
            # test model
            if i % self.eval_every == 0:
                stats = self.test()
                stats_train = self.train_error_and_loss()
                self.metrics.accuracies.append(stats)
                self.metrics.train_accuracies.append(stats_train)
                tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))

            # Select clients for the round
            selected_clients = self.select_clients(i, num_clients=self.clients_per_round)
            # Optionally drop some clients at random
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1-self.drop_percent)), replace=False)

            csolns = [] # buffer for receiving client solutions

            for c in active_clients:
                #c.set_params(latest_params, clone=True)

                # Execute inner gradient descent
                soln, stats = c.solve_inner(
                    init_params=latest_params,
                    num_epochs=self.num_epochs,
                    batch_size=self.batch_size
                )

                # Gather solutions from clients
                csolns.append(soln)

                # Track stats
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            # Update models
            latest_params = self.aggregate(csolns)

        # Final test model
        self.client_model.set_params(latest_params, clone=False)
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
        self.metrics.write()
