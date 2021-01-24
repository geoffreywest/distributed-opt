import numpy as np
from tqdm import trange, tqdm
import torch
import importlib
import sys

from .fedbase import BaseFederated
sys.path.append('/Users/geoffreywest/Desktop/Research/Srebro/Code/distributed-opt/')
from utils.model_utils import batch_data

class PDSVRGOptimizer:
    '''
    Performs the inner optimization for one client using step size =lr,
    making a gradient adjustment in each step
    TODO comment
    '''
    def __init__(self, client_model, lr):
        self.client_model = client_model    # the model to optimize
        self.lr = lr                        # the learning rate
        self.grad_sum_at_ref = None         # sum the observed gradients at the reference location
        self.ref_params = None              # the reference location to track gradient
        self.grad_count = 0                 # number of gradients summed
        self.flops = 0                      # TODO

    def reset_meta(self):
        '''
        Reset the origin and set the gradient sum to zero
        '''
        self.orig = [p.clone() for p in self.client_model.parameters()]
        self.grad_sum_at_ref = [torch.zeros_like(p) for p in self.client_model.parameters()]
        self.grad_count = 0

    def add_grad_at_ref(self, grad):
        '''
        Add an observed gradient to the running sum
        '''
        self.grad_count += 1
        for cur, new in zip(self.grad_sum_at_ref, grad):
            cur += new

    def step(self, cur_param, cur_grad, local_ref_grad, ref_grad):
        '''
        Perform the PDSVRG step and return new iterate as a list of tensors
        according to the formula:
            w^+ <- w_cur - lr * (cur_grad - local_ref_grad + ref_grad)
        '''
        return [
            cp - self.lr * (cg - lrg + rg)
        for cp, cg, lrg, rg in zip(cur_param, cur_grad, local_ref_grad, ref_grad)]

    def grad_mean(self):
        '''
        Return the mean gradient from the running sum
        '''
        return [g / self.grad_count for g in self.grad_sum_at_ref]

    def solve_inner(self, data, ref_params, ref_grad, num_epochs, batch_size):
        '''
        Perform the inner optimization routine
        '''
        self.reset_meta()

        cur_param = list(self.client_model.parameters())
        for _ in range(num_epochs):
            for X, y in batch_data(data, batch_size):
                # Update the parameter iterate
                cur_param = self.step(
                    cur_param,
                    self.client_model.gradient(cur_param, X, y),
                    self.client_model.gradient(ref_params, X, y),
                    ref_grad
                )
                # Update the gradient sum
                self.add_grad_at_ref(
                    self.client_model.gradient(self.orig, X, y)
                )

        comp = 0 # TODO
        return (cur_param, self.grad_mean()), comp



class Server(BaseFederated):
    '''
    Server for distributed optimization with pipelined DSVRG method
    '''
    def __init__(self, options, model, dataset):
        print('Using PDSVRG method to optimize.')
        self.client_model = model(*options['model_params'])
        self.inner_opt = PDSVRGOptimizer(self.client_model, options['learning_rate']) # TODO
        super(Server, self).__init__(options, model, dataset)

    def run(self):
        '''
        Train using PDSVRG method
        '''
        print('Training with {} workers ---'.format(self.clients_per_round))

        # Current parameter iterate
        latest_params = [p.clone() for p in self.client_model.get_params()]
        # Previous round parameters
        prev_params = [p.clone() for p in self.client_model.get_params()]
        # Mean gradient at prev_params during the previous round
        grad_mean = [torch.zeros_like(p) for p in self.client_model.get_params()]

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

            csolns = [] # Solutions received from clients
            cgrads = [] # Mean gradients received from clients

            for c in active_clients:
                # Execute inner optimization
                (soln, grad_mean), stats = c.solve_inner(
                    init_params=latest_params,
                    ref_params=prev_params,
                    ref_grad=grad_mean,
                    num_epochs=self.num_epochs,
                    batch_size=self.batch_size
                )

                csolns.append(soln)
                cgrads.append(grad_mean)

            # Update models
            prev_params = self.latest_params
            latest_params = self.aggregate(csolns)
            grad_mean = self.aggregate(cgrads)

        self.client_model.set_params(latest_params, clone=False)
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
        self.metrics.write()
