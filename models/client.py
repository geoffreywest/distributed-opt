import numpy as np
import torch

class Client(object):

    def __init__(self, id, group=None, train_data={'X':[],'y':[]}, eval_data={'X':[],'y':[]}, client_model=None, inner_opt=None):
        '''
        TODO comment
        '''
        self.client_model = client_model
        self.inner_opt = inner_opt
        self.id = id
        self.group = group
        self.train_data = {k: torch.tensor(np.asarray(v, dtype=np.float32)).long() for k, v in train_data.items()}
        self.eval_data = {k: torch.tensor(np.asarray(v, dtype=np.float32)).long() for k, v in eval_data.items()}
        self.num_samples = len(self.train_data['X'])
        self.num_eval_samples = len(self.eval_data['X'])
        pass

    def set_params(self, model_params, clone):
        '''
        Sets local model params
        Clone flag (boolean) determines whether parameters are cloned
        '''
        self.client_model.set_params(model_params, clone)

    def get_params(self):
        '''
        Returns local model params
        '''
        return self.client_model.get_params()

    def get_grads(self, model_len):
        '''
        Solves local gradients
        '''
        return self.client_model.get_gradients(self.train_data, model_len)

    def solve_grad(self):
        '''
        Solves local gradients and compute/communication info

        Return:
            1. num_samples: number of samples used in training
            2. grads: local gradients # TODO what form
            3. bytes_r: number of bytes received
            4. comp: number of FLOPs executed in training
            5. bytes_w: number of bytes transmitted to server
        '''
        bytes_w = self.client_model.size
        grads = self.client_model.get_gradients(self.train_data)
        comp = self.client_model.flops * self.num_samples
        bytes_r = self.client_model.size
        return (self.num_samples, grads), (bytes_r, bytes_w)

    def solve_inner(self, init_params, **kwargs):
        '''
        Solves local optimization problem starting from init_params
        Params:
            - init_params: the params to start optimizing from
            - kwargs: other arguments for the optimizer

        Return:
            1. num_samples: number of samples used in training
            2. soln: local optimization solution
            3. bytes_r: number of bytes received
            4. comp: number of FLOPs executed in training
            5. bytes_w: number of bytes transmitted to server
        '''
        bytes_w = self.client_model.size
        self.set_params(init_params, clone=True)
        soln, comp = self.inner_opt.solve_inner(self.train_data, **kwargs)
        bytes_r = self.client_model.size
        return soln, (bytes_w, comp, bytes_r)

    def solve_iters(self, init_params, **kwargs):
        '''
        Perform the inner optimization routine for a set number of iterations
        '''
        bytes_w = self.client_model.size
        self.set_params(init_params, clone=True)
        soln, comp = self.inner_opt.solve_iters(self.train_data, **kwargs)
        bytes_r = self.client_model.size
        return soln, (bytes_w, comp, bytes_r)

    def train_error_and_loss(self):
        loss, num_correct, num_samples = self.client_model.test(self.train_data)
        return loss, num_correct, num_samples

    def test(self):
        '''
        Tests current model on local eval_data

        Return:
            tot_correct: total correct predictions
            test_samples: int
        '''
        loss, num_correct, num_samples = self.client_model.test(self.eval_data)
        return loss, num_correct, num_samples
