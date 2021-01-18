import torch
import torch.nn as nn
import torch.optim as optim

class LocalOptimizer(optim.Optimizer):

    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        for group in self.params:
            for p in group['params']:
                p.add_(p.grad, alpha=-self.lr)
