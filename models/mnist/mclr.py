import torch
import torch.nn as nn
import numpy as np
import sys

sys.path.append('/Users/geoffreywest/Desktop/Research/Srebro/Code/distributed-opt/')
from utils.model_utils import batch_data

class Model(nn.Module):
    def __init__(self, n_classes, input_dim=28*28):
        super(Model, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.size = n_classes * (input_dim + 1)
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, X):
        return self.linear(X.float()).float() # TODO

    def set_params(self, model_params, clone):
        '''
        Update the model parameters
        '''
        # Todo delete
        '''
        assert(len(model_params) == 2)
        weight, bias = model_params
        self.linear.weight.data = weight
        self.linear.bias.data = bias
        return
        '''
        with torch.no_grad():
            for cur, new in zip(self.parameters(), model_params):
                cur.data = new.clone() if clone else new

    def get_params(self):
        with torch.no_grad():
            return self.parameters()

    def gradient(self, params, X_batch, y_batch):
        '''
        Return the gradient of the model as a list of tensors.
        '''
        if params is not None:
            self.set_params(params, clone=False)
        # Zero the gradient
        self.zero_grad()
        # Autograd the loss
        loss = self.loss_fn(self.forward(X_batch), y_batch)
        loss.backward()
        # Return the gradient as list of tensors
        return [p.grad for p in self.parameters()]

    def test(self, data):
        '''
        Test the model on given data
        '''
        num_correct = 0
        num_samples = 0
        losses = []
        with torch.no_grad():
            for X, y in batch_data(data, batch_size=32):
                out = self.forward(X)
                preds = torch.argmax(out, dim=1)
                num_correct += (preds == y).int().sum().item()
                num_samples += X.shape[0]
                losses.append(self.loss_fn(out, y).item())
        loss = np.mean(losses)
        return loss, num_correct, num_samples

    def save(self):
        print('Saving model...')
        path = '/Users/geoffreywest/Desktop/Research/Srebro/Code/distributed-opt/models/mnist/mod.pt'
        torch.save(self.linear, path)


def _test():
    '''
    Test the model implementation
    '''
    mod = Model(20,5)
    x = torch.randn(size=(40, 20))
    y = torch.randint(low=0, high=5, size=(40,))
    out = mod(x)
    print(x.shape, y.shape, out.shape)
    loss = mod.loss_fn(out, y)
    loss.backward()
    print(loss)


if __name__ == '__main__':
    _test()
