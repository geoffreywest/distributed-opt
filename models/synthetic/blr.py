import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, input_dim):
        super(Model, self).__init__()
        self._input_dim = input_dim
        self._linear = nn.Linear(input_dim, 1)
        self._loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self._linear(x)

    @property
    def loss_fn(self):
        return self._loss



def _test():
    mod = Model(20)
    x = torch.randn(size=(40, 20), dtype=torch.float32)
    y = torch.randint(low=0, high=2, size=(40, 1), dtype=torch.float32)
    out = mod(x)
    print(x.shape, y.shape, out.shape)
    loss = mod.loss_fn(out, y)
    loss.backward()
    print(loss)


if __name__ == '__main__':
    _test()
