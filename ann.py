import numpy as np

import torch
import torch.nn as nn

from model import Model
from utils import set_seed

# Added
#import torch
from torch import func
#from torch import nn
import torch.nn.functional as F
from torch import Tensor

# Based on nn.Linear
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear

class Quadratic(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    fc_1: nn.Linear
    fc_2: nn.Linear

    def __init__(self, in_features: int, out_features: int,
                 bias_1: bool = True, bias_2: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc_1 = nn.Linear(in_features, out_features, bias_1)
        self.fc_2 = nn.Linear(in_features, out_features, bias_2)

    def reset_parameters(self) -> None:
        self.fc_1.reset_parameters()
        self.fc_2.reset_parameters()

    def forward(self, input: Tensor) -> Tensor:
        return self.fc_1(input) * (1 + self.fc_2(input))

    def weight_jacobians(self, input: Tensor) -> tuple:
        def f(x, weight1, weight2, bias1=None, bias2=None):
            return F.linear(x, weight1, bias1) * (1 + F.linear(x, weight2, bias2))

        # Current STDP methods don't include learning the bias of the network.
        # If they did, we would do argnums=(1,2,3,4) and change the code accordingly.
        return func.jacrev(f, argnums=(1,2))(input, self.fc_1.weight, self.fc_2.weight,
                                                    self.fc_1.bias,   self.fc_2.bias)

        # Sum over dimension 0 because the Hadamard formulation of the quadratic
        # causes each output entry to depend only on a specific row of weights;
        # i.e., only one row of (out_features, in_features) is ever non-zero in
        # the Jacobian for a single entry of out_features.
        # Therefore, since each entry along axis 0 is independent
        # and doesn't interfere with other entries, we can sum along axis 0
        # to get a Jacobian of (out_features, in_features).


    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias_1st={}, bias_2nd={}'.format(
            self.in_features, self.out_features,
            self.fc_1.bias is not None, self.fc_2.bias is not None
        )
        
class ANN(Model):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
    # Add 2nd param, need to change build_model call in config.py to say 'Quadratic'
    def build_model(self, type='Linear'):
        # Wrap original in if statement
        if type == 'Linear':
            self.blocks = [[nn.Linear(self.config.n_inputs, self.config.n_hidden_neurons, bias=self.config.bias),
                        nn.ReLU(),
                        nn.Dropout(self.config.dropout_p)]]
            if self.config.use_batchnorm: self.blocks[0].insert(1, nn.BatchNorm1d(self.config.n_hidden_neurons))


            for i in range(self.config.n_hidden_layers-1):
                self.block = [  nn.Linear(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=self.config.bias),
                            nn.ReLU(),
                            nn.Dropout(self.config.dropout_p)]
                if self.config.use_batchnorm: self.block.insert(1, nn.BatchNorm1d(self.config.n_hidden_neurons))
                self.blocks.append(self.block)


            self.blocks.append([nn.Linear(self.config.n_hidden_neurons, self.config.n_outputs, bias=self.config.bias)])

            self.model = [l for block in self.blocks for l in block]
            self.model = nn.Sequential(*self.model)
        elif type == 'Quadratic':
            # Implement Quadratic: wherever nn.Linear is used, use nn.Quadratic
            self.blocks = [[nn.Quadratic(self.config.n_inputs, self.config.n_hidden_neurons, bias=self.config.bias),
                        nn.ReLU(),
                        nn.Dropout(self.config.dropout_p)]]
            if self.config.use_batchnorm: self.blocks[0].insert(1, nn.BatchNorm1d(self.config.n_hidden_neurons))


            for i in range(self.config.n_hidden_layers-1):
                self.block = [  nn.Quadratic(self.config.n_hidden_neurons, self.config.n_hidden_neurons, bias=self.config.bias),
                            nn.ReLU(),
                            nn.Dropout(self.config.dropout_p)]
                if self.config.use_batchnorm: self.block.insert(1, nn.BatchNorm1d(self.config.n_hidden_neurons))
                self.blocks.append(self.block)


            self.blocks.append([nn.Quadratic(self.config.n_hidden_neurons, self.config.n_outputs, bias=self.config.bias)])

            self.model = [l for block in self.blocks for l in block]
            self.model = nn.Sequential(*self.model)
            

        print(self.model)


    def init_model(self):
        set_seed(self.config.seed)

        if self.config.init_w_method == 'kaiming_uniform':
            for i in range(self.config.n_hidden_layers+1):
                torch.nn.init.kaiming_uniform_(self.blocks[i][0].weight, nonlinearity='relu')

    def reset_model(self):
        pass

    def decrease_sig(self, epoch):
        pass

    def forward(self, x):
        out = []
        for t in range(x.size()[0]):
            out.append(self.model(x[t]).unsqueeze(0))

        return torch.concat(out)
