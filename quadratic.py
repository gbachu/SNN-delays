import torch
from torch import func
from torch import nn
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

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias_1st={}, bias_2nd={}'.format(
            self.in_features, self.out_features,
            self.fc_1.bias is not None, self.fc_2.bias is not None
        )
