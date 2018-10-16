import torch
import torch.nn as nn

from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn  import functional as F


class _CBBNorm(_BatchNorm):
    def __init__(self, num_features, num_con, eps=1e-5, momentum=0.1, affine=False):
        super(_CBBNorm, self).__init__(
            num_features, eps, momentum, affine)
        self.ConBias = nn.Sequential(
            nn.Linear(num_con, num_features),
            nn.Tanh()
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input, ConInfor):
        b, c = input.size(0), input.size(1)
        biasTar = self.ConBias(ConInfor).view(b,c,1,1)
        weight = self.weight.repeat(b).view(b,c,1,1)
        bias = self.bias.repeat(b).view(b,c,1,1)

        out = F.batch_norm(
            input, self.running_mean, self.running_var, None, None,
            self.training, self.momentum, self.eps)
        biasSor = self.avgpool(out)
        if self.affine:
            return (out - biasSor + biasTar)*weight + bias
        else:
            return out - biasSor + biasTar
    def eval(self):
        return self

class CBBNorm2d(_CBBNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        super(CBBNorm2d, self)._check_input_dim(input)