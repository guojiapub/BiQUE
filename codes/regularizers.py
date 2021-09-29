import torch
from torch import nn
from typing import Tuple
from abc import ABC, abstractmethod
from qutils import get_norm



class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass



class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += self.weight * torch.sum(
                    torch.abs(f) ** 3
                ) / f.shape[0]
        return norm


class wN3(Regularizer):
    def __init__(self, weight: float):
        super(wN3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            h,r,t = factor
            norm += 2.0 * torch.sum(h**3) 
            norm += 2.0 * torch.sum(t**3) 
            norm += 0.5 * torch.sum(r**3) 
        return self.weight * norm / h.shape[0]





