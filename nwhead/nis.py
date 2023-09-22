import torch
import torch.nn as nn

class BaseNIS(nn.Module):
    def __init__(self, n_classes) -> None:
        super().__init__()
        self.nis_class = n_classes
    
    def forward(self, x):
        pass
    
    def add_nis_class_to_sy(self, sy, n_shot=1):
        '''Add NIS class to support labels'''
        nis_class = torch.full((n_shot,), self.nis_class).to(sy.device)
        sy = torch.cat((sy, nis_class))
        return sy

class ConstantNIS(BaseNIS):
    def __init__(self, n_classes, c) -> None:
        super().__init__(n_classes)
        self.c = c
    
    def forward(self, x):
        return self.c

class ScalarNIS(BaseNIS):
    def __init__(self, n_classes, init_val=-10.0) -> None:
        super().__init__(n_classes)
        self.c = nn.Parameter(torch.tensor(init_val))
    
    def forward(self, x):
        return self.c

class LinearNIS(BaseNIS):
    def __init__(self, n_classes, in_dim) -> None:
        super().__init__(n_classes)
        self.linear = nn.Linear(in_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

def get_nis_module(nis_type, n_classes, in_dim=None, c=None):
    if nis_type == 'constant':
        return ConstantNIS(n_classes, c)
    elif nis_type == 'scalar':
        return ScalarNIS(n_classes, c)
    elif nis_type == 'linear':
        return LinearNIS(n_classes, in_dim)
    else:
        raise ValueError(f'NIS type {nis_type} is not supported.')