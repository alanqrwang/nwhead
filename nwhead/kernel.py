import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

'''Kernels.

Forward Args:
    x: (bs, num_x, embed_dim)
    y: (bs, num_y, embed_dim)
'''

class EuclideanDistance(nn.Module):
    def forward(self, x, y):
        return -torch.cdist(x, y)

class HypersphereEuclideanDistance(nn.Module):
    def forward(self, x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return -torch.cdist(x, y)

class CosineDistance(nn.Module):
    def forward(self, x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        # (B, num_queries, embed_dim) x (B, embed_dim, num_keys) -> (B, num_queries, num_keys)
        return torch.bmm(x, y.transpose(-2, -1))

class DotProduct(nn.Module):
    def forward(self, x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        # (B, num_queries, embed_dim) x (B, embed_dim, num_keys) -> (B, num_queries, num_keys)
        return torch.bmm(x, y.transpose(-2, -1))

class Clip(nn.Module):
    def __init__(self):
        super(Clip, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        logit_scale = self.logit_scale.exp()
        return logit_scale * torch.bmm(x, y.transpose(-2, -1))

class RelationNetwork(nn.Module):
    def __init__(self, in_ch_size, input_size, hidden_size, symmetric=False):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(in_ch_size,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)
        self.symmetric = symmetric

    def _forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = F.sigmoid(self.fc2(out))
        return out

    def forward(self, x, y):
        bs = len(x)
        num_q, num_s = x.shape[1], y.shape[1]
        x = x.expand(-1, y.shape[1], -1, -1, -1) # Expand q to num_supp size
        if self.symmetric:
            diff = (x - y).abs() # Diff and absval
        scores = -self._forward(diff.view(-1, *diff.shape[2:]))
        return scores.view(bs, num_q, num_s)

def get_kernel(kernel_type):
    if kernel_type == 'euclidean':
        kernel = EuclideanDistance()
    elif kernel_type == 'hypersphere_euclidean':
        kernel = HypersphereEuclideanDistance()
    elif kernel_type == 'cosine':
        kernel = CosineDistance()
    elif kernel_type == 'dotproduct':
        kernel = DotProduct()
    elif kernel_type == 'clip':
        kernel = Clip()
    # elif kernel_type == 'relationnet':
    #     kernel = RelationNetwork(d_out, 64, 8)
    # elif kernel_type == 'relationnetsym':
    #     kernel = RelationNetwork(d_out, 64, 8, symmetric=True)
    else:
        raise NotImplementedError
    return kernel