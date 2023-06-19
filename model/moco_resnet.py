import torch
import torch.nn as nn
import torchvision.models as models

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
#         self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
#             self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

#         for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
#             param_k.data.copy_(param_q.data)  # initialize
#             param_k.requires_grad = False  # not update by gradient

#         # create the queue
#         self.register_buffer("queue", torch.randn(dim, K))
#         self.queue = nn.functional.normalize(self.queue, dim=0)

#         self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

def moco_resnet50(pretrained=True, **kwargs):
  arch = 'resnet50'
  moco_dim=128
  moco_k=65536
  moco_m=0.999
  moco_t=0.07
  mlp=True

  model = MoCo(
        models.__dict__[arch],
        moco_dim, moco_k, moco_m, moco_t, mlp)
  model = torch.nn.DataParallel(model)
  if pretrained:
    state_dict = torch.load('/home/aw847/NPK/moco_v2_800ep_pretrain.pth.tar')
    model.load_state_dict(state_dict['state_dict'])
  return model.module.encoder_q