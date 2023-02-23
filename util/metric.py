import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def check_type(x):
  return x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x

def acc(scores, targets):
  '''Returns accuracy given batch of probabilities and one-hot targets.'''
  return accuracy_score(torch.argmax(targets, dim=-1).cpu().detach().numpy(),
                        torch.argmax(scores, dim=-1).cpu().detach().numpy())

def roc(pr, gt):
  """pr  : prediction, B 
     gt  : ground-truth binary mask, B."""
  pr = check_type(pr)
  gt = check_type(gt)
  return 100 * float(roc_auc_score(gt, pr))

def support_influence(softmaxes, qlabels, sweights, slabels):
  '''
  Influence is defined as L(rescaled_softmax, qlabel) - L(softmax, qlabel).
  Positive influence => removing support image increases loss => support image was helpful
  Negative influence => removing support image decreases loss => support image was harmful
  bs should be 1.
  
  softmaxes: (bs, num_classes)
  qlabel: One-hot encoded query label (bs, num_classes)
  sweights: Weights between query and each support (bs, num_support)
  slabels: One-hot encoded support label (bs, num_support, num_classes)
  '''
  # assert len(softmaxes) == 1
  batch_influences = []
  bs = len(softmaxes)
  for bid in range(bs):
    softmax = softmaxes[bid]
    qlabel = qlabels[bid]
    sweight = sweights[bid]
    # slabels = slabels[bid]

    qlabel_cat = qlabel.argmax(-1).item()
    slabels_cat = slabels.argmax(-1)
    
    p = softmax[qlabel_cat]
    indicator = (slabels_cat==qlabel_cat).long()
    influences = torch.log((p - p*sweight)/(p - sweight*indicator))

    batch_influences.append(influences[None])
  return torch.cat(batch_influences, dim=0)

class Metric:
  def __init__(self) -> None:
    self.tot_val = 0
    self.num_samples = 0

  def update_state(self, val, samples):
    if isinstance(val, torch.Tensor):
      val = val.cpu().detach().item()
    if isinstance(val, np.ndarray):
      val = np.asscalar(val)
    self.num_samples += samples
    self.tot_val += (val * samples)
  
  def result(self):
    if self.num_samples == 0:
      return 0
    return self.tot_val / self.num_samples
  
  def reset_state(self):
    self.tot_val = 0
    self.num_samples = 0