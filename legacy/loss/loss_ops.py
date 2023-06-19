import torch
import torch.nn.functional as F

class NLLLoss(torch.nn.Module):
  '''
  Args:
    pred: Log-softmax of predictions (bs, num_classes)
    target_onehot: One-hot ground truth (bs, num_classes)
  '''
  def forward(self, pred, target_onehot):
    targets = torch.argmax(target_onehot, dim=-1)
    return F.nll_loss(pred, targets)

class LabelSmoothingLoss(torch.nn.Module):
  '''
  Args:
    y_pred: Log-softmax of predictions (bs, num_classes)
    y_true: One-hot ground truths (bs, num_classes)
  '''
  def __init__(self, label_smoothing=0) -> None:
    super().__init__()
    self.ls = label_smoothing

  def forward(self, y_pred, y_true):
    y_pred = torch.clamp(torch.exp(y_pred), 1e-9, 1 - 1e-9)

    y_true = (1-self.ls)*y_true + self.ls/y_true.shape[-1]
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()

# https://github.com/gpleiss/temperature_scaling
class ECELoss(torch.nn.Module):
  """
  Calculates the Expected Calibration Error of a model.
  (This isn't necessary for temperature scaling, just a cool metric).
  The input to this loss is the logits of a model, NOT the softmax scores.
  This divides the confidence outputs into equally-sized interval bins.
  In each bin, we compute the confidence gap:
  bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
  We then return a weighted average of the gaps, based on the number
  of samples in each bin
  See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
  "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
  2015.
  """
  def __init__(self, n_bins=15):
    """
    n_bins (int): number of confidence interval bins
    """
    super(ECELoss, self).__init__()
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    self.bin_lowers = bin_boundaries[:-1]
    self.bin_uppers = bin_boundaries[1:]

  def forward(self, softmaxes, labels):
    confidences, predictions = torch.max(softmaxes, dim=1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=softmaxes.device)
    for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
      # Calculated |confidence - accuracy| in each bin
      in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
      prop_in_bin = in_bin.float().mean()
      if prop_in_bin.item() > 0:
        accuracy_in_bin = accuracies[in_bin].float().mean()
        avg_confidence_in_bin = confidences[in_bin].mean()
        ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece