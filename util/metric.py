import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

def check_type(x):
    return x.cpu().detach().numpy() if isinstance(x, torch.Tensor) else x

def acc(pred, targets):
    '''Returns accuracy given batch of categorical predictions and targets.'''
    pred = check_type(pred)
    targets = check_type(targets)
    return accuracy_score(targets, pred)

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
    batch_influences = []
    bs = len(softmaxes)
    for bid in range(bs):
        softmax = softmaxes[bid]
        qlabel = qlabels[bid]
        sweight = sweights[bid]

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
            val = val.item()
        self.num_samples += samples
        self.tot_val += (val * samples)
  
    def result(self):
        if self.num_samples == 0:
            return 0
        return self.tot_val / self.num_samples
    
    def reset_state(self):
        self.tot_val = 0
        self.num_samples = 0

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

class SmoothNLLLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def k_one_hot(self, targets:torch.Tensor, n_classes:int, smoothing=0.0):
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                                  .fill_(smoothing /(n_classes-1)) \
                                  .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
        if self.reduction == 'sum' else loss

    def forward(self, log_preds, targets):
        assert 0 <= self.smoothing < 1

        targets = self.k_one_hot(targets, log_preds.size(-1), self.smoothing)
        # log_preds = F.log_softmax(inputs, -1)

        if self.weight is not None:
            log_preds = log_preds * self.weight.unsqueeze(0)

        return self.reduce_loss(-(targets * log_preds).sum(dim=-1))