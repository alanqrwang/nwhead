# Code for "A Flexible Nadaraya-Watson Head Can Offer Explainable and Calibrated Classification" (TMLR 2023)
Repository containing training and evaluation code for the NW head -- an interpretable/explainable, nonparametric classification head which can be used with any neural network.
![Architecture](figs/arch.png)
[link to paper](https://openreview.net/forum?id=iEq6lhG4O3&referrer=%5BTMLR%5D(%2Fgroup%3Fid%3DTMLR)

## NW Head
The NW head code can be found in `model/classifier.py`:
```
import torch
import torch.nn as nn
import torch.nn.functional as F

class NWHead(nn.Module):
  def forward(self,
              query_feats,
              support_feats,
              support_labels,
              scores_only=False):
    """
    Computes Nadaraya-Watson prediction.
    Returns predicted probabilities, weights, and scores.
    Args:
      query_feats: (b, embed_dim)
      support_feats: (b, num_support, embed_dim)
      support_labels: (b, num_support, num_classes)
    """
    query_feats = query_feats.unsqueeze(1)

    scores = -torch.cdist(query_feats, support_feats)
    if scores_only:
      return scores 
    probs = F.softmax(scores, dim=-1)
    output = torch.bmm(probs, support_labels).squeeze(1)
    return output
```

An example of usage in a CNN can be found in `model/net.py`.

In particular, the ranking support images by the `scores` variable enables sorting the support images by similarity, as in this figure:
![Similarities](figs/weights.png)

## Support influence
The NW head naturally lends itself to a notion of “support influence" (Section 3.4 in the paper) which finds the most helpful and most harmful examples in the support set for a given query image. The function to compute this is given in `util/metric.py`:
```
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
```

This figure shows results of ranking support images using support influence by most helpful and most harmful: 
![Influences](figs/influence.png)

## Training
Command for training NW head with paper's hyperparameters:
```
python train.py \
  --models_dir out/ \ # Directory to save model outputs
  --dataset bird  \ # Dataset to use
  --arch resnet18 \ # Feature extractor, $g_\theta$ in paper
  --train_method nwhead \ # Model to train, choose from [fchead, nwhead]
  --batch_size 4 \
  --lr 1e-3 \
  --num_epochs 40000 \
  --scheduler_milestones 20000 30000 \ # Epoch milestones to decrease lr via scheduler
  --embed_dim 128 \ # Embedding dimension, $d$ in paper
  --num_steps_per_epoch 32 \ # Number of gradient steps per training epoch
  --num_val_steps_per_epoch 32  # Number of gradient steps per validation epoch
```
These hyperparameters are probably not optimal. Notably, `num_epochs` could probably be reduced with appropriate adjustments to the learning rate scheduler. Your mileage may vary on different datasets.

Command for training (baseline) FC head with paper's hyperparameters:
```
python train.py \
  --models_dir out/ \ # Directory to save model outputs
  --dataset bird \ # Dataset to use
  --arch resnet18 \ # Feature extractor, $g_\theta$ in paper
  --train_method fchead \ # Model to train, choose from [fchead, nwhead]
  --batch_size 32 \
  --lr 1e-1 \
  --num_epochs 200 \
  --scheduler_milestones 100 150 \ # Epoch milestones to decrease lr via scheduler
  --embed_dim 0  # Embedding dimension, $d$ in paper. 0 indicates no projection.
```

## Evaluation
Evaluation pre-computes all embeddings in the dataset and saves them to the disk (in folder defined by `--models_dir`) before running forward passes. This is done for computational efficiency but can be changed depending on your needs. The script is set up to run all test modes reported in the paper (see the `args.list_of_test_modes` variable in `eval.py`).

Command for evaluating NW head with paper's hyperparameters:
```
python eval.py \
  --models_dir out/ \ # Directory to save model outputs
  --load_path weights.h5 \ # Model checkpoint to load weights from
  --dataset bird \
  --arch resnet18 \
  --test_batch_size 4 \
  --seed 1 \
  --embed_dim 128 \
```
Toggle the command `--recompute_embeddings` or `--no_recompute_embeddings` to skip recomputing embeddings each time.

## Requirements
This code was run and tested on an Nvidia A6000 GPU with the following dependencies:
+ python 3.7.11
+ torch 1.10.1
+ torchvision 0.11.2
+ numpy 1.21.5

## Citation
If you use NW head or some part of the code, please cite:
```
@article{
wang2022nwhead,
title={A Flexible Nadaraya-Watson Head Can Offer Explainable and Calibrated Classification},
author={Alan Q. Wang and Mert R. Sabuncu},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2022},
url={https://openreview.net/forum?id=iEq6lhG4O3},
note={}
}
```
