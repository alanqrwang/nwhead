import torch
import torch.nn as nn
import torch.nn.functional as F

class FCHead(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(FCHead, self).__init__()
    self.fc = nn.Linear(in_dim, out_dim)

  def forward(self, x):
    x = self.fc(x)
    return x

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