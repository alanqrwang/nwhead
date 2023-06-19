import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import FCHead, NWHead

class FCNet(nn.Module):
  def __init__(self, feature_extractor, in_dim, num_classes):
    super(FCNet, self).__init__()
    self.feature_extractor = feature_extractor
    self.classifier = FCHead(in_dim, out_dim=num_classes)

  def extract_feat(self, qimg):
    return self.feature_extractor(qimg)

  def forward(self, inputs, return_features=False):
    features = self.feature_extractor(inputs)
    logits = self.classifier(features)
    if return_features:
      return F.log_softmax(logits, dim=-1), features
    return F.log_softmax(logits, dim=-1)

class NWNet(nn.Module):
  def __init__(self, feature_extractor, test_batch_size, num_classes):
    super(NWNet, self).__init__()
    self.test_batch_size = test_batch_size
    self.feature_extractor = feature_extractor
    self.num_classes = num_classes
    self.classifier = NWHead()
  
  def extract_feat(self, qimg):
    return self.feature_extractor(qimg)

  def forward(self, qimg, simg, slabel):
    bs, n_supp, n_ch, l, w = simg.shape
    inputs = torch.cat((qimg, simg.view(-1, n_ch, l, w)), dim=0)
    feats = self.feature_extractor(inputs)
    qfeat, sfeat = feats[:len(qimg)], feats[len(qimg):]
    output = self.classifier(qfeat,
                             sfeat.view(bs, n_supp, -1), 
                             slabel)
    return torch.log(output + 1e-12)

  def forward_test(self, qimg, sfeat, slabel):
    sfeat, slabel = sfeat[:len(qimg)], slabel[:len(qimg)]
    qfeat = self.feature_extractor(qimg)
    if len(sfeat.shape) == 5: # sfeat is an image with (bs, ch, l, w)
      bs, ns, nch, l, w = sfeat.shape
      sfeat = self.feature_extractor(sfeat.view(-1, nch, l, w)).view(bs, ns, -1)
    output = self.classifier(qfeat, sfeat, slabel)
    return torch.log(output)
  
  def forward_test_full(self, qimg, embedding_loader, use_embeddings=True):
    device = qimg.device
    qfeat = self.feature_extractor(qimg)
    batch_size = len(qimg)
    attn_scores = torch.zeros(batch_size, len(embedding_loader.dataset))
    slabels = []
    for j, (sfeat, slabel, sid) in enumerate(embedding_loader):
      sfeat = sfeat.float().to(device)
      slabel = slabel.float().to(device)
      if not use_embeddings:
        sfeat = self.feature_extractor(sfeat)
      slabels.append(slabel)
      sfeat = sfeat[None].repeat(batch_size, 1, 1)
      slabel = slabel[None].repeat(batch_size, 1, 1)
      attn_score = self.classifier(qfeat, sfeat, slabel, scores_only=True)
      attn_scores[:, j*self.test_batch_size:j*self.test_batch_size+attn_score.shape[-1]] = attn_score.squeeze(1).cpu().detach()
    attn_map = F.softmax(attn_scores, dim=-1).to(device)
    slabels = torch.cat(slabels, dim=0)
    output = torch.matmul(attn_map, slabels)
    return torch.log(output)