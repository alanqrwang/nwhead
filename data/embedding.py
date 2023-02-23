import os
import torch
from glob import glob
import numpy as np
from torch.utils.data import DataLoader
from .dataset import QueryDataset
from .sampler import RandomSupportSampler

class EmbeddingDataset(QueryDataset):
  def __init__(self, embeddings, label, num_classes):
    self.embeddings = embeddings
    self.labels = label
    super().__init__(False, num_classes, None)
  
  def gather(self, is_train):
    return self.embeddings, self.labels

class Embedding:
  def __init__(self, 
               embedding_path,
               batch_size, 
               num_classes, 
               num_supp_per_batch=None, 
               indices=None,
               filter_class=None):
    self.batch_size = batch_size
    self.num_supp_per_batch = num_supp_per_batch
    self.indices = indices
    assert os.path.exists(embedding_path), '{} doesn\'t exist.'.format(embedding_path)
    embed_path = os.path.join(embedding_path, 'embeddings.npy')
    label_path = os.path.join(embedding_path, 'labels.npy')
    print('Loading embeddings from:', embed_path)
    embeddings = np.load(embed_path)
    labels = np.load(label_path)
    self.ds = EmbeddingDataset(embeddings, labels, num_classes)

  def get_support_loader(self):
    assert self.num_supp_per_batch is not None
    assert self.indices is not None
    return DataLoader(self.ds, batch_size=self.batch_size, shuffle=False, 
                    sampler=RandomSupportSampler(self.indices, self.num_supp_per_batch))

  def get_query_loader(self):
    return DataLoader(self.ds, batch_size=self.batch_size, shuffle=False)