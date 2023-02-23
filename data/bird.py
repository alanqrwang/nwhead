import os
import pandas as pd
import torch
from torchvision import transforms
import numpy as np
import csv
from PIL import Image

from .data_util import BaseData, get_class_indices
from .dataset import QueryDataset

root_path = '/share/sablab/nfs04/data/CUB-200-2011/CUB_200_2011/'

class BirdDataset(QueryDataset):
  def __init__(self, is_train, transform):
    num_classes = 200
    super().__init__(is_train, num_classes, transform)
  
  def gather(self, is_train):
    split_path = os.path.join(root_path, 'train_test_split.txt')
    img_path = os.path.join(root_path, 'images.txt')
    label_path = os.path.join(root_path, 'image_class_labels.txt')

    df = pd.read_csv(split_path, sep=' ', header=None, names=['id', 'split'], skipinitialspace=True)
    if is_train:
      df = df[df['split'] == 0] 
    else:
      df = df[df['split'] == 1] 
    
    img_ids = df['id'] # Get ids in the split

    # Get paths and labels associated with ids
    img_df = pd.read_csv(img_path, sep=' ', header=None, names=['id', 'path'])
    img_df = img_df[img_df['id'].isin(img_ids)]
    label_df = pd.read_csv(label_path, sep=' ', header=None, names=['id', 'label'])
    label_df = label_df[label_df['id'].isin(img_ids)]
    
    paths = [os.path.join(root_path, 'images', p) for p in img_df['path'].to_numpy()]
    labels = label_df['label'].to_numpy()-1
    return paths, labels

  def loader(self, idx):
    idx = np.array(idx)
    target = self.targets[idx]
    if idx.ndim > 0:
      img = [Image.open(self.data[i]).convert('RGB') for i in idx]
    else:
      img = Image.open(self.data[idx]).convert('RGB')
    return img, torch.tensor(target)

class Bird(BaseData):
  def __init__(self, 
               batch_size, 
               test_batch_size, 
               transform_train,
               transform_test,
               num_supp_per_batch=1, 
               include_support=True, 
               subsample_size=10, 
               mislabeled_percent=0):
    self.num_classes = 200
    self.transform_train = transform_train
    self.transform_test = transform_test
    super().__init__(batch_size, test_batch_size, num_supp_per_batch, self.num_classes, include_support, subsample_size)

  def get_query_set(self, is_train):
    return BirdDataset(is_train=is_train, 
                        transform=self.transform_train if is_train else self.transform_test)
  
  def get_support_set(self, is_train):
    return BirdDataset(is_train=True,
                        transform=self.transform_train if is_train else self.transform_test)