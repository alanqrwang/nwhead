import os
import pandas as pd
import torch
from torchvision import transforms
import numpy as np
import csv
from PIL import Image

from .data_util import BaseData, NPKDBatch, get_class_indices
from .dataset import QueryDataset

root_path = '/share/sablab/nfs04/data/StanfordDogs/'

class DogDataset(QueryDataset):
  def __init__(self, is_train, transform):
    num_classes = 120
    super().__init__(is_train, num_classes, transform)
  
  def gather(self, is_train):
    if is_train:
      file_names = os.path.join(root_path, 'train_list.csv')
    else:
      file_names = os.path.join(root_path, 'test_list.csv')
    df = pd.read_csv(file_names, sep=',', header=None, names=['path', 'label'])

    return [os.path.join(root_path, 'Images', path) for path in df.path.to_numpy()], df.label.to_numpy()-1

  def loader(self, idx):
    idx = np.array(idx)
    target = self.targets[idx]
    if idx.ndim > 0:
      img = [Image.open(self.data[i]).convert('RGB') for i in idx]
    else:
      img = Image.open(self.data[idx]).convert('RGB')
    return img, torch.tensor(target)

class Dog(BaseData):
  def __init__(self, 
               batch_size, 
               test_batch_size, 
               transform_train,
               transform_test,
               num_supp_per_batch=1, 
               include_support=True, 
               subsample_size=10):
    self.num_classes = 120
    self.transform_train = transform_train
    self.transform_test = transform_test
    super().__init__(batch_size, test_batch_size, num_supp_per_batch, self.num_classes, include_support, subsample_size)

  def get_query_set(self, is_train):
    return DogDataset(is_train=is_train, 
                        transform=self.transform_train if is_train else self.transform_test)
  
  def get_support_set(self, is_train):
    return DogDataset(is_train=True,
                        transform=self.transform_train if is_train else self.transform_test)

class StanfordDogDataset(torch.utils.data.Dataset):
  def __init__(self, is_train, transform):
    super().__init__()
    self.num_classes = 120
    self.is_train = is_train
    self.transform = transform
    self.gather()
  
  def gather(self):
      if self.is_train:
          file_names = os.path.join(root_path, 'train_list.csv')
      else:
          file_names = os.path.join(root_path, 'test_list.csv')
      df = pd.read_csv(file_names, sep=',', header=None, names=['path', 'label'])

      return [os.path.join(root_path, 'Images', path) for path in df.path.to_numpy()], df.label.to_numpy()-1

  def __len__(self):
      return len(self.paths)

  def __getitem__(self, idx):
      idx = np.array(idx)
      target = self.targets[idx]
      if idx.ndim > 0:
          img = [Image.open(self.paths[i]).convert('RGB') for i in idx]
      else:
          img = Image.open(self.paths[idx]).convert('RGB')
        
      img = self.transform(img)
      return img, torch.tensor(target)