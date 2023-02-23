import numpy as np
import torch
import torch.nn.functional as F
import random
from abc import ABC, abstractmethod

class QueryDataset(ABC):
  def __init__(self, 
               is_train, 
               num_classes,
               transform, 
               train_test_split=0.8):
    self.is_train = is_train
    self.num_classes = num_classes
    self.train_test_split = train_test_split
    self.transform = transform
    self.data, self.targets = self.gather(is_train)
    self.targets = np.array(self.targets)

  def __len__(self):
    return len(self.data)

  def loader(self, idx):
    return self.data[idx], torch.tensor(self.targets[idx])
  
  @abstractmethod
  def gather():
    pass

  def __getitem__(self, idx):
    img, label = self.loader(idx)
      
    # Transformations
    img = self.apply_transform(img)
    label = F.one_hot(label, self.num_classes)
    return img, label, idx
  
  def apply_transform(self, img):
    if isinstance(img, list) or (isinstance(img, (np.ndarray, torch.Tensor)) and len(img.shape) == 4):
      simgs = None
      for im in img:
        if self.transform is not None:
          im = self.transform(im)
        
        simgs = im[None] if simgs is None else torch.cat((simgs, im[None]), dim=0)
      return simgs
    else:
      if self.transform is not None:
        img = self.transform(img)
      return img

class QuerySupportDataset:
  def __init__(self, 
               qdata,
               sdata):
    self.qdata = qdata
    self.sdata = sdata
    self.data = self.qdata.data
    self.targets = self.qdata.targets

  def __getitem__(self, idx):
    assert isinstance(idx, tuple)
    qidx, sidx = idx

    qimg, qlabel, qidx = self.qdata[qidx]
    simg, slabel, sidx = self.sdata[sidx]
    return (qimg, qlabel, qidx), (simg, slabel, sidx)
  
  def __len__(self):
    return len(self.qdata)