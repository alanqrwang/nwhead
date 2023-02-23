import numpy as np
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

from .dataset import QueryDataset, QuerySupportDataset
from .sampler import QuerySampler, RandomSupportSampler, QuerySupportSampler, SubsampleQuerySupportSampler

def get_class_indices(targets, num_classes):
  assert len(np.unique(targets)) == num_classes, \
    'Not all classes in dataset, found {} out of {}'.format(len(np.unique(targets)), num_classes)
  indices = [[] for _ in range(num_classes)]
  for i, c in enumerate(targets):
    indices[c].append(i)
  return indices

class BaseData(ABC):
  def __init__(self, 
               batch_size, 
               test_batch_size, 
               num_supp_per_batch=1, 
               num_classes=2, 
               include_support=True, 
               subsample_size=10):
    self.batch_size = batch_size
    self.test_batch_size = test_batch_size
    self.num_classes = num_classes
    self.num_supp_per_batch = num_supp_per_batch
    self.include_support = include_support
    self.subsample_size = subsample_size
    self.ds_train, self.ds_val, self.ds_test = self.get_datasets()
    self.sampler_train, self.sampler_val, self.sampler_test = self.get_sampler()

  @abstractmethod
  def get_query_set(self):
    pass

  @abstractmethod
  def get_support_set(self):
    pass

  def get_support_loader(self, is_train):
    feature_ds = self.get_support_set(is_train)
    return DataLoader(feature_ds, batch_size=self.batch_size, shuffle=False)

  def get_datasets(self):
    if self.include_support:
      ds_train = QuerySupportDataset(
                                     self.get_query_set(True),
                                     self.get_support_set(True))
      ds_val = QuerySupportDataset(
                                     self.get_query_set(False),
                                     self.get_support_set(False))
    else:
      ds_train = self.get_query_set(True)
      ds_val = self.get_query_set(False)
    ds_test = self.get_query_set(False)
    return ds_train, ds_val, ds_test
    
  def get_sampler(self):
    train_indices = get_class_indices(self.ds_train.targets, self.num_classes)
    val_indices = get_class_indices(self.ds_val.targets, self.num_classes)

    train_sampler = SubsampleQuerySupportSampler(train_indices, train_indices, self.num_supp_per_batch, subsample_size=self.subsample_size)
    val_sampler = SubsampleQuerySupportSampler(val_indices, train_indices, self.num_supp_per_batch, subsample_size=10)
    test_sampler = None
    return train_sampler, val_sampler, test_sampler

  def get_loaders(self):
    if self.include_support:
      train_loader = DataLoader(self.ds_train, batch_size=self.batch_size, drop_last=True, num_workers=0,
                                  sampler=self.sampler_train)
      val_loader = DataLoader(self.ds_val, batch_size=self.batch_size, drop_last=True, num_workers=0,
                                sampler=self.sampler_val)
    else:
      train_loader = DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, drop_last=True)
      val_loader = DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=True, drop_last=True)

    test_loader = DataLoader(self.ds_test, batch_size=self.test_batch_size, drop_last=False)
    return train_loader, val_loader, test_loader
  
class NPKDBatch(ABC):
  def __init__(self, 
               batch_size, 
               test_batch_size, 
               num_supp_per_batch=1, 
               num_classes=2, 
               label_dropout=False, 
               include_nis=False, 
               include_support=True, 
               subsample_size=10):
    self.batch_size = batch_size
    self.test_batch_size = test_batch_size
    self.num_classes = num_classes
    self.num_supp_per_batch = num_supp_per_batch
    self.label_dropout = label_dropout
    self.include_nis = include_nis
    self.include_support = include_support
    self.subsample_size = subsample_size
    self.ds_train, self.ds_val, self.ds_test = self.get_datasets()
    self.sampler_train, self.sampler_val, self.sampler_test = self.get_sampler()

  @abstractmethod
  def get_query_set(self):
    pass

  @abstractmethod
  def get_support_set(self):
    pass

  def get_support_loader(self, is_train):
    feature_ds = self.get_support_set(is_train)
    return DataLoader(feature_ds, batch_size=self.batch_size, shuffle=False)

  def get_datasets(self):
    ds_train = self.get_query_set(True)
    if self.include_support:
      ds_val = QuerySupportDataset(
                                     self.get_query_set(False),
                                     self.get_support_set(False),
                                     num_classes=self.num_classes,
                                     label_dropout=self.label_dropout, 
                                     include_nis=self.include_nis)
    else:
      ds_val = self.get_query_set(False)
    ds_test = self.get_query_set(False)
    return ds_train, ds_val, ds_test

  def get_sampler(self):
    train_indices = get_class_indices(self.ds_train.targets, self.num_classes)
    val_indices = get_class_indices(self.ds_val.targets, self.num_classes)

    train_sampler = QuerySupportBatchSampler(train_indices, train_indices, self.batch_size, self.subsample_size)
    val_sampler = SubsampleQuerySupportSampler(val_indices, train_indices, self.num_supp_per_batch, subsample_size=10)
    test_sampler = None
    return train_sampler, val_sampler, test_sampler

  def get_loaders(self):
    if self.include_support:
      train_loader = DataLoader(self.ds_train, batch_sampler=self.sampler_train)
      val_loader = DataLoader(self.ds_val, batch_size=self.test_batch_size, drop_last=True, num_workers=0,
                                sampler=self.sampler_val)
    else:
      train_loader = DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True, drop_last=True)
      val_loader = DataLoader(self.ds_val, batch_size=self.test_batch_size, shuffle=True, drop_last=True)

    test_loader = DataLoader(self.ds_test, batch_size=self.test_batch_size, drop_last=False)
    return train_loader, val_loader, test_loader