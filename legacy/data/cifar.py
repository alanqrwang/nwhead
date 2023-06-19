import torchvision
from torchvision import transforms
import numpy as np

from .data_util import BaseData, get_class_indices
from .dataset import QueryDataset
from .sampler import QuerySampler, RandomSupportSampler, QuerySupportSampler

root_path = '/share/sablab/nfs04/data/cifar/'

class CIFARDataset(QueryDataset):
  def __init__(self, dataset, is_train, transform):
    self.dataset = dataset
    self.transform = transform
    num_classes = 10 if self.dataset == 'cifar10' else 100
    super().__init__(is_train, num_classes, transform)

  def gather(self, is_train):
    if self.dataset == 'cifar10':
      ds = torchvision.datasets.CIFAR10(root_path, is_train, self.transform, download=True)
    else:
      ds = torchvision.datasets.CIFAR100(root_path, is_train, self.transform, download=True)

    targets = np.array(ds.targets)
    return np.array(ds.data), targets

class CIFAR(BaseData):
  def __init__(self, 
               dataset, 
               batch_size, 
               test_batch_size, 
               transform_train=None,
               transform_test=None,
               num_supp_per_batch=1, 
               include_support=True, 
               subsample_size=10):
    self.dataset = dataset
    self.num_classes = 10 if self.dataset == 'cifar10' else 100
    self.transform_train = transform_train
    self.transform_test = transform_test
    super().__init__(batch_size, test_batch_size, num_supp_per_batch, self.num_classes, include_support, subsample_size)

  def get_query_set(self, is_train):
    return CIFARDataset(self.dataset, is_train=is_train, 
                        transform=self.transform_train if is_train else self.transform_test)
  
  def get_support_set(self, is_train):
    return CIFARDataset(self.dataset, is_train=True,
                        transform=self.transform_train if is_train else self.transform_test)

  def get_sampler(self):
    train_indices = get_class_indices(self.ds_train.targets, self.num_classes)
    val_indices = get_class_indices(self.ds_val.targets, self.num_classes)
    train_sampler = QuerySupportSampler(
                                    QuerySampler(train_indices),
                                    RandomSupportSampler(train_indices, self.num_supp_per_batch))

    val_sampler = QuerySupportSampler(
                                    QuerySampler(val_indices),
                                    RandomSupportSampler(train_indices, self.num_supp_per_batch))
    test_sampler = None
    return train_sampler, val_sampler, test_sampler