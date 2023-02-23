import os
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from scipy.io import loadmat

from .data_util import BaseData, get_class_indices
from .dataset import QueryDataset

root_path = '/share/sablab/nfs04/data/OxfordFlowers/'

class FlowerDataset(QueryDataset):
  def __init__(self, is_train, transform):
    num_classes = 102
    super().__init__(is_train, num_classes, transform)
  
  def gather(self, is_train):
    split_str = 'trnid' if is_train else 'tstid'
    set_ids = loadmat(os.path.join(root_path, 'setid.mat'), squeeze_me=True)
    image_ids = set_ids[split_str].tolist()
    self._images_folder = os.path.join(root_path, "jpg")

    labels = loadmat(os.path.join(root_path, 'imagelabels.mat'), squeeze_me=True)
    image_id_to_label = dict(enumerate((labels["labels"] - 1).tolist(), 1))

    _labels = []
    _image_files = []
    for image_id in image_ids:
        _labels.append(image_id_to_label[image_id])
        _image_files.append(os.path.join(self._images_folder, f"image_{image_id:05d}.jpg"))
    return _image_files, _labels

  def loader(self, idx):
    idx = np.array(idx)
    target = self.targets[idx]
    if idx.ndim > 0:
      img = [Image.open(self.data[i]).convert('RGB') for i in idx]
    else:
      img = Image.open(self.data[idx]).convert('RGB')
    return img, torch.tensor(target)

class Flower(BaseData):
  def __init__(self, 
               batch_size, 
               test_batch_size, 
               transform_train,
               transform_test,
               num_supp_per_batch=1, 
               include_support=True, 
               subsample_size=10):
    self.num_classes = 102
    self.transform_train = transform_train
    self.transform_test = transform_test
    super().__init__(batch_size, test_batch_size, num_supp_per_batch, self.num_classes, include_support, subsample_size)

  def get_query_set(self, is_train):
    return FlowerDataset(is_train=is_train, 
                        transform=self.transform_train if is_train else self.transform_test)
  
  def get_support_set(self, is_train):
    return FlowerDataset(is_train=True,
                        transform=self.transform_train if is_train else self.transform_test)