import os
import pandas as pd
import torch
from torchvision import transforms
import numpy as np
import csv
from PIL import Image

from .data_util import BaseData, get_class_indices
from .dataset import QueryDataset

root_path = '/share/sablab/nfs04/data/FGVCAircraft/fgvc-aircraft-2013b/'

class AircraftDataset(QueryDataset):
  def __init__(self, is_train, transform):
    num_classes = 100
    super().__init__(is_train, num_classes, transform)
  
  def gather(self, is_train, annotation_level='variant'):
    _split = 'train' if is_train else 'test'
    self._annotation_level = annotation_level 

    self._data_path = root_path

    annotation_file = os.path.join(
        self._data_path,
        "data",
        {
            "variant": "variants.txt",
            "family": "families.txt",
            "manufacturer": "manufacturers.txt",
        }[self._annotation_level],
    )
    with open(annotation_file, "r") as f:
        self.classes = [line.strip() for line in f]

    self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

    image_data_folder = os.path.join(self._data_path, "data", "images")
    labels_file = os.path.join(self._data_path, "data", f"images_{self._annotation_level}_{_split}.txt")

    self._image_files = []
    self._labels = []

    with open(labels_file, "r") as f:
      for line in f:
        image_name, label_name = line.strip().split(" ", 1)
        self._image_files.append(os.path.join(image_data_folder, f"{image_name}.jpg"))
        self._labels.append(self.class_to_idx[label_name])
    return self._image_files, self._labels

  def loader(self, idx):
    idx = np.array(idx)
    target = self.targets[idx]
    if idx.ndim > 0:
      img = [Image.open(self.data[i]).convert('RGB') for i in idx]
    else:
      img = Image.open(self.data[idx]).convert('RGB')
    return img, torch.tensor(target)

class Aircraft(BaseData):
  def __init__(self, 
               batch_size, 
               test_batch_size, 
               transform_train,
               transform_test,
               num_supp_per_batch=1, 
               include_support=True, 
               subsample_size=10):
    self.num_classes = 100
    self.transform_train = transform_train
    self.transform_test = transform_test
    super().__init__(batch_size, test_batch_size, num_supp_per_batch, self.num_classes, include_support, subsample_size)

  def get_query_set(self, is_train):
    return AircraftDataset(is_train=is_train, 
                        transform=self.transform_train if is_train else self.transform_test)
  
  def get_support_set(self, is_train):
    return AircraftDataset(is_train=True,
                        transform=self.transform_train if is_train else self.transform_test)