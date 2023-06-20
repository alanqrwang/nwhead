import os
import pandas as pd
import torch
import numpy as np
from PIL import Image

class StanfordDogDataset(torch.utils.data.Dataset):
  def __init__(self, root_path, is_train, transform):
    super().__init__()
    self.num_classes = 120
    self.root_path = root_path
    self.is_train = is_train
    self.transform = transform
    self.gather()
  
  def gather(self):
      if self.is_train:
          file_names = os.path.join(self.root_path, 'train_list.csv')
      else:
          file_names = os.path.join(self.root_path, 'test_list.csv')
      df = pd.read_csv(file_names, sep=',', header=None, names=['path', 'label'])

      return [os.path.join(self.root_path, 'Images', path) for path in df.path.to_numpy()], df.label.to_numpy()-1

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