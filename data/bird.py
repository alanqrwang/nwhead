import os
import pandas as pd
import torch
import numpy as np
from PIL import Image

root_path = '/share/sablab/nfs04/data/CUB-200-2011/CUB_200_2011/'

class Cub200Dataset(torch.utils.data.Dataset):
  def __init__(self, is_train, transform):
    super().__init__()
    self.num_classes = 200
    self.is_train = is_train
    self.transform = transform
    self.gather()
  
  def gather(self):
      split_path = os.path.join(root_path, 'train_test_split.txt')
      img_path = os.path.join(root_path, 'images.txt')
      label_path = os.path.join(root_path, 'image_class_labels.txt')

      df = pd.read_csv(split_path, sep=' ', header=None, names=['id', 'split'], skipinitialspace=True)
      if self.is_train:
          df = df[df['split'] == 0] 
      else:
          df = df[df['split'] == 1] 
      
      img_ids = df['id'] # Get ids in the split

      # Get paths and labels associated with ids
      img_df = pd.read_csv(img_path, sep=' ', header=None, names=['id', 'path'])
      img_df = img_df[img_df['id'].isin(img_ids)]
      label_df = pd.read_csv(label_path, sep=' ', header=None, names=['id', 'label'])
      label_df = label_df[label_df['id'].isin(img_ids)]
      
      self.paths = [os.path.join(root_path, 'images', p) for p in img_df['path'].to_numpy()]
      self.targets = label_df['label'].to_numpy()-1

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