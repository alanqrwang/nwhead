import torch
import torch.nn.functional as F
import numpy as np
import os
import shutil

def summary(network):
  """Print model summary."""
  print('')
  print('Model Summary')
  print('---------------------------------------------------------------')
  for name, _ in network.named_parameters():
    print(name)
  print('Total parameters:', sum(p.numel() for p in network.parameters() if p.requires_grad))
  print('---------------------------------------------------------------')
  print('')

######### Saving/Loading checkpoints ############
def load_checkpoint(network, path, optimizer=None, scheduler=None, verbose=True):
  if verbose:
    print('Loading checkpoint from', path)
    if optimizer:
      print('Loading optimizer')
    if scheduler:
      print('Loading scheduler')
  checkpoint = torch.load(path, map_location=torch.device('cpu'))
  
  # Legacy checkpoint loading, when:
  #   1. feature extractor and classifier were saved separately. 
  #   2. proj_weight was in classifier and not feature_extractor.
  if 'feature_extractor_state_dict' in checkpoint.keys():
    feat_dict = checkpoint['feature_extractor_state_dict']
    class_dict = checkpoint['classifier_state_dict']
    if 'proj_weight' in class_dict:
      proj_weight = class_dict.pop('proj_weight')
      feat_dict['proj_weight'] = proj_weight
    network.feature_extractor.load_state_dict(feat_dict)
    network.classifier.load_state_dict(class_dict)
  # Current checkpoint loading
  else:
    network.load_state_dict(checkpoint['network_state_dict'])

  if optimizer is not None and scheduler is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return network, optimizer, scheduler

  elif optimizer is not None:
    optimizer.load_state_dict(checkpoint['optimizer'])
    return network, optimizer

  else:
    return network

def save_checkpoint(epoch, network, optimizer, model_folder, scheduler=None, is_best=False):
  state = {
    'epoch': epoch,
    'network_state_dict': network.state_dict(),
    'optimizer' : optimizer.state_dict()
  }
  if scheduler is not None:
    state['scheduler'] = scheduler.state_dict()

  filename = os.path.join(model_folder, 'model.{epoch:04d}.h5')
  torch.save(state, filename.format(epoch=epoch))
  print('Saved checkpoint to', filename.format(epoch=epoch))
  if is_best:
    shutil.copyfile(filename.format(epoch=epoch), os.path.join(model_folder, 'model.best.h5'))

def save_metrics(save_dir, logger, *metrics):
  """
  metrics (list of strings): e.g. ['loss_d', 'loss_g', 'rmse_test', 'mae_train']
  """
  for metric in metrics:
    metric_arr = logger[metric]
    np.savetxt(save_dir + '/{}.txt'.format(metric), metric_arr)