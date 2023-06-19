import os
import random
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import argparse
from pprint import pprint
import json

from data.cifar import CIFAR
from data.bird import Bird
from data.dog import Dog
from data.flower import Flower
from data.aircraft import Aircraft
from util.metric import Metric
from util import utils
from util import metric
from loss import loss_ops
from model import load_model
from model.net import FCNet, NWNet

class Parser(argparse.ArgumentParser):
  def __init__(self):
    super(Parser, self).__init__(description='FC and NW Head Training')
    # I/O parameters
    self.add_argument('--models_dir', default='./',
              type=str, help='directory to save models')
    self.add_argument('--log_interval', type=int,
              default=25, help='Frequency of logs')
    self.add_argument('--viz_interval', type=int,
              default=25, help='Frequency of logs')
    self.add_argument('--load', type=str, default=None,
              help='Load checkpoint at .h5 path')
    self.add_argument('--cont', type=int, default=0,
              help='Load checkpoint at .h5 path')
    self.add_argument('--gpu_id', type=int, default=0,
              help='gpu id to train on')
    self.add_bool_arg('save_preds', False)
    self.add_bool_arg('debug_mode', False)
    self.add_bool_arg('recompute_embeddings', False)

    # Machine learning parameters
    self.add_argument('--dataset', type=str, default='cifar10')
    self.add_argument('--lr', type=float, default=1e-3,
              help='Learning rate')
    self.add_argument('--batch_size', type=int,
              default=1, help='Batch size')
    self.add_argument('--num_steps_per_epoch', type=int,
              default=10000000, help='Num steps per epoch')
    self.add_argument('--num_val_steps_per_epoch', type=int,
              default=10000000, help='Num validation steps per epoch')
    self.add_argument('--num_epochs', type=int, default=200,
              help='Total training epochs')
    self.add_argument('--scheduler_milestones', nargs='+', type=int,
              default=(100, 150), help='Step size for scheduler')
    self.add_argument('--scheduler_gamma', type=float,
              default=0.1, help='Multiplicative factor for scheduler')
    self.add_argument('--seed', type=int,
              default=0, help='Seed')
    self.add_argument('--weight_decay', type=float,
              default=1e-4, help='Weight decay')
    self.add_argument('--arch', type=str, default='resnet18')
    self.add_argument(
      '--train_method', default='nwhead')
    self.add_argument(
      '--test_batch_size', type=int, default=32)

    # NW head parameters
    self.add_argument('--embed_dim', type=int,
              default=0)
    self.add_argument('--num_classes_per_batch_support', type=int,
              default=1)
    self.add_argument('--subsample_size', type=int,
              default=10, help='size of subsample sampler')

  def add_bool_arg(self, name, default=True):
    """Add boolean argument to argparse parser"""
    group = self.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no_' + name, dest=name, action='store_false')
    self.set_defaults(**{name: default})

  def parse(self):
    args = self.parse_args()
    args.run_dir = os.path.join(args.models_dir,
                  'method{method}_dataset{dataset}_arch{arch}_lr{lr}_bs{batch_size}_embeddim{embed_dim}_numsupp{numsupp}_subsample{subsample}_wd{wd}_seed{seed}'.format(
                    method=args.train_method,
                    dataset=args.dataset,
                    arch=args.arch,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    embed_dim=args.embed_dim,
                    numsupp=args.num_classes_per_batch_support,
                    subsample=args.subsample_size,
                    wd=args.weight_decay,
                    seed=args.seed
                  ))
    args.ckpt_dir = os.path.join(args.run_dir, 'checkpoints')
    if not os.path.exists(args.run_dir):
      os.makedirs(args.run_dir)
    if not os.path.exists(args.ckpt_dir):
      os.makedirs(args.ckpt_dir)

    # Print args and save to file
    print('Arguments:')
    pprint(vars(args))
    with open(args.run_dir + "/args.txt", 'w') as args_file:
      json.dump(vars(args), args_file, indent=4)
    return args


def main():
  # Parse arguments
  args = Parser().parse()

  # Set random seed
  seed = args.seed
  if seed > 0:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
  
  # Set device
  if torch.cuda.is_available():
    args.device = torch.device('cuda:'+str(args.gpu_id))
  else:
    args.device = torch.device('cpu')
    print('No GPU detected... Training will be slow!')
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

  # Get transforms
  if args.dataset in ['cifar10', 'cifar100']:
    transform_train = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              ])
    transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              ])
  else:
    transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
  print('Transforms:\n', transform_train, transform_test)

  # Get dataloaders
  include_support = True if args.train_method == 'nwhead' else False
  if args.dataset in ['cifar10', 'cifar100']:
    ds = CIFAR(args.dataset,
                    args.batch_size, 
                    args.test_batch_size,
                    transform_train=transform_train,
                    transform_test=transform_test,
                    num_supp_per_batch=args.num_classes_per_batch_support, 
                    include_support=include_support,
                    subsample_size=args.subsample_size)
  elif args.dataset == 'bird':
    ds = Bird(args.batch_size, 
                        args.test_batch_size,
                        transform_train=transform_train,
                        transform_test=transform_test,
                        num_supp_per_batch=args.num_classes_per_batch_support, 
                        include_support=include_support,
                        subsample_size=args.subsample_size)
  elif args.dataset == 'dog':
    ds = Dog(args.batch_size, 
                        args.test_batch_size,
                        transform_train=transform_train,
                        transform_test=transform_test,
                        num_supp_per_batch=args.num_classes_per_batch_support, 
                        include_support=include_support,
                        subsample_size=args.subsample_size)
  elif args.dataset == 'flower':
    ds = Flower(args.batch_size, 
                        args.test_batch_size,
                        transform_train=transform_train,
                        transform_test=transform_test,
                        num_supp_per_batch=args.num_classes_per_batch_support, 
                        include_support=include_support,
                        subsample_size=args.subsample_size)
  elif args.dataset == 'aircraft':
    ds = Aircraft(args.batch_size, 
                        args.test_batch_size,
                        transform_train=transform_train,
                        transform_test=transform_test,
                        num_supp_per_batch=args.num_classes_per_batch_support, 
                        include_support=include_support,
                        subsample_size=args.subsample_size)
  else:
    raise NotImplementedError()
  train_loader, val_loader, test_loader = ds.get_loaders()
  num_classes = ds.num_classes

  # Get network
  if args.arch == 'resnet18':
    feat_dim = 512
    if args.dataset in ['cifar10', 'cifar100']:
      feature_extractor = load_model('CIFAR_ResNet18', num_classes, args.embed_dim)
    else:
      feature_extractor = load_model('resnet18', num_classes, args.embed_dim)
  elif args.arch == 'densenet121':
    feat_dim = 1024
    if args.dataset in ['cifar10', 'cifar100']:
      feature_extractor = load_model('CIFAR_DenseNet121', num_classes, args.embed_dim)
    else:
      feature_extractor = load_model('densenet121', num_classes, args.embed_dim)
  else:
    raise NotImplementedError()
  
  if args.train_method == 'fchead':
    network = FCNet(feature_extractor, 
                    feat_dim, 
                    num_classes)
  elif args.train_method == 'nwhead':
    network = NWNet(feature_extractor, 
                    args.test_batch_size,
                    num_classes)
  else:
    raise NotImplementedError()
  utils.summary(network)
  network.to(args.device)

  # Set loss, optimizer, and scheduler
  criterion = loss_ops.NLLLoss()
  optimizer = torch.optim.SGD(network.parameters(), 
                              lr=args.lr, 
                              momentum=0.9, 
                              weight_decay=args.weight_decay, 
                              nesterov=True)
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                         milestones=args.scheduler_milestones,
                         gamma=args.scheduler_gamma)

  
  # Tracking metrics
  list_of_metrics = [
      'loss:train',
      'acc:train',
  ]
  list_of_val_metrics = [
      'loss:val',
      'acc:val',
  ] 
  args.metrics = {}
  args.metrics.update({key: Metric() for key in list_of_metrics})
  args.val_metrics = {}
  args.val_metrics.update({key: Metric() for key in list_of_val_metrics})

  # Training loop
  start_epoch = 1
  best_acc1 = 0
  for epoch in range(start_epoch, args.num_epochs+1):
    train_epoch(train_loader, network, criterion, optimizer, args)
    acc1 = eval_epoch(val_loader, network, criterion, optimizer, args)
    scheduler.step()

    # Remember best acc and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    if epoch % args.log_interval == 0:
      utils.save_checkpoint(epoch, network, optimizer,
                  args.ckpt_dir, scheduler, is_best=is_best)
    print('Epoch:', epoch)
    print("Train loss={:.6f}, train acc={:.6f}, lr={:.6f}".format(
        args.metrics['loss:train'].result(), args.metrics['acc:train'].result(), scheduler.get_last_lr()[0]))
    print("Val loss={:.6f}, val acc={:.6f}".format(
        args.val_metrics['loss:val'].result(), args.val_metrics['acc:val'].result()))
    print()

    # Reset metrics
    for _, metric in args.metrics.items():
      metric.reset_state()
    for _, metric in args.val_metrics.items():
      metric.reset_state()

def train_epoch(train_loader, network, criterion, optimizer, args):
  """Train for one epoch."""
  network.train()

  for i, batch in tqdm(enumerate(train_loader), 
    total=min(len(train_loader), args.num_steps_per_epoch)):
    if args.train_method == 'fchead':
      loss, acc, batch_size = fc_train_val_step(batch, network, criterion, optimizer, args, is_train=True)
    else:
      loss, acc, batch_size = nw_train_val_step(batch, network, criterion, optimizer, args, is_train=True)
    args.metrics['loss:train'].update_state(loss, batch_size)
    args.metrics['acc:train'].update_state(acc, batch_size)
    if i == args.num_steps_per_epoch:
      break

def eval_epoch(val_loader, network, criterion, optimizer, args):
  '''Eval for one epoch.'''
  network.eval()

  for i, batch in tqdm(enumerate(val_loader), 
    total=min(len(val_loader), args.num_val_steps_per_epoch)):
    if args.train_method == 'fchead':
      loss, acc, batch_size = fc_train_val_step(batch, network, criterion, optimizer, args, is_train=False)
    else:
      loss, acc, batch_size = nw_train_val_step(batch, network, criterion, optimizer, args, is_train=False)
    args.val_metrics['loss:val'].update_state(loss, batch_size)
    args.val_metrics['acc:val'].update_state(acc, batch_size)
    if i == args.num_val_steps_per_epoch:
      break
  return args.val_metrics['acc:val'].result()

def fc_train_val_step(batch, network, criterion, optimizer, args, is_train=True):
  '''Train/val for one step.'''
  img, label, _ = batch
  img = img.float().to(args.device)
  label = label.float().to(args.device)
  optimizer.zero_grad()
  with torch.set_grad_enabled(is_train):
    output = network(img)
    loss = criterion(output, label)
    if is_train:
      loss.backward()
      optimizer.step()
    acc = metric.acc(output, label)

  return loss.cpu().detach().numpy(), acc, len(img)

def nw_train_val_step(batch, network, criterion, optimizer, args, is_train=True):
    '''Train/val for one step.'''
    def prepare_batch(batch):
      img, label, idx = batch
      img = img.float().to(args.device)
      label = label.float().to(args.device)
      return img, label, idx
    
    qbatch, sbatch = batch
    qimg, qlabel, _ = prepare_batch(qbatch)
    simg, slabel, _ = prepare_batch(sbatch)
    optimizer.zero_grad()
    with torch.set_grad_enabled(is_train):
      output = network(qimg, simg, slabel)
      loss = criterion(output, qlabel)
      if is_train:
        loss.backward()
        optimizer.step()
      acc = metric.acc(output, qlabel)

    return loss.cpu().detach().numpy(), acc, len(qimg)

if __name__ == '__main__':
  main()