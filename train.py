import os
import random
import numpy as np
import torch
from torchvision import transforms, datasets
import torchvision
from tqdm import tqdm
import argparse
from pprint import pprint
import json
import wandb

from data.bird import Cub200Dataset
from data.dog import StanfordDogDataset
from util.metric import Metric
from util.utils import parse_bool, ParseKwargs, summary, save_checkpoint, initialize_wandb
from util import metric
from model import load_model
from nwhead.nw import NWNet
from fchead.fc import FCNet

class Parser(argparse.ArgumentParser):
  def __init__(self):
    super(Parser, self).__init__(description='NW Head Training')
    # I/O parameters
    self.add_argument('--models_dir', default='./',
              type=str, help='directory to save models')
    self.add_argument('--data_dir', default='./',
              type=str, help='directory where data lives')
    self.add_argument('--log_interval', type=int,
              default=25, help='Frequency of logs')
    self.add_argument('--workers', type=int, default=0,
              help='Num workers')
    self.add_argument('--gpu_id', type=int, default=0,
              help='gpu id to train on')
    self.add_bool_arg('debug_mode', False)

    # Machine learning parameters
    self.add_argument('--dataset', type=str, required=True)
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

    # NW head parameters
    self.add_argument('--kernel_type', type=str, default='euclidean',
              help='Kernel type')
    self.add_argument('--embed_dim', type=int,
              default=0)
    self.add_argument('--supp_num_per_class', type=int,
              default=1)
    self.add_argument('--subsample_classes', type=int,
              default=None, help='size of subsample sampler')

    # Weights & Biases
    self.add_bool_arg('use_wandb', False)
    self.add_argument('--wandb_api_key_path', type=str,
                        help="Path to Weights & Biases API Key. If use_wandb is set to True and this argument is not specified, user will be prompted to authenticate.")
    self.add_argument('--wandb_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for wandb.init() passed as key1=value1 key2=value2')

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
                    subsample=args.subsample_classes,
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
    if args.dataset in ['cifar10', 'cifar100']:
        train_dataset = datasets.CIFAR10(args.data_dir, True, transform_train, download=True)
        val_dataset = datasets.CIFAR10(args.data_dir, False, transform_test, download=True)
        train_dataset.num_classes = 10
    elif args.dataset == 'bird':
        train_dataset = Cub200Dataset(args.data_dir, True, transform_train)
        val_dataset = Cub200Dataset(args.data_dir, False, transform_test)
    elif args.dataset == 'dog':
        train_dataset = StanfordDogDataset(args.data_dir, True, transform_train)
        val_dataset = StanfordDogDataset(args.data_dir, False, transform_test)
    elif args.dataset == 'flower':
        train_dataset = datasets.Flowers102(args.data_dir, 'train', transform_train, download=True)
        val_dataset = datasets.Flowers102(args.data_dir, 'test', transform_test, download=True)
        train_dataset.num_classes = 102
        train_dataset.targets = train_dataset._labels
    elif args.dataset == 'aircraft':
        train_dataset = datasets.FGVCAircraft(args.data_dir, 'trainval', transform_train, download=True)
        val_dataset = datasets.FGVCAircraft(args.data_dir, 'test', transform_test, download=True)
        train_dataset.num_classes = 100
        train_dataset.targets = train_dataset._labels
    else:
      raise NotImplementedError()

    train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=args.batch_size, shuffle=True,
      num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
      val_dataset, batch_size=args.batch_size, shuffle=False,
      num_workers=args.workers, pin_memory=True)
    num_classes = train_dataset.num_classes

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
        raise NotImplementedError
    
    if args.train_method == 'fchead':
        network = FCNet(feature_extractor, 
                        feat_dim, 
                        num_classes,
                        use_nll_loss=args.use_nll_loss)
    elif args.train_method == 'nwhead':
        network = NWNet(feature_extractor, 
                        train_dataset,
                        num_classes,
                        feat_dim,
                        kernel_type=args.kernel_type,
                        num_per_class=args.supp_num_per_class,
                        subsample_classes=args.subsample_classes,
                        embed_dim=args.embed_dim,
                        debug_mode=args.debug_mode)
    else:
        raise NotImplementedError()
    summary(network)
    network.to(args.device)

    # Set loss, optimizer, and scheduler
    if args.use_nll_loss:
        criterion = torch.nn.NLLLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
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
    if args.train_method == 'nwhead':
        list_of_val_metrics = [
            'loss:val:random',
            'loss:val:full',
            'loss:val:cluster',
            'acc:val:random',
            'acc:val:full',
            'acc:val:cluster',
        ] 
    else:
        list_of_val_metrics = [
            'loss:val',
            'acc:val',
        ] 
    args.metrics = {}
    args.metrics.update({key: Metric() for key in list_of_metrics})
    args.val_metrics = {}
    args.val_metrics.update({key: Metric() for key in list_of_val_metrics})

    if args.use_wandb:
        initialize_wandb(args)

    # Training loop
    start_epoch = 1
    best_acc1 = 0
    for epoch in range(start_epoch, args.num_epochs+1):
        if args.train_method == 'nwhead':
            network.eval()
            network.precompute()
            eval_epoch(val_loader, network, criterion, optimizer, args, mode='random')
            acc1 = eval_epoch(val_loader, network, criterion, optimizer, args, mode='full')
            eval_epoch(val_loader, network, criterion, optimizer, args, mode='cluster')
        else:
            acc1 = eval_epoch(val_loader, network, criterion, optimizer, args)

        train_epoch(train_loader, network, criterion, optimizer, args)
        scheduler.step()

        # Remember best acc and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if epoch % args.log_interval == 0:
            save_checkpoint(epoch, network, optimizer,
                      args.ckpt_dir, scheduler, is_best=is_best)
        print('Epoch:', epoch)
        print("Train loss={:.6f}, train acc={:.6f}, lr={:.6f}".format(
            args.metrics['loss:train'].result(), args.metrics['acc:train'].result(), scheduler.get_last_lr()[0]))
        if args.train_method == 'fchead':
            print("Val loss={:.6f}, val acc={:.6f}".format(
                args.val_metrics['loss:val'].result(), args.val_metrics['acc:val'].result()))
            print()
        else:
            print("Val loss={:.6f}, val acc={:.6f}".format(
                args.val_metrics['loss:val:random'].result(), args.val_metrics['acc:val:random'].result()))
            print("Val loss={:.6f}, val acc={:.6f}".format(
                args.val_metrics['loss:val:full'].result(), args.val_metrics['acc:val:full'].result()))
            print("Val loss={:.6f}, val acc={:.6f}".format(
                args.val_metrics['loss:val:cluster'].result(), args.val_metrics['acc:val:cluster'].result()))
            print()

        if args.use_wandb:
            wandb.log({k: v.result() for k, v in args.metrics.items()})
            wandb.log({k: v.result() for k, v in args.val_metrics.items()})

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
            loss, acc, batch_size = nw_train_step(batch, network, criterion, optimizer, args)
        args.metrics['loss:train'].update_state(loss, batch_size)
        args.metrics['acc:train'].update_state(acc, batch_size)
        if i == args.num_steps_per_epoch:
            break

def eval_epoch(val_loader, network, criterion, optimizer, args, mode='random'):
    '''Eval for one epoch.'''
    network.eval()

    for i, batch in tqdm(enumerate(val_loader), 
        total=min(len(val_loader), args.num_val_steps_per_epoch)):
        if args.train_method == 'fchead':
            loss, acc, batch_size = fc_train_val_step(batch, network, criterion, optimizer, args, is_train=False)
            args.val_metrics['loss:val'].update_state(loss, batch_size)
            args.val_metrics['acc:val'].update_state(acc, batch_size)
        else:
            loss, acc, batch_size = nw_val_step(batch, network, criterion, optimizer, args, mode=mode)
            args.val_metrics[f'loss:val:{mode}'].update_state(loss, batch_size)
            args.val_metrics[f'acc:val:{mode}'].update_state(acc, batch_size)
        if i == args.num_val_steps_per_epoch:
            break
    if args.train_method == 'fchead':
        return args.val_metrics['acc:val'].result()
    else:
        return args.val_metrics[f'acc:val:{mode}'].result()

def fc_train_val_step(batch, network, criterion, optimizer, args, is_train=True):
    '''Train/val for one step.'''
    img, label = batch
    img = img.float().to(args.device)
    label = label.to(args.device)
    optimizer.zero_grad()
    with torch.set_grad_enabled(is_train):
        output = network(img)
        loss = criterion(output, label)
        if is_train:
            loss.backward()
            optimizer.step()
        acc = metric.acc(output.argmax(-1), label)

    return loss.cpu().detach().numpy(), acc, len(img)

def nw_train_step(batch, network, criterion, optimizer, args):
    '''Train for one step.'''
    img, label = batch
    img = img.float().to(args.device)
    label = label.to(args.device)
    optimizer.zero_grad()
    with torch.set_grad_enabled(True):
        output = network(img, label)[0]
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        acc = metric.acc(output.argmax(-1), label)

    return loss.cpu().detach().numpy(), acc, len(img)

def nw_val_step(batch, network, criterion, optimizer, args, mode='random'):
    '''Val for one step.'''
    img, label = batch
    img = img.float().to(args.device)
    label = label.to(args.device)
    optimizer.zero_grad()
    with torch.set_grad_enabled(False):
        output = network.predict(img, mode)[0]
        loss = criterion(output, label)
        acc = metric.acc(output.argmax(-1), label)

    return loss.cpu().detach().numpy(), acc, len(img)

if __name__ == '__main__':
    main()