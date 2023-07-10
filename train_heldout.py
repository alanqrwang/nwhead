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

from data.fewshot_loaders import get_fewshot_loaders
from data.bird import Cub200Dataset
from data.dog import StanfordDogDataset
from util.metric import Metric, ECELoss
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
        self.add_bool_arg('freeze_featurizer', False)

        # NW head parameters
        self.add_argument('--kernel_type', type=str, default='euclidean',
                  help='Kernel type')
        self.add_argument('--proj_dim', type=int,
                  default=0)
        self.add_argument('--supp_num_per_class', type=int,
                  default=1)
        self.add_argument('--subsample_classes', type=int,
                  default=None, help='size of subsample sampler')
        self.add_bool_arg('use_nis_embedding', False)
        self.add_bool_arg('random_dropout', False)
        self.add_bool_arg('do_held_out_training', False)

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
                      'method{method}_dataset{dataset}_arch{arch}_lr{lr}_bs{batch_size}_projdim{proj_dim}_numsupp{numsupp}_subsample{subsample}_wd{wd}_seed{seed}'.format(
                        method=args.train_method,
                        dataset=args.dataset,
                        arch=args.arch,
                        lr=args.lr,
                        batch_size=args.batch_size,
                        proj_dim=args.proj_dim,
                        numsupp=args.supp_num_per_class,
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
        
        if args.debug_mode:
            args.num_steps_per_epoch = 5
            args.num_val_steps_per_epoch = 5
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
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(args.data_dir, True, transform_train, download=True)
        val_dataset = datasets.CIFAR10(args.data_dir, False, transform_test, download=True)
        train_dataset.num_classes = 10
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(args.data_dir, True, transform_train, download=True)
        val_dataset = datasets.CIFAR100(args.data_dir, False, transform_test, download=True)
        train_dataset.num_classes = 100
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
        val_dataset.targets = val_dataset._labels
    elif args.dataset == 'aircraft':
        train_dataset = datasets.FGVCAircraft(args.data_dir, 'trainval', transform=transform_train, download=True)
        val_dataset = datasets.FGVCAircraft(args.data_dir, 'test', transform=transform_test, download=True)
        train_dataset.num_classes = 100
        train_dataset.targets = train_dataset._labels
        val_dataset.targets = val_dataset._labels
    else:
      raise NotImplementedError()

    held_out_class = train_dataset.num_classes - 1
    train_loader, val_loader, heldout_val_loader = \
        get_fewshot_loaders(train_dataset, val_dataset, 
                            args.do_held_out_training,
                            held_out_class,
                            batch_size=args.batch_size,
                            workers=args.workers)
    num_classes = train_dataset.num_classes
    support_dataset = train_loader.dataset

    # Get network
    if args.arch == 'resnet18':
        feat_dim = 512
        if args.dataset in ['cifar10', 'cifar100']:
            featurizer = load_model('CIFAR_ResNet18')
        else:
            featurizer = load_model('resnet18')
    elif args.arch == 'resnet18imagenet':
        feat_dim = 512
        if args.dataset in ['cifar10', 'cifar100']:
            featurizer = load_model('CIFAR_ResNet18', pretrained=True)
        else:
            featurizer = load_model('resnet18', pretrained=True)
    elif args.arch == 'densenet121':
        feat_dim = 1024
        if args.dataset in ['cifar10', 'cifar100']:
            featurizer = load_model('CIFAR_DenseNet121')
        else:
            featurizer = load_model('densenet121')
    elif args.arch == 'dinov2_vits14':
        feat_dim = 384
        featurizer = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    else:
        raise NotImplementedError
    
    if args.freeze_featurizer:
        for param in featurizer.parameters():
            param.requires_grad = False
    
    if args.train_method == 'fchead':
        network = FCNet(featurizer, 
                        feat_dim, 
                        num_classes)
    elif args.train_method == 'nwhead':
        network = NWNet(featurizer, 
                        num_classes,
                        support_dataset=support_dataset,
                        feat_dim=feat_dim,
                        kernel_type=args.kernel_type,
                        num_per_class=args.supp_num_per_class,
                        subsample_classes=args.subsample_classes,
                        proj_dim=args.proj_dim,
                        debug_mode=args.debug_mode,
                        do_held_out_training=args.do_held_out_training,
                        held_out_class=held_out_class,
                        use_nis_embedding=args.use_nis_embedding,
                        random_dropout=args.random_dropout)
    else:
        raise NotImplementedError()
    summary(network)
    network.to(args.device)

    # Set loss, optimizer, and scheduler
    criterion = torch.nn.NLLLoss()
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
            'ece:val:random',
            'ece:val:full',
            'ece:val:cluster',

            'loss_heldout:val:random',
            'loss_heldout:val:full',
            'loss_heldout:val:cluster',
            'acc_heldout:val:random',
            'acc_heldout:val:full',
            'acc_heldout:val:cluster',
            'ece_heldout:val:random',
            'ece_heldout:val:full',
            'ece_heldout:val:cluster',
        ] 
    else:
        list_of_val_metrics = [
            'loss:val',
            'acc:val',
            'ece:val',
            'loss_heldout:val',
            'acc_heldout:val',
            'ece_heldout:val',
        ] 
    args.metrics = {}
    args.metrics.update({key: Metric() for key in list_of_metrics})
    args.val_metrics = {}
    args.val_metrics.update({key: Metric() for key in list_of_val_metrics})

    if args.use_wandb and not args.debug_mode:
        initialize_wandb(args)

    # Training loop
    start_epoch = 1
    best_acc1 = 0
    for epoch in range(start_epoch, args.num_epochs+1):
        print('Epoch:', epoch)
        if args.train_method == 'nwhead':
            network.eval()
            network.precompute()
            # eval_epoch(val_loader, network, criterion, optimizer, args, mode='random')
            acc1 = eval_epoch(val_loader, network, criterion, optimizer, args, mode='full')
            # eval_epoch(val_loader, network, criterion, optimizer, args, mode='cluster')
        else:
            acc1 = eval_epoch(val_loader, network, criterion, optimizer, args)

        # Heldout evaluation
        if args.train_method == 'nwhead':
            # heldout_eval_epoch(heldout_val_loader, network, criterion, optimizer, args, mode='random')
            heldout_eval_epoch(heldout_val_loader, network, criterion, optimizer, args, mode='full')
            # heldout_eval_epoch(heldout_val_loader, network, criterion, optimizer, args, mode='cluster')
        else:
            acc1 = heldout_eval_epoch(heldout_val_loader, network, criterion, optimizer, args)

        train_epoch(train_loader, network, criterion, optimizer, args)
        scheduler.step()

        # Remember best acc and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if epoch % args.log_interval == 0:
            save_checkpoint(epoch, network, optimizer,
                      args.ckpt_dir, scheduler, is_best=is_best)
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

        if args.use_wandb and not args.debug_mode:
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
            step_res = fc_step(batch, network, criterion, optimizer, args, is_train=True)
        else:
            step_res = nw_step(batch, network, criterion, optimizer, args, is_train=True)
        args.metrics['loss:train'].update_state(step_res['loss'], step_res['batch_size'])
        args.metrics['acc:train'].update_state(step_res['acc'], step_res['batch_size'])
        if i == args.num_steps_per_epoch:
            break

def eval_epoch(val_loader, network, criterion, optimizer, args, mode='random'):
    '''Eval for one epoch.'''
    network.eval()

    probs = []
    gts = []
    for i, batch in tqdm(enumerate(val_loader), 
        total=min(len(val_loader), args.num_val_steps_per_epoch)):
        if args.train_method == 'fchead':
            step_res = fc_step(batch, network, criterion, optimizer, args, is_train=False)
            args.val_metrics['loss:val'].update_state(step_res['loss'], step_res['batch_size'])
            args.val_metrics['acc:val'].update_state(step_res['acc'], step_res['batch_size'])
        else:
            step_res = nw_step(batch, network, criterion, optimizer, args, is_train=False, mode=mode)
            args.val_metrics[f'loss:val:{mode}'].update_state(step_res['loss'], step_res['batch_size'])
            args.val_metrics[f'acc:val:{mode}'].update_state(step_res['acc'], step_res['batch_size'])
        probs.append(step_res['prob'])
        gts.append(step_res['gt'])
        if i == args.num_val_steps_per_epoch:
            break
    
    ece = (ECELoss()(torch.cat(probs, dim=0), torch.cat(gts, dim=0)) * 100).item()
    if args.train_method == 'fchead':
        args.val_metrics['ece:val'].update_state(ece, 1)
        return args.val_metrics['acc:val'].result()
    else:
        args.val_metrics[f'ece:val:{mode}'].update_state(ece, 1)
        return args.val_metrics[f'acc:val:{mode}'].result()

def heldout_eval_epoch(val_loader, network, criterion, optimizer, args, mode='random'):
    '''Eval for one epoch.'''
    network.eval()

    probs = []
    gts = []
    for i, batch in tqdm(enumerate(val_loader), 
        total=min(len(val_loader), args.num_val_steps_per_epoch)):
        if args.train_method == 'fchead':
            step_res = fc_step(batch, network, criterion, optimizer, args, is_train=False)
            args.val_metrics['loss_heldout:val'].update_state(step_res['loss'], step_res['batch_size'])
            args.val_metrics['acc_heldout:val'].update_state(step_res['acc'], step_res['batch_size'])
        else:
            step_res = nw_step(batch, network, criterion, optimizer, args, is_train=False, mode=mode)
            args.val_metrics[f'loss_heldout:val:{mode}'].update_state(step_res['loss'], step_res['batch_size'])
            args.val_metrics[f'acc_heldout:val:{mode}'].update_state(step_res['acc'], step_res['batch_size'])
        probs.append(step_res['prob'])
        gts.append(step_res['gt'])
        if i == args.num_val_steps_per_epoch:
            break
    
    ece = (ECELoss()(torch.cat(probs, dim=0), torch.cat(gts, dim=0)) * 100).item()
    if args.train_method == 'fchead':
        args.val_metrics['ece_heldout:val'].update_state(ece, 1)
        return args.val_metrics['acc_heldout:val'].result()
    else:
        args.val_metrics[f'ece_heldout:val:{mode}'].update_state(ece, 1)
        return args.val_metrics[f'acc_heldout:val:{mode}'].result()

def fc_step(batch, network, criterion, optimizer, args, is_train=True):
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

    return {'loss': loss.cpu().detach().numpy(), \
            'acc': acc*100, \
            'batch_size': len(img), \
            'prob': output.exp(), \
            'gt': label}

def nw_step(batch, network, criterion, optimizer, args, is_train=True, mode='random'):
    '''Train/val for one step.'''
    img, label = batch
    img = img.float().to(args.device)
    label = label.to(args.device)
    optimizer.zero_grad()
    with torch.set_grad_enabled(is_train):
        if is_train:
            output = network(img, label)
        else:
            output = network.predict(img, mode)
        loss = criterion(output, label)
        if is_train:
            loss.backward()
            optimizer.step()
        acc = metric.acc(output.argmax(-1), label)

    return {'loss': loss.cpu().detach().numpy(), \
            'acc': acc*100, \
            'batch_size': len(img), \
            'prob': output.exp(), \
            'gt': label}

if __name__ == '__main__':
    main()