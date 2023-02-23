import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
import argparse
from pprint import pprint
import json
from sklearn.cluster import KMeans

from data.cifar import CIFAR
from data.bird import Bird
from data.dog import Dog
from data.flower import Flower
from data.aircraft import Aircraft
from data.embedding import Embedding
from util.metric import Metric
from util import utils
from util import metric
from loss import loss_ops
from model import load_model
from model.net import FCNet, NWNet
from data.data_util import get_class_indices

class Parser(argparse.ArgumentParser):
  def __init__(self):
    super(Parser, self).__init__(description='FC and NW Head Evaluation')
    # I/O parameters
    self.add_argument('--models_dir', default='./',
              type=str, help='directory to save models')
    self.add_argument('--load_path', type=str, default=None,
              help='Load checkpoint at .h5 path')
    self.add_argument('--gpu_id', type=int, default=0,
              help='gpu id to train on')
    self.add_bool_arg('save_preds', False)
    self.add_bool_arg('recompute_embeddings', True)

    # Machine learning parameters
    self.add_argument('--dataset', type=str, default='cifar10')
    self.add_argument('--batch_size', type=int,
              default=1, help='Batch size')
    self.add_argument('--seed', type=int,
              default=0, help='Seed')
    self.add_argument('--arch', type=str, default='resnet18')
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
    args.run_dir = os.path.join(args.models_dir)
    args.output_dir = os.path.join(args.run_dir, 'outputs')
    if not os.path.exists(args.run_dir):
      os.makedirs(args.run_dir)
    if not os.path.exists(args.output_dir):
      os.makedirs(args.output_dir)

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
    print('No GPU detected... Evaluation will be slow!')
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
  print('\nTransforms:\n', transform_train, transform_test)

  # Get dataloaders
  include_support = True
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
  _, _, test_loader = ds.get_loaders()
  support_loader = ds.get_support_loader(is_train=False)
  args.num_classes = ds.num_classes

  # Get network
  if args.arch == 'resnet18':
    feat_dim = 512
    if args.dataset in ['cifar10', 'cifar100']:
      feature_extractor = load_model('CIFAR_ResNet18', args.num_classes, args.embed_dim)
    else:
      feature_extractor = load_model('resnet18', args.num_classes, args.embed_dim)
  elif args.arch == 'densenet121':
    feat_dim = 1024
    if args.dataset in ['cifar10', 'cifar100']:
      feature_extractor = load_model('CIFAR_DenseNet121', args.num_classes, args.embed_dim)
    else:
      feature_extractor = load_model('densenet121', args.num_classes, args.embed_dim)
  else:
    raise NotImplementedError()
  
  network = NWNet(feature_extractor, 
                  args.test_batch_size,
                  args.num_classes)
  utils.summary(network)
  network.to(args.device)
  network = utils.load_checkpoint(
          network, args.load_path)
  network.eval()

  # Set loss
  criterion = loss_ops.NLLLoss()
  
  # Tracking metrics
  args.list_of_test_metrics = [
    'xe:test',
    'acc:test',
    'ece:test',
  ] 
  args.list_of_test_modes = [
    'full',
    'random_1',
    'random_2',
    'random_3',
    'random_4',
    'random_5',
    'random_6',
    'random_7',
    'random_8',
    'random_9',
    'random_10',
    'cluster_1',
    'cluster_2',
    'cluster_3',
    'cluster_4',
    'cluster_5',
    'cluster_6',
    'cluster_7',
    'cluster_8',
    'cluster_9',
    'cluster_10',
    'closestcluster_1',
    'closestcluster_2',
    'closestcluster_3',
    'closestcluster_4',
    'closestcluster_5',
    'closestcluster_6',
    'closestcluster_7',
    'closestcluster_8',
    'closestcluster_9',
    'closestcluster_10',
  ]  
  args.list_of_test_metrics = [metric+':'+mode for metric in args.list_of_test_metrics
                                    for mode in args.list_of_test_modes]
  args.test_metrics = {}
  args.test_metrics.update({key: Metric() for key in args.list_of_test_metrics})

  # Test
  summary_dict = test(test_loader, support_loader, network, criterion, args)

  print()
  print('---------------------------------------------------------------')
  print('Evaluation is done. Below are file path and metrics:\n')
  print('File path:\n')
  print(args.run_dir)
  print('Metrics:\n')
  print(json.dumps(summary_dict, sort_keys=True, indent=4))
  print('---------------------------------------------------------------')
  print()

def build_embedding_loader(test_mode, num_supp, indices, args):
  embedding_path = os.path.join(args.run_dir, 'embedding')
  if test_mode in ['full', 'cluster', 'closestcluster']:
    embedding_loader = Embedding(
                  embedding_path,
                  args.test_batch_size,
                  args.num_classes,
                  ).get_query_loader()
  elif test_mode == 'random':
    embedding_loader = iter(Embedding(
                  embedding_path,
                  args.test_batch_size, 
                  args.num_classes,
                  num_supp_per_batch=num_supp,
                  indices=indices).get_support_loader())
  else:
    raise NotImplementedError('Invalid test mode')
  
  return embedding_loader

def test(test_loader, support_loader, network, criterion, args):
  # Extract embeddings
  if args.recompute_embeddings:
    save_embeddings(support_loader, network, args)
  indices = get_class_indices(support_loader.dataset.targets, args.num_classes)
  
  for test_mode_str in args.list_of_test_modes:
    print('\nTesting {} support...'.format(test_mode_str))

    preds, gts = None, []

    test_mode, num_supp = parse_test_mode(test_mode_str)
    if test_mode == 'cluster':
      embedding_loader = build_embedding_loader(test_mode, num_supp, indices, args)
      sbatch = compute_cluster_centers(embedding_loader, num_supp, args)
    elif test_mode == 'closestcluster':
      embedding_loader = build_embedding_loader(test_mode, num_supp, indices, args)
      sbatch = compute_cluster_centers(embedding_loader, num_supp, args, closest=True)
    elif test_mode == 'random':
      embedding_loader = build_embedding_loader(test_mode, num_supp, indices, args)
      sbatch = embedding_loader.next()
    elif test_mode == 'full':
      embedding_loader = build_embedding_loader(test_mode, num_supp, indices, args)
    for qbatch in tqdm(test_loader, total=len(test_loader)):
      if test_mode in ['random', 'cluster', 'closestcluster']:
        res = test_step(qbatch, sbatch, network, criterion, args)
      else:
        res = test_step_full(qbatch, embedding_loader, network, criterion, args)
      preds = res['pred'].cpu().detach().numpy() if preds is None else np.concatenate((preds, res['pred'].cpu().detach().numpy()), axis=0)
      gts += list(torch.argmax(res['label'], dim=-1).flatten().cpu().detach().numpy())
    
      args.test_metrics['xe:test:{}'.format(test_mode_str)].update_state(res['loss'], res['batch_size'])
      args.test_metrics['acc:test:{}'.format(test_mode_str)].update_state(res['acc'], res['batch_size'])
      # print('Acc', args.test_metrics['acc:test:{}'.format(test_mode_str)].result())
      # ece_val = (loss_ops.ECELoss()(torch.tensor(preds), torch.tensor(gts)) * 100).item()
      # print('ECE', ece_val)

    ece_val = (loss_ops.ECELoss()(torch.tensor(preds), torch.tensor(gts)) * 100).item()
    args.test_metrics['ece:test:{}'.format(test_mode_str)].update_state(ece_val, 1)

    if args.save_preds:
      pred_save_path = os.path.join(args.output_dir, 'preds_{}.npy'.format(test_mode_str))
      print('Saving prediction vectors to', pred_save_path)
      np.save(pred_save_path, preds)
    print('Accuracy:', args.test_metrics['acc:test:{}'.format(test_mode_str)].result())

  return save_summary_json(args)

def compute_cluster_centers(embedding_loader, num_clusters, args, closest=False):
  '''Performs k-means clustering to find support set.
  
  Args:
    in_support: If True, uses support embeddings closest to cluster centroids. Otherwise,
                uses true cluster centroids.
  '''
  print(f'Computing clusters for k={num_clusters}...')
  labels = embedding_loader.dataset.labels
  embeddings = torch.tensor(embedding_loader.dataset.data)
  img_ids = np.arange(len(embeddings))
  sfeat = None
  slabel = []
  for c in np.unique(labels):
    embeddings_class = embeddings[labels==c]
    img_ids_class = img_ids[labels==c]
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings_class)
    centroids = torch.tensor(kmeans.cluster_centers_).float()
    slabel += [c] * num_clusters 
    if closest:
      dist_matrix = torch.cdist(centroids, embeddings_class)
      min_indices = dist_matrix.argmin(dim=-1)
      dataset_indices = img_ids_class[min_indices]
      if num_clusters == 1:
        dataset_indices = [dataset_indices]
      closest_embedding = embeddings[dataset_indices]
      sfeat = closest_embedding if sfeat is None else torch.cat((sfeat, closest_embedding), dim=0)
    else:
      sfeat = centroids if sfeat is None else torch.cat((sfeat, centroids), dim=0)

  sfeat = sfeat[None].repeat(args.test_batch_size, 1, 1)
  slabel = F.one_hot(torch.tensor(slabel))[None].repeat(args.test_batch_size, 1, 1)

  return sfeat, slabel, None

def parse_test_mode(mode):
  if mode == 'full':
    return mode, None
  mode, num_supp = mode.split('_')
  return mode, int(num_supp)

def prepare_batch(batch, args):
  img, label, idx = batch
  img = img.float().to(args.device)
  label = label.float().to(args.device)
  return img, label, idx

def test_step(qbatch, sbatch, network, criterion, args):
  qimg, qlabel, qidx = prepare_batch(qbatch, args)
  sfeat, slabel, sidx = prepare_batch(sbatch, args)
  # Make sure batch sizes are the same
  sfeat, slabel = sfeat[:len(qimg)], slabel[:len(qlabel)]
  with torch.set_grad_enabled(False):
    output = network.forward_test(qimg, sfeat, slabel)
  batch_size = len(output)
  qlabel = qlabel.float().to(args.device)
  loss = criterion(output, qlabel)
  acc = metric.acc(output, qlabel)

  return {'pred':torch.exp(output), 
          'label':qlabel,
          'loss':loss, 
          'acc':acc, 
          'batch_size':batch_size}

def test_step_full(qbatch, embedding_loader, network, criterion, args):
  qimg, qlabel, qidx = prepare_batch(qbatch, args)
  with torch.set_grad_enabled(False):
    output = network.forward_test_full(qimg, embedding_loader)
  batch_size = len(output)
  qlabel = qlabel.float().to(args.device)
  loss = criterion(output, qlabel)
  acc = metric.acc(output, qlabel)

  return {'pred':torch.exp(output), 
          'label':qlabel,
          'loss':loss, 
          'acc':acc, 
          'batch_size':batch_size}

def save_embeddings(loader, network, args):
  print('Precomputing embeddings...')
  root_dir = os.path.join(args.run_dir, 'embedding')
  if not os.path.exists(root_dir):
    os.makedirs(root_dir)
  embedding_path = os.path.join(root_dir, 'embeddings.npy')
  label_path = os.path.join(root_dir, 'labels.npy')
  id_path = os.path.join(root_dir, 'ids.npy')
  embeddings = []
  labels = []
  ids = []
  for i, (qimg, qlabel, qid) in tqdm(enumerate(loader), 
    total=len(loader)):
    qimg = qimg.float().to(args.device)
    embedding = network.extract_feat(qimg).cpu().detach().numpy()
    embeddings.append(embedding)
    labels.append(torch.argmax(qlabel, dim=-1).long().cpu().detach().numpy())
    ids.append(qid.numpy())
  embeddings = torch.tensor(np.concatenate(embeddings, axis=0))
  labels = torch.tensor(np.concatenate(labels, axis=0))
  ids = torch.tensor(np.concatenate(ids, axis=0))
    
  print('Saving embeddings...')
  np.save(embedding_path, embeddings.cpu().detach().numpy())
  np.save(label_path, labels.cpu().detach().numpy())
  np.save(id_path, ids.cpu().detach().numpy())
  
  return embeddings, labels, ids

def save_summary_json(args):
  summary_dict = {}
  summary_dict.update({key: args.test_metrics[key].result()
              for key in args.test_metrics})
  
  path_name = 'summary.json'
  summary_path = os.path.join(args.run_dir, path_name)
  with open(summary_path, 'w') as outfile:
    json.dump(summary_dict, outfile, sort_keys=True, indent=4)
  return summary_dict

if __name__ == '__main__':
  main()