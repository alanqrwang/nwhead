import numpy as np
import random
import torch
from torch.utils.data import Dataset, ConcatDataset
from .utils import DatasetMetadata, FeatureDataset, InfiniteUniformClassLoader, FullDataset, HNSW, KNN, compute_clusters

class SupportSet:
    '''Support set base class for NW.'''
    def __init__(self, 
                 support_set, 
                 n_classes,
                 env_array=None,
                 ):
        self.y_array = np.array(support_set.targets)
        self.n_classes = n_classes

        # If env_array is provided, then support dataset should be a single
        # Pytorch Dataset. 
        if env_array is not None:
            self.env_array = env_array
            support_set = DatasetMetadata(support_set, self.env_array)
            self.combined_dataset = support_set
            self.env_datasets = self._separate_env_datasets(support_set)
        # Otherwise, it should be a list of Datasets.
        elif env_array is None and all(isinstance(d, Dataset) for d in support_set):
            assert all(isinstance(d, Dataset) for d in support_set)
            self.env_array = []
            for i, ds in enumerate(support_set):
                self.env_array += [i for _ in range(len(ds))]
            support_set = DatasetMetadata(support_set, self.env_array)
            self.env_datasets = support_set
            self.combined_dataset = self._combine_env_datasets(support_set)
        # Simplest case, no environment info and single support dataset
        else:
            self.env_array = np.zeros(len(support_set))
            support_set = DatasetMetadata(support_set, self.env_array)
            self.combined_dataset = support_set
            self.env_datasets = self._separate_env_datasets(support_set)

    def _combine_env_datasets(self, env_datasets):
        self.env_map = {i:i for i in range(len(env_datasets))}
        combined_dataset = ConcatDataset(env_datasets)
        combined_dataset.targets = np.concatenate([env.targets for env in self.env_datasets])
        assert len(combined_dataset) == len(combined_dataset.targets)
        return combined_dataset

    def _separate_env_datasets(self, combined_dataset):
        env_datasets = []
        self.env_map = {}
        for i, attr in enumerate(np.unique(self.env_array)):
            self.env_map[attr] = i
            indices = (self.env_array==attr).nonzero()[0]
            env_dataset = torch.utils.data.Subset(combined_dataset, indices)
            env_dataset.targets = self.y_array[indices]
            env_datasets.append(env_dataset)
        return env_datasets

class SupportSetTrain(SupportSet):
    '''Support set for NW training.'''
    def __init__(self, 
                 support_set, 
                 n_classes,
                 train_type, 
                 n_shot, 
                 n_way=None,
                 env_array=None,
                 ):
        super().__init__(support_set, n_classes, env_array)
        self.train_type = train_type
        self.n_shot = n_shot
        self.n_way = n_way
        self.train_iter = self._build_iter()

    def get_support(self, y):
        '''Samples a support for training.'''
        if self.train_type == 'irm':
            train_iter = np.random.choice(self.train_iter)
            sx, sy, sm = train_iter.next()
        else:
            sx, sy, sm = self.train_iter.next(y)

        return sx, sy, sm

    def _build_iter(self):
        '''Iterators for random sampling during training.
        Samples images from dataset.'''
        if self.train_type == 'random':
            train_iter = InfiniteUniformClassLoader(
                self.combined_dataset, self.n_shot, 
                self.n_way)
        else:
            train_iter = [iter(InfiniteUniformClassLoader(env, self.n_shot)) for env in self.env_datasets]
        return train_iter

class SupportSetEval(SupportSet):
    '''Support set for NW evaluation.'''
    def __init__(self, 
                 support_set, 
                 n_classes,
                 n_shot_random,
                 n_shot_full, 
                 n_shot_cluster=3,
                 n_neighbors=20,
                 env_array=None,
                 ):
        super().__init__(support_set, n_classes, env_array)
        self.n_shot_random = n_shot_random
        self.n_shot_full = n_shot_full
        self.n_shot_cluster = n_shot_cluster
        self.n_neighbors = n_neighbors
        self.support_loaders = self._build_full_loader()

    def build_infer_iters(self, sfeat, sy, smeta, sfeat_env, sy_env, smeta_env):
        # Full
        self.full_feat = sfeat
        self.full_y = sy
        self.full_meta = smeta
        self.full_feat_sep = sfeat_env
        self.full_y_sep = sy_env
        self.full_meta_sep = smeta_env

        # Cluster
        self.cluster_feat, self.cluster_y = compute_clusters(self.full_feat, self.full_y, self.n_shot_cluster)

        # Random
        feat_dataset = FeatureDataset(self.full_feat, self.full_y, self.full_meta)
        eval_loader = InfiniteUniformClassLoader(
            feat_dataset, self.n_shot_random)
        self.random_iter = iter(eval_loader)

        # KNN and HNSW
        self.knn = KNN(self.full_feat, self.full_y, n_neighbors=self.n_neighbors)
        self.hnsw = HNSW(self.full_feat, self.full_y, n_neighbors=self.n_neighbors)

    def get_support(self, mode, x=None):
        '''Samples a support for inference depending on mode.'''
        try:
            if mode == 'random':
                sfeat, sy, _ = self.random_iter.next()
            elif mode == 'full':
                sfeat, sy = self.full_feat, self.full_y
            elif mode == 'cluster':
                sfeat, sy = self.cluster_feat, self.cluster_y
            elif mode == 'ensemble':
                sfeat, sy = self.full_feat_sep, self.full_y_sep
            elif mode == 'knn':
                sfeat, sy = self.knn(x)
            elif mode == 'hnsw':
                sfeat, sy = self.hnsw(x)
            else:
                raise NotImplementedError
            return sfeat, sy
        except AttributeError:
            raise AttributeError('Did you run precompute()?')

    def _build_full_loader(self):
        '''Full loader for precomputing features during evaluation.
        Because the model assumes balanced classes during training and
        test, the support loader samples evenly across classes.
        '''
        self.full_datasets = []
        for env in self.env_datasets:
            self.full_datasets.append(FullDataset(env, self.n_shot_full))
        return [torch.utils.data.DataLoader(
                env, batch_size=128, shuffle=False, num_workers=0) for env in self.full_datasets]