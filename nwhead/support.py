import numpy as np
import random
import torch
from torch.utils.data import Dataset, ConcatDataset
from sklearn.cluster import KMeans
from .utils import DatasetMetadata, FeatureDataset, InfiniteUniformClassLoader, FullDataset, HNSW, KNN

class SupportSet:
    '''Support set for NW.'''
    def __init__(self, 
                 support_set, 
                 train_type, 
                 n_shot, 
                 n_shot_full, 
                 n_classes,
                 n_way=None,
                 env_array=None,
                 n_clusters=3,
                 class_dropout=0,
                 nis_scalar=None,
                 ):
        self.train_type = train_type
        self.n_shot = n_shot
        self.n_shot_full = n_shot_full
        self.y_array = np.array(support_set.targets)
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_clusters = n_clusters
        self.class_dropout = class_dropout
        self.nis_scalar = nis_scalar

        # NIS
        if self.nis_scalar:
            self.nis_class = self.n_classes

        # If env_array is provided, then support dataset should be a single
        # Pytorch Dataset. 
        if env_array is not None:
            # assert train_type in ['match', 'mixmatch']
            self.env_array = env_array
            support_set = DatasetMetadata(support_set, self.env_array)
            self.combined_dataset = support_set
            self.env_datasets = self._separate_env_datasets(support_set)
        # Otherwise, it should be a list of Datasets.
        elif env_array is None and all(isinstance(d, Dataset) for d in support_set):
            assert all(isinstance(d, Dataset) for d in support_set)
            assert train_type in ['match', 'mixmatch']
            self.env_array = []
            for i, ds in enumerate(support_set):
                self.env_array += [i for _ in range(len(ds))]
            support_set = DatasetMetadata(support_set, self.env_array)
            self.env_datasets = support_set
            self.combined_dataset = self._combine_env_datasets(support_set)
        # Simplest case, no environment info and single support dataset
        else:
            assert train_type in ['random', 'unbalanced']
            self.env_array = np.zeros(len(support_set))
            support_set = DatasetMetadata(support_set, self.env_array)
            self.combined_dataset = support_set
            self.env_datasets = self._separate_env_datasets(support_set)

        self.support_loaders = self._build_full_loader()

        self.train_iter = self._build_train_iter()

    def build_infer_iters(self, sfeat, sy, smeta, sfeat_env, sy_env, smeta_env):
        # Full
        self.full_feat = sfeat
        self.full_y = sy
        self.full_meta = smeta
        self.full_feat_sep = sfeat_env
        self.full_y_sep = sy_env
        self.full_meta_sep = smeta_env

        # Cluster
        self.cluster_feat, self.cluster_y = self._compute_clusters()

        # Random
        feat_dataset = FeatureDataset(self.full_feat, self.full_y, self.full_meta)
        eval_loader = InfiniteUniformClassLoader(
            feat_dataset, self.n_shot)
        self.random_iter = iter(eval_loader)

        # KNN and HNSW
        self.knn = KNN(self.full_feat, self.full_y, n_neighbors=20)
        self.hnsw = HNSW(self.full_feat, self.full_y, n_neighbors=20)

    def get_train_support(self, y, env_index):
        '''Samples a support for training.'''
        if self.train_type == 'mixmatch':
            train_iter = np.random.choice(self.train_iter)
            sx, sy, sm = train_iter.next()
        elif self.train_type == 'match':
            assert torch.equal(env_index, torch.ones_like(env_index)*env_index[0])
            meta_idx = self.env_map[env_index[0].item()]
            train_iter = self.train_iter[meta_idx]
            sx, sy, sm = train_iter.next()
        else:
            sx, sy, sm = self.train_iter.next(y)

        # Occasionally verify sampled classes
        if random.random() < 0.01 and self.train_type != 'unbalanced':
            self._verify_sy(sy)

        # Add NIS class if specified
        if self.nis_scalar:
            nis_class = torch.full((self.n_shot,), self.nis_class).to(sy.device)
            sy = torch.cat((sy, nis_class))

        return sx, sy, sm

    def get_infer_support(self, mode, x=None):
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

            # Add NIS class if specified
            if self.nis_scalar:
                if mode == 'random':
                    n_shot = self.n_shot
                elif mode == 'full':
                    n_shot = self.n_shot_full
                elif mode == 'cluster':
                    n_shot = self.n_clusters
                else:
                    raise NotImplementedError # TODO

                nis_class = torch.full((n_shot,), self.nis_class).to(sy.device)
                sy = torch.cat((sy, nis_class))
            return sfeat, sy
        except AttributeError:
            print('Did you run precompute()?')

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

    def _build_train_iter(self):
        '''Iterators for random sampling during training.
        Samples images from dataset.'''
        if self.train_type == 'random':
            train_iter = InfiniteUniformClassLoader(
                self.combined_dataset, self.n_shot, 
                self.n_way)
        else:
            train_iter = [iter(InfiniteUniformClassLoader(env, self.n_shot)) for env in self.env_datasets]
        return train_iter

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

    def _compute_clusters(self, closest=False):
        '''Performs k-means clustering to find support set.
        
        :param num_clusters: Number of cluster centroids per class.
        :param closest: If True, uses support features closest to cluster centroids. Otherwise,
                    uses true cluster centroids.
        '''
        embeddings = self.full_feat
        labels = self.full_y
        img_ids = np.arange(len(embeddings))
        sfeat = None
        slabel = []
        for c in np.unique(labels):
            embeddings_class = embeddings[labels==c]
            img_ids_class = img_ids[labels==c]
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init='auto').fit(embeddings_class)
            centroids = torch.tensor(kmeans.cluster_centers_).float()
            slabel += [c] * self.n_clusters 
            if closest:
                dist_matrix = torch.cdist(centroids, embeddings_class)
                min_indices = dist_matrix.argmin(dim=-1)
                dataset_indices = img_ids_class[min_indices]
                if self.n_clusters == 1:
                    dataset_indices = [dataset_indices]
                closest_embedding = embeddings[dataset_indices]
                sfeat = closest_embedding if sfeat is None else torch.cat((sfeat, closest_embedding), dim=0)
            else:
                sfeat = centroids if sfeat is None else torch.cat((sfeat, centroids), dim=0)

        slabel = torch.tensor(slabel)
        return sfeat, slabel

    def _verify_sy(self, sy):
        # unique_labels, counts = torch.unique(sy, return_counts=True)
        # assert torch.equal(unique_labels, torch.arange(self.n_classes))
        # assert torch.equal(counts, torch.ones_like(unique_labels) * self.num_per_class) 
        pass