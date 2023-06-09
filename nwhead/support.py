import numpy as np
import random
import torch
from torch.utils.data import Dataset, ConcatDataset
from sklearn.cluster import KMeans
from .utils import DatasetMetadata, FeatureDataset, UniformClassLoader, InfiniteUniformClassLoader, InfiniteRandomLoader

class SupportSet:
    '''Support set for NW. 
    
    Args:
        support_set: Pytorch dataset with .targets attribute, and __getitem__
                     returns input and target as first two elements in tuple'''
    def __init__(self, 
                 support_set, 
                 train_type, 
                 num_per_class, 
                 total_per_class, 
                 num_classes,
                 env_array=None, 
                 include_nis=False):
        self.train_type = train_type
        self.num_per_class = num_per_class
        self.y_array = np.array(support_set.targets)
        self.include_nis = include_nis
        self.num_classes = num_classes

        # Simplest case, no environment info and single support dataset
        if env_array is None and isinstance(support_set, Dataset):
            assert train_type in ['random', 'unbalanced']
            self.env_array = np.zeros(len(support_set))
            support_set = DatasetMetadata(support_set, self.env_array)
            self.combined_dataset = support_set
            self.env_datasets = self._separate_env_datasets(support_set)
        # If env_array is provided, then support dataset should be a single
        # Pytorch Dataset. 
        elif env_array is not None:
            assert train_type in ['match', 'mixmatch']
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
        else:
            raise NotImplementedError

        self.support_loaders = self._build_support_loaders(total_per_class)

        self.train_iter, self.eval_iter = self._build_train_support_iter()

    def update_feats(self, sfeat, sy, smeta, sfeat_env, sy_env, smeta_env):
        self.full_feat = sfeat
        self.full_y = sy
        self.full_meta = smeta
        self.full_feat_sep = sfeat_env
        self.full_y_sep = sy_env
        self.full_meta_sep = smeta_env

        self.cluster_feat, self.cluster_y = self._compute_clusters()
        self.random_iter = self._build_random_support_iter()

    def get_train_support(self, env_index):
        '''Samples a support for training.'''
        if self.train_type == 'mixmatch':
            train_iter = np.random.choice(self.train_iter)
            sx, sy, sm = train_iter.next()
        elif self.train_type == 'match':
            assert torch.equal(env_index, torch.ones_like(env_index)*env_index[0])
            meta_idx = self.env_map[env_index[0].item()]
            train_iter = self.train_iter[meta_idx]
            sx, sy, sm = train_iter.next()
        elif self.train_type == 'unbalanced':
            sx, sy = self.train_iter[0].next()
            sx1, sy1 = self.train_iter[1].next()
            sx = torch.cat([sx, sx1], dim=0)
            sy = torch.cat([sy, sy1], dim=0)
        else:
            sx, sy, sm = self.train_iter.next()

        # Occasionally verify sampled classes
        if random.random() < 0.01 and self.train_type != 'unbalanced':
            self._verify_sy(sy)

        if self.include_nis:
            nis_y = torch.tensor(self.nis_y).to(sy.device).view(1)
            sy = torch.cat([sy, nis_y])
        return sx, sy, sm

    def get_infer_support(self, mode):
        '''Samples a support for inference depending on mode.'''
        if mode == 'random':
            sfeat, sy, _ = next(self.random_iter)
        elif mode == 'full':
            sfeat, sy = self.full_feat, self.full_y
        elif mode == 'cluster':
            sfeat, sy = self.cluster_feat, self.cluster_y
        elif mode == 'ensemble':
            sfeat, sy = self.full_feat_sep, self.full_y_sep
        else:
            raise NotImplementedError

        if self.include_nis:
            nis_y = torch.tensor(self.nis_y).to(sy.device).view(1)
            sy = torch.cat([sy, nis_y])
        return sfeat, sy

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

    def _build_train_support_iter(self):
        '''Iterators for random sampling during training.
        Samples images from dataset.'''
        if self.train_type == 'random':
            train_iter = InfiniteUniformClassLoader(
                self.combined_dataset, self.num_per_class)
        elif self.train_type == 'unbalanced':
            # Compute remaining images so that it matches uniform class loading
            remaining = (self.num_per_class-1)*self.num_classes
            return [InfiniteUniformClassLoader(self.combined_dataset, num_per_class=1), 
                    InfiniteRandomLoader(self.combined_dataset, num_per_batch=remaining)]
        else:
            train_iter = [iter(InfiniteUniformClassLoader(env, self.num_per_class)) for env in self.env_datasets]
        eval_iter = InfiniteUniformClassLoader(
            self.combined_dataset, self.num_per_class)
        return train_iter, eval_iter

    def _build_support_loaders(self, total_per_class=100):
        '''Support loader for precomputing features during evaluation.
        Because the model assumes balanced classes during training and
        test, the support loader samples evenly across classes.
        '''
        return [UniformClassLoader(env, total_per_class=total_per_class) for env in self.env_datasets]

    def _build_random_support_iter(self):
        '''Iterator for random sampling during evaluation.
        Samples features from precomputed feature dataset.'''
        feat_dataset = FeatureDataset(self.full_feat, self.full_y, self.full_meta)
        eval_loader = InfiniteUniformClassLoader(
            feat_dataset, self.num_per_class)
        return iter(eval_loader)

    def _compute_clusters(self, num_clusters=3, closest=False, save_nearest_imgs=False):
        '''Performs k-means clustering to find support set.
        
        Args:
        in_support: If True, uses support embeddings closest to cluster centroids. Otherwise,
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

            # if save_nearest_imgs:
            #     dist_matrix = torch.cdist(centroids, embeddings_class)
            #     min_indices = dist_matrix.argmin(dim=-1)
            #     dataset_indices = img_ids_class[min_indices]
            #     if num_clusters == 1:
            #         dataset_indices = [dataset_indices]
            #     imgs, label, _ = self.support_dataset[dataset_indices]
            #     assert (label.argmax(-1) == torch.full_like(label.argmax(-1), c)).all(), 'Label extracted from support set doesn\'t match current label'
            #     imgs = linear_normalization(imgs)
            #     save_path = os.path.join(self.img_dir, 'nearest_cluster_k={}_label={}'.format(num_clusters, c))
            #     np.save(save_path, imgs.cpu().detach().numpy())
            #     if self.debug_mode:
            #         print('label:', c)
            #         img_grid = torchvision.utils.make_grid(imgs, nrow=5)
            #         fig, axes = plt.subplots(1, 1, figsize=(5, 5))
            #         axes.axis('off')
            #         axes.imshow(img_grid.permute(1, 2, 0).cpu().detach().numpy())
            #         fig.show()
            #         plt.show()

        slabel = torch.tensor(slabel)
        return sfeat, slabel

    def _verify_sy(self, sy):
        unique_labels, counts = torch.unique(sy, return_counts=True)
        assert torch.equal(unique_labels, torch.arange(self.num_classes))
        assert torch.equal(counts, torch.ones_like(unique_labels) * self.num_per_class) 