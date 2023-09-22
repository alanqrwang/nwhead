import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import hnswlib
from sklearn.cluster import KMeans

class DatasetMetadata(Dataset):
    def __init__(self, dataset, metadata):
        super().__init__()
        self.dataset = dataset
        self.targets = self.dataset.targets
        self.metadata = metadata

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        datum = self.dataset[idx]
        return datum[0], datum[1], self.metadata[idx]

class FeatureDataset(Dataset):
    def __init__(self, features, targets, metadata):
        super().__init__()
        self.features = features
        self.targets = targets
        self.metadata = metadata

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.metadata[idx]

class FullDataset(Dataset):
    """Dataset used by full evaluation."""
    def __init__(self, underlying_dataset, n_shot_full):
        super().__init__()
        self.underlying_dataset = underlying_dataset
        y_array = underlying_dataset.targets
        self.indices = get_separated_indices(y_array)

        # Ensures that classes are balanced
        min_length = min([len(l) for l in self.indices])
        n_shot_full = min(n_shot_full, min_length)

        self.keys = []
        for l in self.indices:
            self.keys += l[:n_shot_full]

    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)

class RandomLoader(DataLoader):
    '''Use for regression tasks.'''
    def __init__(self, dataset, total_samples):
        self.dataset = dataset
        self.total_samples = total_samples
        super(RandomLoader, self).__init__(dataset)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        self.i += 1
        if self.i > self.total_samples:
            raise StopIteration
        support_idxs = [self.i]
        return self.collate_fn([self.dataset[i] for i in support_idxs])

    def next(self):
        return self.__next__()

class InfiniteRandomLoader(DataLoader):
    def __init__(self,
                 dataset,
                 num_per_batch,
                 ):
        self.dataset = dataset
        self.num_per_batch = num_per_batch
        super(InfiniteRandomLoader, self).__init__(dataset)

    def __iter__(self):
        return self

    def __next__(self):
        support_idxs = np.random.choice(len(self.dataset), size=self.num_per_batch, replace=False)
        return self.collate_fn([self.dataset[i] for i in support_idxs])

    def next(self):
        return self.__next__()

class InfiniteUniformClassLoader(DataLoader):
    def __init__(self,
                 dataset,
                 n_shot,
                 n_way=None,
                 ):
        self.dataset = dataset
        y_array = dataset.targets
        self.indices = get_separated_indices(y_array)
        self.n_classes = len(self.indices)
        self.n_shot = n_shot
        self.n_way = n_way
        if n_way:
            assert n_way <= len(self.indices)

        super(InfiniteUniformClassLoader, self).__init__(dataset)

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def next(self, qy=None):
        if self.n_way:
            assert len(qy) <= self.n_way, "qy must be smaller than n_way"
            qy = qy.cpu().detach().numpy()
            probs = np.ones(len(self.indices))
            probs[qy] = 0
            probs /= probs.sum()
            subclasses = np.random.choice(self.n_classes, size=(self.n_way-len(qy)), replace=False, p=probs)

            subclasses = np.concatenate([subclasses, qy])
            indices = [self.indices[i] for i in subclasses] 
        else:
            indices = self.indices

        support_idxs = np.array([np.random.choice(
            row, size=self.n_shot, replace=False) for row in indices]).flatten()

        # Get support data from dataset and collate into mini-batch
        return self.collate_fn([self.dataset[i] for i in support_idxs])

def get_separated_indices(vals):
    '''
    Separates a list of values into a list of lists,
    where each list is the indices of a fixed label/attribute.

    Maps labels/attributes to natural numbers, if needed.
    
    E.g. [0, 1, 1, 2, 3] -> [[0], [1, 2], [3], [4]]
    '''
    if torch.is_tensor(vals):
        vals = vals.cpu().detach().numpy()
    num_unique_vals = len(np.unique(vals))
    # Map (potentially non-consecutive) labels to (consecutive) natural numbers
    d = dict([(y,x) for x,y in enumerate(sorted(set(vals)))])
    indices = [[] for _ in range(num_unique_vals)]
    for i, c in enumerate(vals):
        indices[d[c]].append(i)
    return indices

def linear_normalization(arr, new_range=(0, 1)):
    """Linearly normalizes a batch of images into new_range
    arr: (batch_size, n_ch, l, w)
    """
    bs, nch, _, _ = arr.shape
    flat_arr = torch.flatten(arr, start_dim=2, end_dim=3)
    max_per_batch, _ = torch.max(flat_arr, dim=2, keepdim=True) 
    min_per_batch, _ = torch.min(flat_arr, dim=2, keepdim=True) 

    # Handling the edge case of image of all 0's
    max_per_batch[max_per_batch==0] = 1 

    max_per_batch = max_per_batch.view(bs, nch, 1, 1)
    min_per_batch = min_per_batch.view(bs, nch, 1, 1)

    return (arr - min_per_batch) * (new_range[1]-new_range[0]) / (max_per_batch - min_per_batch) + new_range[0]

class KNN:
    '''KNN.'''
    def __init__(self, data, labels, n_neighbors=20) -> None:
        self.data = data
        self.labels = labels
        self.n_neighbors = n_neighbors
    
    def __call__(self, x):
        '''Query for nearest neighbors'''
        distances = -torch.cdist(x, self.data.to(x.device))
        indices = torch.argsort(distances, dim=-1, descending=True).cpu().detach().numpy()
        indices = indices[:, :self.n_neighbors]

        data = torch.cat([self.data[ind] for ind in indices], dim=0)
        labels = torch.cat([self.labels[ind] for ind in indices], dim=0)
        return data, labels

class HNSW:
    '''HNSW index for fast approximate nearest neighbor search.'''
    def __init__(self, data, labels, n_neighbors=20) -> None:
        self.data = data
        self.labels = labels
        self.n_neighbors = n_neighbors
        # Create an HNSW index
        num_elements, self.dim = data.shape
        self.index = hnswlib.Index(space='l2', dim=self.dim)

        # Initialize the index and add data points
        self.index.init_index(max_elements=num_elements, ef_construction=100, M=16)
        self.index.add_items(data)
    
    def __call__(self, x):
        '''Query for nearest neighbors'''
        x = x.cpu().detach().numpy()
        indices, _ = self.index.knn_query(x, k=self.n_neighbors)
        indices = indices.astype(np.int64)
        data = torch.cat([self.data[ind] for ind in indices], dim=0)
        labels = torch.cat([self.labels[ind] for ind in indices], dim=0)
        return data, labels

def compute_clusters(embeddings, labels, n_clusters, closest=False):
    '''Performs k-means clustering to find support set.
    
    :param closest: If True, uses support features closest to cluster centroids. Otherwise,
                uses true cluster centroids.
    '''
    img_ids = np.arange(len(embeddings))
    sfeat = []
    slabel = []
    for c in np.unique(labels):
        embeddings_class = embeddings[labels==c]
        img_ids_class = img_ids[labels==c]
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings_class)
        centroids = torch.tensor(kmeans.cluster_centers_).float()
        slabel += [c] * n_clusters 
        if closest:
            dist_matrix = torch.cdist(centroids, embeddings_class)
            min_indices = dist_matrix.argmin(dim=-1)
            dataset_indices = img_ids_class[min_indices]
            if n_clusters == 1:
                dataset_indices = [dataset_indices]
            closest_embedding = embeddings[dataset_indices]
            sfeat.append(closest_embedding)
        else:
            sfeat.append(centroids)

    sfeat = torch.cat(sfeat, dim=0)
    slabel = torch.tensor(slabel)
    return sfeat, slabel