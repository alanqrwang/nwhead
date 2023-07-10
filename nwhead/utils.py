import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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

class UniformClassLoader(DataLoader):
    def __init__(self, dataset, total_per_class):
        self.dataset = dataset
        y_array = dataset.targets

        self.indices = get_separated_indices(y_array)
        self.total_per_class = total_per_class
        super(UniformClassLoader, self).__init__(dataset)

    def __len__(self):
        return self.total_per_class

    def __iter__(self):
        self.i = 0
        self.class_iters = [iter(l) for l in self.indices]
        return self

    def __next__(self):
        self.i += 1
        if self.i > self.total_per_class:
            raise StopIteration

        indices = [next(l) for l in self.class_iters]
        # Get support data from dataset and collate into mini-batch
        return self.collate_fn([self.dataset[i] for i in indices])

    def next(self):
        return self.__next__()

class InfiniteUniformClassLoader(DataLoader):
    def __init__(self,
                 dataset,
                 num_per_class,
                 subsample_classes=None,
                 do_held_out_training=False,
                 held_out_class=None,
                 random_dropout=False
                 ):
        self.dataset = dataset
        y_array = dataset.targets
        self.indices = get_separated_indices(y_array)
        if do_held_out_training:
            del self.indices[held_out_class]
        self.num_classes = len(self.indices)
        self.subsample_classes = subsample_classes
        if subsample_classes:
            assert subsample_classes <= len(self.indices)
        self.random_dropout = random_dropout

        self.num_per_class = num_per_class
        super(InfiniteUniformClassLoader, self).__init__(dataset)

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def next(self, qy=None):
        if self.subsample_classes:
            qy = qy.cpu().detach().numpy()
            probs = np.ones(len(self.indices))
            probs[qy] = 0
            probs /= probs.sum()
            subclasses = np.random.choice(self.num_classes, size=self.subsample_classes, replace=False, p=probs)

            if self.random_dropout:
                drop_idx = np.random.choice(len(qy))
                qy = np.delete(qy, drop_idx)
            subclasses = np.concatenate([subclasses, qy])
            indices = [self.indices[i] for i in subclasses] 
        else:
            indices = self.indices

        support_idxs = np.array([np.random.choice(
            row, size=self.num_per_class, replace=False) for row in indices]).flatten()

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