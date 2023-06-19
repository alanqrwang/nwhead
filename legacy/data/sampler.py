import numpy as np

class QuerySampler:
  def __init__(self, indices,
               classes=None, 
               ):
    self.indices = indices
    if classes:
      self.indices = self.indices[classes]

  def __iter__(self):
    self.query_iter = iter(np.random.permutation(self.flatten(self.indices)))
    return self

  def __len__(self):
    return len(self.flatten(self.indices))
  
  def __next__(self):
    return next(self.query_iter)
  
  def flatten(self, t):
    return [item for sublist in t for item in sublist]

class RandomSupportSampler:
  def __init__(self, indices,
               num_supp_per_class, 
               classes=None, 
               ):
    self.indices = indices
    self.num_supp_per_class = num_supp_per_class
    if classes:
      self.indices = self.indices[classes]

  def __iter__(self):
    return self
  
  def __next__(self):
    support_idxs = np.array([np.random.choice(row, size=self.num_supp_per_class, replace=False) for row in self.indices]).flatten()
    return support_idxs
  
class QuerySupportSampler:
  '''
  Returns indices of query and support for a single
  __getitem__ call, where query is an iterator through
  the entire dataset, and support is randomly sampled
  from the dataset. Thus, there can be overlap in the 
  indices.
  '''
  def __init__(self, 
               query_sampler,
               support_sampler,
               ):
    self.query_sampler = query_sampler
    self.support_sampler = support_sampler

  def __iter__(self):
    self.query_iter = iter(self.query_sampler)
    self.support_iter = iter(self.support_sampler)
    return self

  def __len__(self):
    return len(self.query_sampler)
  
  def __next__(self):
    query_idx = next(self.query_iter)
    support_idxs = next(self.support_iter)
    return query_idx, support_idxs

class SubsampleQuerySupportSampler:
  def __init__(self, 
               query_indices,
               support_indices,
               num_supp_per_class, 
               subsample_size=10,
               classes_in_query=None, 
               classes_in_support=None, 
               ):
    self.query_indices = query_indices
    self.support_indices = support_indices
    self.num_supp_per_class = num_supp_per_class
    self.subsample_size = subsample_size
    self.classes_in_query = classes_in_query
    self.classes_in_support = classes_in_support
  
  def __iter__(self):
    return self

  def __len__(self):
    return len(self.flatten(self.query_indices)) # TODO: don't know what to put here lol
  
  def __next__(self):
    # Sample set of classes
    num_classes = len(self.support_indices)
    classes = np.random.choice(num_classes, size=self.subsample_size, replace=False)
    query_indices = [self.query_indices[i] for i in classes]
    support_indices = [self.support_indices[i] for i in classes]

    # Sample support images from set
    support_idxs = np.array([np.random.choice(row, size=self.num_supp_per_class, replace=False) for row in support_indices]).flatten()

    # Sample query image from set
    query_idx = np.random.choice(self.flatten(query_indices))
    return query_idx, support_idxs

  def flatten(self, t):
    return [item for sublist in t for item in sublist]

class QuerySupportBatchSampler:
  '''Samples a batch for queries and a batch for support.'''
  def __init__(self, 
               query_indices,
               support_indices,
               batch_size,
               subsample_size,
               num_supp_per_class=1, 
               classes_in_query=None, 
               classes_in_support=None, 
               ):
    self.query_indices = query_indices
    self.support_indices = support_indices
    self.num_supp_per_class = num_supp_per_class
    self.batch_size = batch_size
    self.subsample_size = subsample_size
    self.classes_in_query = classes_in_query
    self.classes_in_support = classes_in_support
  
  def __iter__(self):
    return self

  def __len__(self):
    return len(self.flatten(self.query_indices)) # TODO: don't know what to put here lol
  
  def __next__(self):
    # Sample set of labels for support labels.
    num_classes = len(self.query_indices)
    support_classes = np.random.choice(num_classes, size=self.subsample_size, replace=False)
    possible_support_indices = [self.support_indices[i] for i in support_classes]
    
    # Sample subset of possible support indices as query indices
    query_idxs = np.random.choice(self.flatten(possible_support_indices), size=self.batch_size, replace=False)

    # Sample support images from set
    support_idxs = np.array([np.random.choice(row, size=self.num_supp_per_class, replace=False) for row in possible_support_indices]).flatten()

    return np.concatenate((query_idxs, support_idxs))

  def flatten(self, t):
    return [item for sublist in t for item in sublist]
