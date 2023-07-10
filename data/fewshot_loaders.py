import numpy as np
import torch

def get_fewshot_loaders(train_dataset, val_dataset, 
                        do_held_out_training, 
                        held_out_class, 
                        batch_size,
                        workers):
    # If held_out_class is specified, then we need to use a custom sampler for training
    if do_held_out_training:
        train_sampler = HeldOutSampler(train_dataset, shuffle=True, heldout=False, held_out_class=held_out_class)
        train_loader = torch.utils.data.DataLoader(
          train_dataset, batch_size=batch_size,
          num_workers=workers, pin_memory=True, sampler=train_sampler)
        # heldout_train_sampler = HeldOutSampler(train_dataset, shuffle=True, heldout=True, held_out_class=args.held_out_class)
        # heldout_train_loader = torch.utils.data.DataLoader(
        #   train_dataset, batch_size=args.batch_size,
        #   num_workers=args.workers, pin_memory=True, sampler=heldout_train_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(
          train_dataset, batch_size=batch_size, shuffle=True,
          num_workers=workers, pin_memory=True)

    # For validation, always use custom sampler
    val_sampler = HeldOutSampler(val_dataset, shuffle=False, heldout=False, held_out_class=held_out_class)
    val_loader = torch.utils.data.DataLoader(
      val_dataset, batch_size=batch_size,
      num_workers=workers, pin_memory=True, sampler=val_sampler)
    heldout_val_sampler = HeldOutSampler(val_dataset, shuffle=False, heldout=True, held_out_class=held_out_class)
    heldout_val_loader = torch.utils.data.DataLoader(
      val_dataset, batch_size=batch_size,
      num_workers=workers, pin_memory=True, sampler=heldout_val_sampler)
    
    return train_loader, val_loader, heldout_val_loader

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

class HeldOutSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, shuffle=False, heldout=False, held_out_class=None):
        self.dataset = dataset
        y_array = dataset.targets

        self.indices = get_separated_indices(y_array)
        if heldout:
            self.indices = self.indices[held_out_class]
        else:
            del self.indices[held_out_class]
            self.indices = [l for sublist in self.indices for l in sublist]
        
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)