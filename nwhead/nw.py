import torch
import torch.nn.functional as F
import torch.nn as nn
from .support import SupportSetTrain, SupportSetEval
from .kernel import get_kernel
from .utils import linear_normalization
from torch.nn.init import xavier_uniform_

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

class NWNet(nn.Module):
    def __init__(self, 
                 featurizer, 
                 n_classes,
                 support_dataset=None, 
                 feat_dim=None,
                 proj_dim=0, 
                 kernel_type='euclidean', 
                 train_type='random', 
                 n_way=None,
                 n_shot=1, 
                 n_shot_random=1, 
                 n_shot_full=100,
                 n_shot_cluster=1,
                 n_neighbors=10,
                 env_array=None, 
                 debug_mode=False,
                 device='cuda:0',
                 return_mask=False,
                 ):
        '''
        Top level NW net class. Creates kernel, NWHead, and SupportSet as modules.

        :param featurizer: Feature extractor
        :param n_classes: Number of classes in dataset
        :param support_dataset: Pytorch Dataset object. Assumes has attributes
            .targets containing categorical classes of dataset.
        :param feat_dim: Output dimension of featurizer
        :param proj_dim: If > 0, adds a linear projection down to proj_dim after featurizer
        :param kernel_type: Type of kernel to use
        :param train_type: Type of training strategy, choose from ['random', 'irm']
        :param n_way: Number of classes to put in support during training
            (use for large number of classes)
        :param n_shot: Number of datapoints per class to sample for support
            during training
        :param n_shot_full: Number of datapoints per class to use for full
            inference
        :param n_clusters: Number of cluster centroids per class for cluster inference
        :param n_neighbors: Number of neighbors for knn or hnsw inference
        :param env_array: Array of same length as support dataset containing integer
            environment indicators
        :param debug_mode: If true, prints some debugging info and plots images
        :param device: Device used for computation
        :param return_mask: If true, returns mask of same size as batch_size indicating
            if query class is present in support.
        '''
        super(NWNet, self).__init__()
        self.featurizer = featurizer
        self.train_type = train_type
        self.n_way = n_way
        self.debug_mode = debug_mode
        self.n_classes = n_classes
        self.n_shot = n_shot
        self.n_shot_random = n_shot_random
        self.n_shot_full = n_shot_full
        self.n_shot_cluster = n_shot_cluster
        self.n_neighbors = n_neighbors
        self.env_array = env_array
        self.device = device
        self.return_mask = return_mask
        if support_dataset is not None:
            assert hasattr(support_dataset, 'targets'), 'Support set must have .targets attribute'

        # Kernel
        self.kernel = get_kernel(kernel_type)

        # NW Head
        self.nwhead = NWHead(kernel=self.kernel,
                             n_classes=n_classes,
                             feat_dim=feat_dim,
                             proj_dim=proj_dim,
                            )

        # Support dataset
        if support_dataset is not None:
            self.support_train = SupportSetTrain(support_dataset,
                                      self.n_classes,
                                      self.train_type,
                                      self.n_shot,
                                      n_way=self.n_way,
                                      env_array=self.env_array,
                                      )
            self.support_eval = SupportSetEval(support_dataset,
                                      self.n_classes,
                                      self.n_shot_random,
                                      self.n_shot_full,
                                      n_shot_cluster=self.n_shot_cluster,
                                      n_neighbors=self.n_neighbors,
                                      env_array=self.env_array,
                                      )

    def process_support_eval(self, support_dataset):
        '''Processes support dataset into SupportSet object.'''
        self.support_eval = SupportSetEval(support_dataset,
                                      self.n_classes,
                                      self.n_shot_random,
                                      self.n_shot_full,
                                      n_shot_cluster=self.n_shot_cluster,
                                      n_neighbors=self.n_neighbors,
                                      env_array=self.env_array,
                                      )

    def precompute(self):
        '''Precomputes all support features, cluster centroids, and 
           random iterator. Call before running inference.'''
        assert not self.featurizer.training
        sinfo = self._compute_all_support_feats()
        self.full_feat = sinfo[0]
        self.full_y = sinfo[1]
        self.support_eval.build_infer_iters(*sinfo)

    def predict(self, x, mode='random'):
        '''
        Perform prediction given test images.
        
        :param x: Input datapoints (bs, nch, l, w)
        :param mode: Inference mode. 
            One of ['random', 'full', 'cluster', 'ensemble', 'knn', 'hnsw']
        '''
        qfeat = self.featurizer(x)
        sfeat, sy = self.support_eval.get_support(mode, x=qfeat)

        if self.debug_mode:
            print('qx shape:', x.shape)
            print('sfeat shape:', sfeat.shape)
            print('sy:', sy)

        if mode == 'ensemble':
            outputs = 0
            num_envs = len(sfeat)
            for env_feat, env_y in zip(sfeat, sy):
                env_feat, env_y = env_feat.to(
                    x.device), env_y.to(x.device)
                output = self.nwhead(qfeat, env_feat, env_y)
                outputs += output.exp()
            if self.return_mask:
                return torch.log(outputs / num_envs), torch.full((len(x),), True)
            else:
                return torch.log(outputs / num_envs)
        else:
            sfeat, sy = sfeat.to(x.device), sy.to(x.device)
            if self.return_mask:
                return self.nwhead(qfeat, sfeat, sy), torch.full((len(x),), True)
            else:
                return self.nwhead(qfeat, sfeat, sy)

    def forward(self, x, y, metadata=None, support_data=None):
        '''
        Forward pass using images for query and support.
        Support set is some random subset of support dataset.
        
        :param x: Input datapoints (bs, nch, l, w)
        :param y: Corresponding labels (bs)
        :param metadata: Corresponding metadata (bs)
        :param support_data: Optional (sx, sy, sm) tuple for functional implementation
        '''
        if support_data is not None:
            sx, sy, sm = support_data
        else:
            sx, sy, sm = self.support_train.get_support(y)
        if sm is None:
            sm = torch.zeros_like(sy)

        sx, sy, sm = sx.to(x.device), sy.to(x.device), sm.to(x.device)

        batch_size = len(x)
        inputs = torch.cat((x, sx), dim=0)
        feats = self.featurizer(inputs)
        qfeat, sfeat = feats[:batch_size], feats[batch_size:]

        isin = torch.isin(y, sy)

        if self.debug_mode:
            print('qx shape:', x.shape)
            print('sx shape:', sx.shape)
            print('qfeat shape:', qfeat.shape)
            print('sfeat shape:', sfeat.shape)
            print('qy:', y)
            print('sy:', sy)
            print('qy in sy:', isin)
            print(f'Percent query dropped: {(1.0 - isin.float().mean().item())*100}%')
            if metadata is not None:
                print('qmeta:', metadata)
                print('smeta:', sm)
            # if plt:
            #     qgrid = torchvision.utils.make_grid(linear_normalization(x), nrow=8)
            #     sgrid = torchvision.utils.make_grid(linear_normalization(sx), nrow=8)
            #     plt.imshow(qgrid.permute(1, 2, 0).cpu().detach().numpy())
            #     plt.show()
            #     plt.imshow(sgrid.permute(1, 2, 0).cpu().detach().numpy())
            #     plt.show()

        if self.return_mask:
            return self.nwhead(qfeat, sfeat, sy), isin
        else:
            return self.nwhead(qfeat, sfeat, sy)
    
    def _compute_all_support_feats(self):
        feats = []
        labels = []
        meta = []
        separated_feats = []
        separated_labels = []
        separated_meta = []
        for loader in self.support_eval.support_loaders:
            env_feats = []
            env_labels = []
            env_meta = []
            for qimg, qlabel, qmeta in loader:
                qimg = qimg.to(self.device)
                feat = self.featurizer(qimg).cpu().detach()
                feats.append(feat)
                labels.append(qlabel.cpu().detach())
                meta.append(qmeta)
                env_feats.append(feat)
                env_labels.append(qlabel.cpu().detach())
                env_meta.append(qmeta)
            env_feats = torch.cat(env_feats, dim=0)
            env_labels = torch.cat(env_labels, dim=0)
            env_meta = torch.cat(env_meta, dim=0)
            separated_feats.append(env_feats)
            separated_labels.append(env_labels)
            separated_meta.append(env_meta)
        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)
        meta = torch.cat(meta, dim=0)

        return feats, labels, meta, separated_feats, separated_labels, separated_meta
    
    def get_neighbors(self, x):
        '''Returns indices of nearest neighbors of x in support set.'''
        qfeat = self.featurizer(x).cpu().detach()
        distances = self.kernel(qfeat, self.full_feat)
        return torch.argsort(distances, dim=-1, descending=True)

    # def get_influences(self, x):
    #     '''Returns support influence of x in support set.'''
    #     distances = -torch.cdist(x, self.full_feat)
    #     return self.support_set.get_neighbors()

class NWHead(nn.Module):
    def __init__(self, 
                 kernel, 
                 n_classes,
                 feat_dim=None,
                 proj_dim=0, 
                 device='cuda:0', 
                 dtype=torch.float32):
        super(NWHead, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.kernel = kernel
        self.n_classes = n_classes
        self.proj_dim = proj_dim

        if self.proj_dim > 0:
            assert feat_dim is not None, 'Feature dimension must be specified'
            self.proj_weight = nn.Parameter(torch.empty((1, feat_dim, proj_dim), **factory_kwargs))
            xavier_uniform_(self.proj_weight)

    def project(self, x):
        bs = len(x)
        return torch.bmm(x, self.proj_weight.repeat(bs, 1, 1))

    def forward(self, x, sx, sy):
        """
        Computes Nadaraya-Watson head given query x, support x, and support y tensors.
        :param x: Query data (b, feat_dim)
        :param sx: Support data (num_support, feat_dim) or (b, num_support, feat_dim)
            If b not provided, then uses same support set for all elements of mini-batch.
        :param sy: Support targets (num_support) or (b, num_support)
        :return: log of softmaxed probabilities (b, num_classes)
        """
        batch_size = len(x)
        sy = F.one_hot(sy, self.n_classes).float()
        if len(sx.shape) == len(x.shape):
            sx = sx[None].expand(batch_size, *sx.shape)
            sy = sy[None].expand(batch_size, *sy.shape)

        x = x.unsqueeze(1) # Create num_query dimension
        if self.proj_dim > 0:
            x = self.project(x)
            sx = self.project(sx)

        scores = self.kernel(x, sx)

        probs = F.softmax(scores, dim=-1)
        # (B, num_queries, num_keys) x (B, num_keys=num_vals, num_classes) -> (B, num_queries, num_classes)
        output = torch.bmm(probs, sy)
        output = output.squeeze(1) # Remove num_query dimension
        return torch.log(output+1e-12)
