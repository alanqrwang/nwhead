import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from .support import SupportSet
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
                 num_classes,
                 support_dataset=None, 
                 feat_dim=None,
                 kernel_type='euclidean', 
                 train_type='random', 
                 num_per_class=1, 
                 total_per_class=100,
                 subsample_classes=None,
                 num_clusters=3,
                 proj_dim=0, 
                 env_array=None, 
                 debug_mode=False,
                 device='cuda:0', 
                 do_held_out_training=False,
                 held_out_class=None,
                 random_dropout=False,
                 use_nis_embedding=False,
                 dtype=torch.float32):
        '''
        Top level NW net class. Creates kernel, NWHead, and SupportSet as modules.

        :param featurizer: Feature extractor
        :param support_dataset: Pytorch Dataset object. Assumes has attributes
            .targets containing categorical classes of dataset.
        :param num_classes: Number of classes in dataset
        :param feat_dim: Output dimension of featurizer
        :param kernel_type: Type of kernel to use
        :param train_type: Type of training strategy
        :param num_per_class: Number of datapoints per class to sample for support
            during training
        :param total_per_class: Number of datapoints per class to use for full
            inference
        :param subsample_classes: Subsample number of classes to put in support
            (use for large number of classes)
        :param num_clusters: Number of cluster centroids per class for cluster inference
        :param proj_dim: If > 0, adds a linear projection down to proj_dim after featurizer
        :param env_array: Array of same length as support dataset containing
            environment indicators
        :param debug_mode: If set, prints some debugging info and plots images
        :param device: Device used for computation
        '''
        super(NWNet, self).__init__()
        self.featurizer = featurizer
        self.train_type = train_type
        self.device = device
        self.debug_mode = debug_mode
        self.num_classes = num_classes
        self.use_nis_embedding = use_nis_embedding
        self.num_per_class = num_per_class
        if support_dataset is not None:
            assert hasattr(support_dataset, 'targets'), 'Support set must have .targets attribute'

        # Kernel
        kernel = get_kernel(kernel_type)

        # NW Head
        self.nwhead = NWHead(kernel=kernel,
                             feat_dim=feat_dim,
                             proj_dim=proj_dim)

        # Support dataset
        if support_dataset is not None:
            self.sset = SupportSet(support_dataset,
                               train_type,
                               num_per_class,
                               total_per_class,
                               self.num_classes,
                               num_clusters=num_clusters,
                               subsample_classes=subsample_classes,
                               env_array=env_array,
                               do_held_out_training=do_held_out_training,
                               held_out_class=held_out_class,
                               random_dropout=random_dropout)

        # NIS embedding
        if self.use_nis_embedding:
            factory_kwargs = {'device': device, 'dtype': dtype}
            assert feat_dim is not None, 'Feature dimension must be specified'
            self.nis_embedding = nn.Parameter(torch.empty((1, feat_dim), **factory_kwargs))
            xavier_uniform_(self.nis_embedding)
            self.nis_class = self.num_classes

    def precompute(self):
        '''Precomputes all support features, cluster centroids, and 
           random iterator. Call before running inference.'''
        assert not self.featurizer.training
        sinfo = self._compute_all_support_feats()
        self.sset.update_feats(*sinfo)

    def predict(self, x, mode='random', support_data=None):
        '''
        Perform prediction given test images.
        
        :param x: Input datapoints (bs, nch, l, w)
        :param mode: Inference mode. One of ['random', 'full', 'cluster', 'ensemble']
        :param support_data: Optional (sx, sy, sm) tuple for functional implementation
        '''
        qfeat = self.featurizer(x)
        if support_data is not None:
            sx, sy, sm = support_data
            sfeat = self.featurizer(sx)
        else:
            sfeat, sy = self.sset.get_infer_support(mode)

        if self.use_nis_embedding:
            nis_embedding = self.nis_embedding.expand(self.num_per_class, sfeat.shape[-1]).to(sfeat.device)
            sfeat = torch.cat((sfeat, nis_embedding), dim=0)
            nis_class = torch.full((self.num_per_class,), self.nis_class).to(sy.device)
            sy = torch.cat((sy, nis_class))

        # if self.debug_mode:
        #     print('qx shape:', x.shape)
        #     print('sfeat shape:', sfeat.shape)
        #     print('sy:', torch.unique(sy))

        if mode == 'ensemble':
            outputs = 0
            num_envs = len(sfeat)
            for env_feat, env_y in zip(sfeat, sy):
                env_feat, env_y = env_feat.to(
                    x.device), env_y.to(x.device)
                output = self._forward(qfeat, env_feat, env_y)
                outputs += output.exp()
            return torch.log(outputs / num_envs)#, torch.full((len(x),), True)
        else:
            sfeat, sy = sfeat.to(x.device), sy.to(x.device)
            return self._forward(qfeat, sfeat, sy)#, torch.full((len(x),), True)

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
            sx, sy, sm = self.sset.get_train_support(y, metadata)
        sx, sy = sx.to(x.device), sy.to(x.device)

        batch_size = len(x)
        inputs = torch.cat((x, sx), dim=0)
        feats = self.featurizer(inputs)
        qfeat, sfeat = feats[:batch_size], feats[batch_size:]
        
        isin = torch.isin(y, sy)

        if self.use_nis_embedding:
            # Append NIS embedding and label to support set
            nis_embedding = self.nis_embedding.expand(self.num_per_class, sfeat.shape[-1])
            sfeat = torch.cat((sfeat, nis_embedding), dim=0)
            nis_class = torch.full((self.num_per_class,), self.nis_class).to(sy.device)
            sy = torch.cat((sy, nis_class))
            # Set query label as NIS class if not in support set
            y[~isin] = self.nis_class 

        if self.debug_mode:
            print('qx shape:', x.shape)
            print('sx shape:', sx.shape)
            print('qfeat shape:', qfeat.shape)
            print('sfeat shape:', sfeat.shape)
            print('qy:', y)
            print('sy:', sy)
            print('qy in sy:', isin)
            if metadata is not None:
                print('qmeta:', metadata)
                print('smeta:', sm)
            if plt:
                qgrid = torchvision.utils.make_grid(linear_normalization(x), nrow=8)
                sgrid = torchvision.utils.make_grid(linear_normalization(sx), nrow=8)
                plt.imshow(qgrid.permute(1, 2, 0).cpu().detach().numpy())
                plt.show()
                plt.imshow(sgrid.permute(1, 2, 0).cpu().detach().numpy())
                plt.show()

        return self._forward(qfeat, sfeat, sy)#, isin
    
    def _forward(self, qfeat, sfeat, sy):
        '''Forward pass on features'''
        batch_size = len(qfeat)
        if self.use_nis_embedding:
            sy = F.one_hot(sy, self.num_classes+1).float()
        else:
            sy = F.one_hot(sy, self.num_classes).float()
        sfeat = sfeat[None].expand(batch_size, *sfeat.shape)
        sy = sy[None].expand(batch_size, *sy.shape)
        output = self.nwhead(qfeat, sfeat, sy)
        return torch.log(output+1e-12)

    def _compute_all_support_feats(self, save=False):
        feats = []
        labels = []
        meta = []
        separated_feats = []
        separated_labels = []
        separated_meta = []
        for loader in self.sset.support_loaders:
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

class NWHead(nn.Module):
    def __init__(self, 
                 kernel, 
                 feat_dim=None,
                 proj_dim=0, 
                 device='cuda:0', 
                 dtype=torch.float32):
        super(NWHead, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.kernel = kernel
        self.proj_dim = proj_dim

        if self.proj_dim > 0:
            assert feat_dim is not None, 'Feature dimension must be specified'
            self.proj_weight = nn.Parameter(torch.empty((1, feat_dim, proj_dim), **factory_kwargs))
            xavier_uniform_(self.proj_weight)
        
    def project(self, x):
        bs = len(x)
        return torch.bmm(x, self.proj_weight.repeat(bs, 1, 1))

    def forward(self, x, support_x, support_y):
        """
        Computes Nadaraya-Watson head given query x, support x, and support y tensors.
        :param x: (b, proj_dim)
        :param support_x: (b, num_support, proj_dim)
        :param support_y: (b, num_support, num_classes)
        :return: softmaxed probabilities (b, num_classes)
        """
        x = x.unsqueeze(1)
        if self.proj_dim > 0:
            x = self.project(x)
            support_x = self.project(support_x)

        scores = self.kernel(x, support_x)

        probs = F.softmax(scores, dim=-1)
        # (B, num_queries, num_keys) x (B, num_keys=num_vals, num_classes) -> (B, num_queries, num_classes)
        output = torch.bmm(probs, support_y)
        return output.squeeze(1)
