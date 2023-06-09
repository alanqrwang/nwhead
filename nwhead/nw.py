import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from .support import SupportSet
from .kernel import get_kernel

class NWNet(nn.Module):
    def __init__(self, 
                 featurizer, 
                 support_dataset, 
                 num_classes,
                 kernel_type='euclidean', 
                 train_type='random', 
                 metadata_array=None, 
                 num_per_class=1, 
                 total_per_class=100,
                 embed_dim=0, 
                 use_nis=False,
                 debug_mode=False,
                 device='cuda:0', 
                 use_nll_loss=False
                 ):
        '''Top level NW net class. Creates kernel, NWHead, and support set as modules.

        Args:
            featurizer: Feature extractor
            support_dataset: Pytorch Dataset object. Assumes has attributes
                .targets containing categorical classes of dataset.
            d_out: Output dimension of featurizer'''
        super(NWNet, self).__init__()
        self.featurizer = featurizer
        self.train_type = train_type
        self.device = device
        self.debug_mode = debug_mode
        self.use_nis = use_nis
        self.use_nll_loss = use_nll_loss
        if use_nis:
            raise NotImplementedError('need to figure out how to not backprop on nis score in NWHead')
        assert hasattr(support_dataset, 'targets'), 'Support set must have .targets attribute'

        # Kernel
        kernel = get_kernel(kernel_type)

        if use_nis:
            self.num_classes = num_classes + 1 # NIS class
            self.nis_y = self.num_classes # NIS class
        else:
            self.num_classes = num_classes

        # NW Head
        self.nwhead = NWHead(kernel=kernel,
                             embed_dim=embed_dim,
                             use_nis=use_nis)

        # Support dataset
        self.sset = SupportSet(support_dataset,
                               train_type,
                               num_per_class,
                               total_per_class,
                               self.num_classes,
                               metadata_array=metadata_array)

    def precompute(self):
        '''Precomputes all support features, cluster centroids, and 
           random iterator. Call before running inference.'''
        assert not self.featurizer.training
        sinfo = self._compute_all_support_feats()
        self.sset.update_feats(*sinfo)

    def predict(self, x, mode='random'):
        qfeat = self.featurizer(x)
        sfeat, sy = self.sset.get_infer_support(mode)
        if mode == 'ensemble':
            outputs = 0
            num_envs = len(sfeat)
            for env_feat, env_y in zip(sfeat, sy):
                env_feat, env_y = env_feat.to(
                    x.device), env_y.to(x.device)
                if self.use_nis:
                    nis_y = torch.tensor(self.nis_y).to(x.device).view(1)
                    env_y = torch.cat([env_y, nis_y])
                output = self._forward(qfeat, env_feat, env_y)
                if self.use_nll_loss:
                    outputs += output.exp()
                else:
                    outputs += output.log()
            if self.use_nll_loss:
                return torch.log(outputs / num_envs), torch.full((len(x),), True)
            else:
                return (outputs / num_envs).exp(), torch.full((len(x),), True)
        else:
            sfeat, sy = sfeat.to(x.device), sy.to(x.device)
            return self._forward(qfeat, sfeat, sy), torch.full((len(x),), True)

    def forward(self, x, y, metadata=None):
        '''Forward pass using images for query and support.
        Support set is some random subset of support dataset.'''
        sx, sy, sm = self.sset.get_train_support(metadata)
        sx, sy = sx.to(x.device), sy.to(x.device)

        batch_size = len(x)
        inputs = torch.cat((x, sx), dim=0)
        feats = self.featurizer(inputs)
        qfeat, sfeat = feats[:batch_size], feats[batch_size:]
        
        isin = torch.isin(y, sy)
        if self.use_nis:
            # If query has no matching class in support, set it as NIS class
            y[~isin] = self.nis_y

        if self.debug_mode:
            qgrid = torchvision.utils.make_grid(self.linear_normalization(x), nrow=8)
            sgrid = torchvision.utils.make_grid(self.linear_normalization(sx), nrow=8)
            print('qbatch shape:', x.shape)
            print('sbatch shape:', sx.shape)
            # qgrid = torchvision.utils.make_grid(x, nrow=8)
            # sgrid = torchvision.utils.make_grid(sx, nrow=8)
            # qgrid = torch.cat([qgrid.cpu(), torch.zeros(1, *qgrid.shape[1:])], dim=0)
            # sgrid = torch.cat([sgrid.cpu(), torch.zeros(1, *sgrid.shape[1:])], dim=0)
            plt.imshow(qgrid.permute(1, 2, 0).cpu().detach().numpy())
            plt.show()
            plt.imshow(sgrid.permute(1, 2, 0).cpu().detach().numpy())
            plt.show()
            print('qy:', y)
            print('sy:', sy)
            print('qmeta:', metadata[:, self.metadata_idx])
            print('smeta:', sm[:, self.metadata_idx])
            print('qy in sy:', isin)

        return self._forward(qfeat, sfeat, sy), isin
    
    def _forward(self, qfeat, sfeat, sy):
        '''Forward pass on features'''
        batch_size = len(qfeat)
        sy = F.one_hot(sy, self.num_classes).float()
        sfeat = sfeat[None].expand(batch_size, *sfeat.shape)
        sy = sy[None].expand(batch_size, *sy.shape)
        output = self.nwhead(qfeat, sfeat, sy)
        if self.use_nll_loss:
            return torch.log(output+1e-12)
        else:
            return output.exp()

    def _compute_all_support_feats(self, save=False):
        # print('Precomputing all features in support dataset...')
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

        # if save:
        #     print('Saving embeddings...')
        #     root_dir = os.path.join(self.run_dir, 'embedding')
        #     if not os.path.exists(root_dir):
        #         os.makedirs(root_dir)
        #     embedding_path = os.path.join(root_dir, 'embeddings.npy')
        #     label_path = os.path.join(root_dir, 'labels.npy')
        #     metadata_path = os.path.join(root_dir, 'metadata.npy')
        #     np.save(embedding_path, embeddings.cpu().detach().numpy())
        #     np.save(label_path, labels.cpu().detach().numpy())
        #     np.save(metadata_path, metadata.cpu().detach().numpy())

        return feats, labels, meta, separated_feats, separated_labels, separated_meta

class NWHead(nn.Module):
    def __init__(self, 
                 kernel, 
                 embed_dim=0, 
                 use_nis=False, 
                 device='cuda:0', 
                 dtype=torch.float32):
        super(NWHead, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.kernel = kernel
        self.embed_dim = embed_dim
        if self.embed_dim > 0:
            self.projection = nn.LazyLinear(embed_dim)
        
        self.use_nis = use_nis
        if self.use_nis:
            self.nis_token = torch.ones((1, 1, 1), requires_grad=False, **factory_kwargs)
        
    def forward(self, x, support_x, support_y):
        """
        Computes Nadaraya-Watson head on query, key and value tensors.
        Args:
            q: query (b, embed_dim)
            k: key (b, num_support, embed_dim)
            v: labels (b, num_support, num_classes)

        Output:
            output: softmaxed probabilities (b, num_classes)
            attn_probs: Weights per support element, (b, num_support)
            attn_scores: Scores per support element, (b, num_support)
        """
        x = x.unsqueeze(1)
        if self.embed_dim > 0:
            x = self.projection(x)
            support_x = self.projection(support_x)

        scores = self.kernel(x, support_x)

        if self.use_nis:
            # Add constant NIS score
            nis_score = self.nis_token.repeat(len(x), scores.shape[1], 1)
            scores = torch.cat((scores, nis_score), dim=-1)

        probs = F.softmax(scores, dim=-1)
        # (B, num_queries, num_keys) x (B, num_keys=num_vals, num_classes) -> (B, num_queries, num_classes)
        output = torch.bmm(probs, support_y)
        return output.squeeze(1)
