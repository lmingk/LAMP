import math
import random
from turtle import forward
from urllib.parse import ParseResultBytes
from sklearn import multiclass

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import to_undirected
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from models.models import *
from sklearn.metrics import f1_score,accuracy_score

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)



EPS = 1e-15
LOG_VAR_MAX = 10
LOG_VAR_MIN = EPS

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

def negative_sampling(pos_edge_index, num_nodes):


    idx = (pos_edge_index[0] * num_nodes + pos_edge_index[1]) #这里是吧整个矩阵flatten
    idx = idx.to(torch.device('cpu'))

    rng = range(num_nodes**2)
    perm = torch.tensor(random.sample(rng, idx.size(0)))
    mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
    rest = mask.nonzero(as_tuple = False).view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(random.sample(rng, rest.size(0)))
        perm[rest] = tmp
        mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
        rest = mask.nonzero(as_tuple = False).view(-1)

    row, col = torch.div(perm, num_nodes,rounding_mode='floor'), perm % num_nodes   #????




    return torch.stack([row, col], dim=0).to(pos_edge_index.device)



def negative_sampling_identify(pos_edge_index, num_nodes):

    def fetch(arr,num,r = 0):
        d = int((len(arr)+0.0)/num)
        result = list(range(r,len(arr),d))
        if len(result)>num:result = result[:num]
        return result


    idx = (pos_edge_index[0] * num_nodes + pos_edge_index[1]) #这里是吧整个矩阵flatten
    idx = idx.to(torch.device('cpu'))

    rng =  [i * num_nodes+j for i in range(num_nodes) for j in range(num_nodes) if i<j ]  #range(num_nodes**2)
    perm = torch.tensor(fetch(rng, idx.size(0)))
    mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
    rest = mask.nonzero(as_tuple = False).view(-1)

    index = 0
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.tensor(fetch(rng, rest.size(0),index))
        perm[rest] = tmp
        mask = torch.from_numpy(np.isin(perm, idx).astype(np.uint8))
        rest = mask.nonzero(as_tuple = False).view(-1)
        index+=1

    row, col = torch.div(perm, num_nodes,rounding_mode='floor'), perm % num_nodes   #????




    return torch.stack([row, col], dim=0).to(pos_edge_index.device)


        
class MLPDecoder(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MLPDecoder, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        glorot(self.weight)
        zeros(self.bias)

    def forward(self,x):
        return torch.matmul(x, self.weight)+self.bias
         

    
class EdgeDecoder(torch.nn.Module):

    def __init__(self, in_channels):
        super(EdgeDecoder, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, 2))
        self.bias = nn.Parameter(torch.Tensor(2))

        glorot(self.weight)
        zeros(self.bias)

    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] - z[edge_index[1]]) @ self.weight
        if sigmoid:
            return torch.softmax(value,dim = 1)[:,1]





            






class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""

    def forward(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilties for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """


        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj







class MyGAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """

    def __init__(self, encoder, decoder=None):
        super(MyGAE, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder


        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilties."""
        return self.decoder(*args, **kwargs)

    def split_edges(self, data, val_ratio=0.05, test_ratio=0.1):
        r"""Splits the edges of a :obj:`torch_geometric.data.Data` object
        into positve and negative train/val/test edges.

        Args:
            data (Data): The data object.
            val_ratio (float, optional): The ratio of positive validation
                edges. (default: :obj:`0.05`)
            test_ratio (float, optional): The ratio of positive test
                edges. (default: :obj:`0.1`)
        """

        assert 'batch' not in data  # No batch-mode.

        row, col = data.edge_index
        data.edge_index = None

        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]

        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))

        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]

        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)

        r, c = row[n_v + n_t:], col[n_v + n_t:]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        data.train_pos_edge_index = to_undirected(data.train_pos_edge_index)

        # Negative edges.
        num_nodes = data.num_nodes
        neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1)
        neg_adj_mask[row, col] = 0

        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple = False).t()
        perm = torch.tensor(random.sample(range(neg_row.size(0)), n_v + n_t))
        perm = perm.to(torch.long)
        neg_row, neg_col = neg_row[perm], neg_col[perm]

        neg_adj_mask[neg_row, neg_col] = 0
        data.train_neg_adj_mask = neg_adj_mask

        row, col = neg_row[:n_v], neg_col[:n_v]
        data.val_neg_edge_index = torch.stack([row, col], dim=0)

        row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
        data.test_neg_edge_index = torch.stack([row, col], dim=0)

        return data

    def recon_loss(self, z, pos_edge_index):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """

        #print(type(pos_edge_index),pos_edge_index.shape)
        if 3*z.size(0) < pos_edge_index.size(1):
            pos_edge_index = pos_edge_index[:,np.random.choice(pos_edge_index.size(1),3*z.size(0),replace=False)]

        #print(type(pos_edge_index), pos_edge_index.shape)
        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
        #print(pos_loss.item(),neg_loss.item(),'neg and pos loss')
        return pos_loss + neg_loss
    
    def recon_loss2(self, z, pos_edge_index):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """

        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()



        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))




        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

          

        #print(pos_loss.item(),neg_loss.item(),'neg and pos loss')
        return pos_loss + neg_loss

    def class_test(self,x,target_nodes,test_nodes,labels):
        

        sources = target_nodes * len(test_nodes)
        targets = []
        for i in test_nodes:
            targets.extend([i]*len(target_nodes))

        edge_index = torch.tensor(np.array([sources,targets]))
        outputs = self.decoder(x, edge_index, sigmoid=True).view(len(test_nodes),-1)
        #print(outputs[0])
        if self.task == 'node_m':

            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0
        else:
            outputs = outputs.argmax(dim=1)
        test_value = f1_score(labels.cpu().detach(), outputs.cpu().detach(), average='micro')
        return test_value
    def class_loss(self,z, pos_edge_index,neg_edge_index):

        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(#[:,np.random.choice(neg_edge_index.size(1),pos_edge_index.size(1),replace=False)]
            1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return (pos_loss/3 + neg_loss)


    def class_loss2(self,z, pos_edge_index,neg_edge_index):
        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(#[:,np.random.choice(neg_edge_index.size(1),pos_edge_index.size(1),replace=False)]
            1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()
        tmp = z[2000:]
        reg = torch.norm(torch.mm(tmp,tmp.T)-torch.eye(tmp.shape[0]).to(z))
        return (pos_loss/3 + neg_loss) + 0.01*reg
        

    def class_loss3(self,z, pos_edge_index,neg_edge_index):

        output = z[:2000]@(z[2000:].t())
        return nn.CrossEntropyLoss()(output[pos_edge_index[0]],pos_edge_index[1]-2000)






    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)
    


class MyTask(MyGAE):
    def __init__(self,task,encoder,share_dims=0, output_dims=0,decoder = None):
        if task == 'link':
            if decoder is not None:super(MyTask,self).__init__(encoder,decoder=decoder)
            else: super(MyTask,self).__init__(encoder,decoder=None)
        else: 
            if decoder is not None:super(MyTask,self).__init__(encoder,decoder=decoder)
            else: super(MyTask,self).__init__(encoder=encoder,decoder=MLPDecoder(share_dims, output_dims))
            if task == 'node':self.loss = nn.CrossEntropyLoss()
            if task == 'node_m':self.loss = nn.BCEWithLogitsLoss()
        self.task = task


    def class_loss(self,x,target,flag = True):
        if flag:
            outputs = self.decoder(x)
            if self.task == 'node':
                tmp = torch.zeros(outputs.shape[0],outputs.shape[1]).to(x)
                for i in range(len(target)):
                    tmp[i,target[i]] = 1
                target = tmp

            return nn.BCEWithLogitsLoss()(outputs,target)
        else:
            outputs = self.decoder(x)
            return self.loss(outputs,target)



    def class_test(self,x,target):
        #x = F.normalize(x, p=2, dim=1)
        
        outputs = self.decoder(x)
        if self.task == 'node_m':

            outputs[outputs > 0] = 1
            outputs[outputs <= 0] = 0
        else:
            outputs = outputs.argmax(dim=1)

        test_value = f1_score(target.cpu().detach(),outputs.cpu().detach(), average='micro')
        return test_value










        





