
import torch
import math
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid,PPI
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, GCNConv, GAE, VGAE,GINConv
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch.distributions import Normal
from torch import dropout, nn
import torch.nn.functional as F
from utils import uniform
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch import Tensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from typing import Callable, Optional, Union

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class MetaGCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper
    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},
    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 **kwargs):
        super(MetaGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1.0 if not improved else 2.0



        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, weight, bias, edge_weight=None, gamma=None, beta=None):
        """"""
        x = torch.matmul(x, weight)+bias

        ''' FiLM part '''
        if gamma is not None and beta is not None:
            x = torch.mul(x, gamma) + beta

        if not self.cached or self.cached_result is None:
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)






class MyGCNConv(MessagePassing):


    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 **kwargs):
        super(MyGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1.0 if not improved else 2.0

        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        mask = row == col

        norm = deg_inv_sqrt[row] * edge_weight
        norm[mask] = norm[mask]+1


        return edge_index, norm

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        #x = x @ self.weight

        if not self.cached or self.cached_result is None:
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm, bias=self.bias)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out, bias):
        aggr_out = aggr_out @ self.weight
        if self.bias is not None:
            aggr_out = aggr_out + bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)













class LampGCNConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 gating=None,
                 **kwargs):
        super(LampGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.gating = gating



        self.weight_1 = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_2 = Parameter(torch.Tensor(1, 1))
        self.gating_weights = Parameter(torch.Tensor(in_channels, out_channels))


        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        #self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight_1)
        zeros(self.bias)
        glorot(self.gating_weights)


        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1.0 if not improved else 2.0
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, weight_1, weight_2, bias, gating_weights, edge_weight=None, gamma=None, beta=None,parameters = False):
        """"""
        ''' FiLM part '''

        if gamma is not None and beta is not None:
            

            alpha = torch.sigmoid(torch.matmul(x, gating_weights).mean(0))
            gamma = torch.mul(alpha, gamma) + torch.mul(1 - alpha, torch.ones_like(gamma))
            beta = torch.mul(alpha, beta) + torch.mul(1 - alpha, torch.ones_like(beta))


        weight_1 = torch.mul(weight_1, gamma)
        bias = torch.mul(bias, beta)

        if parameters:
            self.weight_new = weight_1
            self.bias_new = bias


        x =  torch.matmul(x, weight_1)+bias
        
        if not self.cached or self.cached_result is None:
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result

        return  self.propagate(edge_index, x=x, norm=norm)



        

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out 
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)










class MetaGatedGCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper
    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},
    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 gating=None,
                 **kwargs):
        super(MetaGatedGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.gating = gating
        if gating == 'signature':
            self.weight_1 = Parameter(torch.Tensor(in_channels, out_channels))
            self.weight_2 = Parameter(torch.Tensor(1, 1))
            self.gating_weights = Parameter(torch.Tensor(out_channels))

        elif gating == 'weights':
            self.weight_1 = Parameter(torch.Tensor(in_channels, out_channels))
            self.weight_2 = Parameter(torch.Tensor(in_channels, out_channels))
            self.gating_weights = Parameter(torch.Tensor(out_channels))

        elif gating == 'signature_cond':
            self.weight_1 = Parameter(torch.Tensor(in_channels, out_channels))
            self.weight_2 = Parameter(torch.Tensor(1, 1))
            self.gating_weights = Parameter(torch.Tensor(in_channels, out_channels))

        elif gating == 'weights_cond':
            self.weight_1 = Parameter(torch.Tensor(in_channels, out_channels))
            self.weight_2 = Parameter(torch.Tensor(in_channels, out_channels))
            self.gating_weights = Parameter(torch.Tensor(in_channels, out_channels))

        else:
            self.weight_1 = Parameter(torch.Tensor(in_channels, out_channels))
            self.weight_2 = Parameter(torch.Tensor(1, 1))
            self.gating_weights = Parameter(torch.Tensor(1, 1))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight_1)
        glorot(self.weight_2)
        zeros(self.bias)

        if self.gating.endswith('cond'):
            glorot(self.gating_weights)
        else:
            zeros(self.gating_weights)

        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1.0 if not improved else 2.0
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, weight_1, weight_2, bias, gating_weights, edge_weight=None, gamma=None, beta=None):
        """"""
        ''' FiLM part '''
        if gamma is not None and beta is not None:
            if self.gating == 'signature':
                alpha = torch.sigmoid(gating_weights)
                gamma = torch.mul(alpha, gamma) + torch.mul(1 - alpha, torch.ones_like(gamma))
                beta = torch.mul(alpha, beta) + torch.mul(1 - alpha, torch.ones_like(beta))

                weight_1 = torch.mul(weight_1, gamma) + beta
                x = torch.matmul(x, weight_1)
                #x = torch.mul(x, gamma) + beta

            elif self.gating == 'weights':
                alpha = torch.sigmoid(gating_weights)

                x_1 = torch.matmul(x, weight_1)
                x_2 = torch.matmul(x, weight_2)
                x_2 = torch.mul(x_2, gamma) + beta

                x = torch.mul(alpha, x_1) + torch.mul(1 - alpha, x_2)

            elif self.gating == 'signature_cond':
                alpha = torch.sigmoid(torch.matmul(x, gating_weights))
                gamma = torch.mul(alpha, gamma) + torch.mul(1 - alpha, torch.ones_like(gamma))
                beta = torch.mul(alpha, beta) + torch.mul(1 - alpha, torch.ones_like(beta))

                #print(gamma.shape,beta.shape)

                #weight_1 = torch.mul(weight_1, gamma) + beta
                x = torch.matmul(x, weight_1)
                x = torch.mul(x, gamma) + beta

                #print(gamma,beta)

            elif self.gating == 'weights_cond':
                alpha = torch.sigmoid(torch.matmul(x, gating_weights))

                x_1 = torch.matmul(x, weight_1)
                x_2 = torch.matmul(x, weight_2)
                x_2 = torch.mul(x_2, gamma) + beta

                x = torch.mul(alpha, x_1) + torch.mul(1 - alpha, x_2)

            else:
                x = torch.matmul(x, weight_1)
                x = torch.mul(x, gamma) + beta

        if not self.cached or self.cached_result is None:
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm, bias=bias)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out, bias):
        if self.bias is not None:
            aggr_out = aggr_out + bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class MetaGRUCell(nn.Module):

    """
    An implementation of GRUCell with Functional Ops.
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(MetaGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden, weights, keys):
        x = x.view(-1, x.size(1))
        gate_x = F.linear(x,weights[keys[1]],weights[keys[2]])
        gate_h = F.linear(x,weights[keys[3]],weights[keys[4]])
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        hy = newgate + inputgate * (hidden - newgate)
        return hy

class MetaGatedGraphConv(MessagePassing):
    r"""The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper
    .. math::
        \mathbf{h}_i^{(0)} &= \mathbf{x}_i \, \Vert \, \mathbf{0}
        \mathbf{m}_i^{(l+1)} &= \sum_{j \in \mathcal{N}(i)} \mathbf{\Theta}
        \cdot \mathbf{h}_j^{(l)}
        \mathbf{h}_i^{(l+1)} &= \textrm{GRU} (\mathbf{m}_i^{(l+1)},
        \mathbf{h}_i^{(l)})
    up to representation :math:`\mathbf{h}_i^{(L)}`.
    The number of input channels of :math:`\mathbf{x}_i` needs to be less or
    equal than :obj:`out_channels`.
    Args:
        out_channels (int): Size of each input sample.
        num_layers (int): The sequence length :math:`L`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 out_channels,
                 num_layers,
                 aggr='add',
                 bias=True,
                 **kwargs):
        super(MetaGatedGraphConv, self).__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.weight = Parameter(torch.Tensor(num_layers, out_channels, out_channels))
        self.rnn = MetaGRUCell(out_channels, out_channels, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.out_channels, self.weight)
        self.rnn.reset_parameters()

    def forward(self, x, edge_index, weights, keys, edge_weight=None):
        """"""
        h = x if x.dim() == 2 else x.unsqueeze(-1)
        if h.size(1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if h.size(1) < self.out_channels:
            zero = h.new_zeros(h.size(0), self.out_channels - h.size(1))
            h = torch.cat([h, zero], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(h, weights[keys[0]][i])
            m = self.propagate(edge_index, x=m, edge_weight=edge_weight)
            h = self.rnn(m, h, weights, keys)

        return h

    def message(self, x_j, edge_weight):
        if edge_weight is not None:
            return edge_weight.view(-1, 1) * x_j
        return x_j

    def __repr__(self):
        return '{}({}, num_layers={})'.format(
            self.__class__.__name__, self.out_channels, self.num_layers)

