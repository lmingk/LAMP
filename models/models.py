
import torch
import math
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, GCNConv, GAE, VGAE
from torch.nn import Parameter,BatchNorm1d,Dropout
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch.distributions import Normal
from torch import dropout, nn
from .layers import *
import torch.nn.functional as F
from utils import uniform
from collections import OrderedDict
from torch_geometric.nn import global_mean_pool
from models.autoencoder import negative_sampling
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from torch_geometric.data import Batch, Data

def softplus(x):
  return torch.nn.functional.softplus(x, beta=100)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

class Encoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.args = args
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
        if args.model in ['GAE']:
            self.conv2 = GCNConv(2 * out_channels, out_channels, cached=False)
        elif args.model in ['VGAE']:
            self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=False)
            self.conv_logvar = GCNConv(
                2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        if self.args.model in ['GAE']:
            return self.conv2(x, edge_index)
        elif self.args.model in ['VGAE']:
            return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)



class MetaMLPEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(MetaMLPEncoder, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(in_channels, 2 * out_channels, bias=True)
        if args.model in ['GAE']:
            self.fc_mu = nn.Linear(2 * out_channels, out_channels, bias=True)
        elif args.model in ['VGAE']:
            self.fc_mu = nn.Linear(2 * out_channels, out_channels, bias=True)
            self.fc_logvar = nn.Linear(2 * out_channels, out_channels, bias=True)

    def forward(self, x, edge_index, weights, inner_loop=True):
        x = F.relu(F.linear(x, weights['encoder.fc1.weight'],weights['encoder.fc1.bias']))
        if self.args.model in ['GAE']:
            return F.relu(F.linear(x, weights['encoder.fc_mu.weight'],weights['encoder.fc_mu.bias']))
        elif self.args.model in ['VGAE']:
            return F.relu(F.linear(x,weights['encoder.fc_mu.weight'],\
                    weights['encoder.fc_mu.bias'])),F.relu(F.linear(x,\
                    weights['encoder.fc_logvar.weight'],weights['encoder.fc_logvar.bias']))

class MLPEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(MLPEncoder, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(in_channels, 2 * out_channels, bias=True)
        self.fc2 = nn.Linear(2 * out_channels, out_channels, bias=True)

    def forward(self, x, edge_index):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x




class LampContext(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(GraphContext, self).__init__()
        self.args = args
        if self.args.use_gcn_sig:
            self.conv1 = MetaGCNConv(in_channels, 2*out_channels, cached=False)
            self.fc1 = nn.Linear(2*out_channels, 2*out_channels, bias=True)
            self.fc2 = nn.Linear(2*out_channels, 2*out_channels, bias=True)
            self.fc3 = nn.Linear(2*out_channels, out_channels, bias=True)
            self.fc4 = nn.Linear(2*out_channels, out_channels, bias=True)
        else:
            self.gated_conv1 = MetaGatedGraphConv(in_channels, args.num_gated_layers)
            self.fc1 = nn.Linear(in_channels, 2 * out_channels, bias=True)
            self.fc2 = nn.Linear(in_channels, 2 * out_channels, bias=True)
            self.fc3 = nn.Linear(in_channels, out_channels, bias=True)
            self.fc4 = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, edge_index, weights, keys):
        if self.args.use_gcn_sig:
            x = F.relu(self.conv1(x, edge_index, \
                    weights['encoder.signature.conv1.weight'],\
                    weights['encoder.signature.conv1.bias']))
        else:
            x = F.relu(self.gated_conv1(x, edge_index, weights,keys))

        x = x.sum(0)
        x_gamma_1 = F.linear(x, weights['encoder.signature.fc1.weight'],\
                weights['encoder.signature.fc1.bias'])
        x_beta_1 = F.linear(x, weights['encoder.signature.fc2.weight'],\
                weights['encoder.signature.fc2.bias'])
        x_gamma_2 = F.linear(x, weights['encoder.signature.fc3.weight'],\
                weights['encoder.signature.fc3.bias'])
        x_beta_2 = F.linear(x, weights['encoder.signature.fc4.weight'],\
                weights['encoder.signature.fc4.bias'])
        return torch.tanh(x_gamma_1), torch.tanh(x_beta_1),\
                torch.tanh(x_gamma_2), torch.tanh(x_beta_2)


class LampGraphContext5(torch.nn.Module):   # 这个signature 为每一个gcn层都计算了一个，从每一个层的角度来看要怎么modulation
    def __init__(self, args, in_channels, out_channels):
        super(LampGraphContext5, self).__init__()
        self.args = args
        if self.args.use_gcn_sig:
            self.conv1 = MetaGCNConv(in_channels, out_channels, cached=False)
            self.conv2 = MetaGCNConv(in_channels, out_channels, cached=False)
            self.conv3 = MetaGCNConv(in_channels, out_channels, cached=False)

            self.sigattn = context_attention(out_channels, 3)


            self.fc1 = nn.Linear(out_channels, out_channels, bias=True)
            self.fc2 = nn.Linear(out_channels, out_channels, bias=True)

    def forward(self, xs, edge_indexs, weights, keys):
        if self.args.use_gcn_sig:
            x1 = F.relu(self.conv1(xs[0], edge_indexs[0], \
                                   weights['encoder.signature.conv1.weight'], \
                                   weights['encoder.signature.conv1.bias']))
            x1 = x1.sum(0)
            x2 = F.relu(self.conv1(xs[1], edge_indexs[1], \
                                   weights['encoder.signature.conv2.weight'], \
                                   weights['encoder.signature.conv2.bias']))
            x2 = x2.sum(0)
            x3 = F.relu(self.conv1(xs[2], edge_indexs[2], \
                                   weights['encoder.signature.conv3.weight'], \
                                   weights['encoder.signature.conv3.bias']))
            x3 = x3.sum(0)

            x = self.sigattn([x1, x2, x3])

        x_gamma = F.linear(x, weights['encoder.signature.fc1.weight'], \
                             weights['encoder.signature.fc1.bias'])
        x_beta = F.linear(x, weights['encoder.signature.fc2.weight'], \
                            weights['encoder.signature.fc2.bias'])

        return torch.tanh(x_gamma), torch.tanh(x_beta)




class LampGraphContext4(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(LampGraphContext4, self).__init__()
        self.args = args
        if self.args.use_gcn_sig:
            self.conv1 = MetaGCNConv(in_channels, 2*out_channels, cached=False)
            self.conv2 = MetaGCNConv(in_channels, 2 * out_channels, cached=False)
            self.conv3 = MetaGCNConv(in_channels, 2 * out_channels, cached=False)

            self.sigattn = context_attention(2 * out_channels, 3)

            self.fc1 = nn.Linear(2*out_channels, 2*out_channels, bias=True)
            self.fc2 = nn.Linear(2*out_channels, 2*out_channels, bias=True)
            self.fc3 = nn.Linear(2*out_channels, out_channels, bias=True)
            self.fc4 = nn.Linear(2*out_channels, out_channels, bias=True)


    def forward(self, xs, edge_indexs, weights, key):


        if key == 0:
       
            x1 = F.relu(self.conv1(xs[0], edge_indexs[0], \
                    weights['encoder.signature.conv1.weight'],\
                    weights['encoder.signature.conv1.bias']))
                  
            x1 = x1.sum(0)
            x2 = F.relu(self.conv2(xs[1], edge_indexs[1], \
                    weights['encoder.signature.conv2.weight'], \
                    weights['encoder.signature.conv2.bias']))
            x2 = x2.sum(0)
            x3 = F.relu(self.conv3(xs[2], edge_indexs[2], \
                    weights['encoder.signature.conv3.weight'], \
                    weights['encoder.signature.conv3.bias']))
            x3 = x3.sum(0)


            x = self.sigattn([x1,x2,x3])
        elif key==1:
            x1 = F.relu(self.conv1(xs, edge_indexs, \
                                   weights['encoder.signature.conv1.weight'], \
                                   weights['encoder.signature.conv1.bias']))
            x = x1.sum(0)

        elif key==2:
            x2 = F.relu(self.conv2(xs, edge_indexs, \
                                   weights['encoder.signature.conv1.weight'], \
                                   weights['encoder.signature.conv1.bias']))
            x = x2.sum(0)

        else:
            x3 = F.relu(self.conv3(xs, edge_indexs, \
                                   weights['encoder.signature.conv1.weight'], \
                                   weights['encoder.signature.conv1.bias']))
            x = x3.sum(0)


        x_gamma_1 = F.linear(x, weights['encoder.signature.fc1.weight'],\
                weights['encoder.signature.fc1.bias'])
        x_beta_1 = F.linear(x, weights['encoder.signature.fc2.weight'],\
                weights['encoder.signature.fc2.bias'])
        x_gamma_2 = F.linear(x, weights['encoder.signature.fc3.weight'],\
                weights['encoder.signature.fc3.bias'])
        x_beta_2 = F.linear(x, weights['encoder.signature.fc4.weight'],\
                weights['encoder.signature.fc4.bias'])
        return torch.tanh(x_gamma_1), torch.tanh(x_beta_1),\
                torch.tanh(x_gamma_2), torch.tanh(x_beta_2)

class LampGraphContext2(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(LampGraphContext2, self).__init__()
        self.args = args
        if self.args.use_gcn_sig:
            self.conv1 = MetaGCNConv(in_channels, 2*out_channels, cached=False)
            self.conv2 = MetaGCNConv(in_channels, 2 * out_channels, cached=False)
            self.conv3 = MetaGCNConv(in_channels, 2 * out_channels, cached=False)

            self.sigattn = context_attention(2 * out_channels, 3)

            self.fc1 = nn.Linear(2*out_channels, 2*out_channels, bias=True)
            self.fc2 = nn.Linear(2*out_channels, 2*out_channels, bias=True)
            self.fc3 = nn.Linear(2*out_channels, out_channels, bias=True)
            self.fc4 = nn.Linear(2*out_channels, out_channels, bias=True)


    def forward(self, x, edge_index, weights,keys):

        if self.args.use_gcn_sig:
            x1 = F.relu(self.conv1(x, edge_index, \
                    weights['encoder.signature.conv1.weight'],\
                    weights['encoder.signature.conv1.bias']))
            x1 = x1.mean(0)
            x2 = F.relu(self.conv2(x, edge_index, \
                    weights['encoder.signature.conv2.weight'], \
                    weights['encoder.signature.conv2.bias']))
            x2 = x2.mean(0)
            x3 = F.relu(self.conv3(x, edge_index, \
                    weights['encoder.signature.conv3.weight'], \
                    weights['encoder.signature.conv3.bias']))
            x3 = x3.mean(0)


            x = self.sigattn([x1,x2,x3])


        x_gamma_1 = F.linear(x, weights['encoder.signature.fc1.weight'],\
                weights['encoder.signature.fc1.bias'])
        x_beta_1 = F.linear(x, weights['encoder.signature.fc2.weight'],\
                weights['encoder.signature.fc2.bias'])
        x_gamma_2 = F.linear(x, weights['encoder.signature.fc3.weight'],\
                weights['encoder.signature.fc3.bias'])
        x_beta_2 = F.linear(x, weights['encoder.signature.fc4.weight'],\
                weights['encoder.signature.fc4.bias'])
        return x_gamma_1, x_beta_1,x_gamma_2, x_beta_2

class LampGraphContext3(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(LampGraphContext3, self).__init__()
        self.args = args
        if self.args.use_gcn_sig:
            self.conv1 = MetaGCNConv(in_channels, 2*out_channels, cached=False)
            self.conv2 = MetaGCNConv(in_channels, 2 * out_channels, cached=False)
            self.conv3 = MetaGCNConv(in_channels, 2 * out_channels, cached=False)

            self.sigattn = context_attention(2 * out_channels, 3)

            self.fc1 = nn.Linear(2*out_channels, 2*out_channels, bias=True)
            self.fc2 = nn.Linear(2*out_channels, 2*out_channels, bias=True)
            self.fc3 = nn.Linear(2*out_channels, out_channels, bias=True)
            self.fc4 = nn.Linear(2*out_channels, out_channels, bias=True)

    def reset_parameters(self):

        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()

        self.sigattn.reset_parameters()


    def forward(self, x, edge_index, weights,keys):

        if self.args.use_gcn_sig:
            x1 = F.relu(self.conv1(x, edge_index, \
                    weights['encoder.signature.conv1.weight'],\
                    weights['encoder.signature.conv1.bias']))
            x1 = x1.mean(0)
            x2 = F.relu(self.conv2(x, edge_index, \
                    weights['encoder.signature.conv2.weight'], \
                    weights['encoder.signature.conv2.bias']))
            x2 = x2.mean(0)
            x3 = F.relu(self.conv3(x, edge_index, \
                    weights['encoder.signature.conv3.weight'], \
                    weights['encoder.signature.conv3.bias']))
            x3 = x3.mean(0)


            x = self.sigattn([x1,x2,x3])


        x_gamma_1 = F.linear(x, weights['encoder.signature.fc1.weight'],\
                weights['encoder.signature.fc1.bias'])
        x_beta_1 = F.linear(x, weights['encoder.signature.fc2.weight'],\
                weights['encoder.signature.fc2.bias'])
        x_gamma_2 = F.linear(x, weights['encoder.signature.fc3.weight'],\
                weights['encoder.signature.fc3.bias'])
        x_beta_2 = F.linear(x, weights['encoder.signature.fc4.weight'],\
                weights['encoder.signature.fc4.bias'])
        return torch.tanh(x_gamma_1), torch.tanh(x_beta_1),\
                torch.tanh(x_gamma_2), torch.tanh(x_beta_2)









class context_attention(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(context_attention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fc1 = nn.Linear(in_channels*out_channels,in_channels)
        self.fc2 = nn.Linear(in_channels,in_channels)
        self.fc3 = nn.Linear(in_channels,out_channels)


    def forward(self,Xs):

        x_in = torch.cat(Xs)
        x_out = self.fc2(F.relu(self.fc1(x_in)))
        x_out = self.fc3(F.relu(x_out))

        x_out = torch.softmax(x_out,dim=-1)
        self.class_out = x_out


        return torch.matmul(x_out , x_in.reshape(self.out_channels,self.in_channels))


    def reset_parameters(self):
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)





















class GraphContext(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(GraphContext, self).__init__()
        self.args = args
        if self.args.use_gcn_sig:
            self.conv1 = MetaGCNConv(in_channels, 2*out_channels, cached=False)
            self.fc1 = nn.Linear(2*out_channels, 2*out_channels, bias=True)
            self.fc2 = nn.Linear(2*out_channels, 2*out_channels, bias=True)
            self.fc3 = nn.Linear(2*out_channels, out_channels, bias=True)
            self.fc4 = nn.Linear(2*out_channels, out_channels, bias=True)
            self.class_weight = nn.Linear(2*out_channels, 2*out_channels, bias=True)
        else:
            self.gated_conv1 = MetaGatedGraphConv(in_channels, args.num_gated_layers)
            self.fc1 = nn.Linear(in_channels, 2 * out_channels, bias=True)
            self.fc2 = nn.Linear(in_channels, 2 * out_channels, bias=True)
            self.fc3 = nn.Linear(in_channels, out_channels, bias=True)
            self.fc4 = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, edge_index, weights, keys, classes = None):
        if self.args.use_gcn_sig:
            x = F.relu(self.conv1(x, edge_index, \
                    weights['encoder.signature.conv1.weight'],\
                    weights['encoder.signature.conv1.bias']))
        else:
            x = F.relu(self.gated_conv1(x, edge_index, weights,keys))
        if classes is not None:
            class_feature = F.linear(x[classes].mean(0), weights['encoder.signature.class_weight.weight'],\
                weights['encoder.signature.class_weight.bias'])
            x = x.mean(0) + class_feature
        else: x = x.mean(0)
        x_gamma_1 = F.linear(x, weights['encoder.signature.fc1.weight'],\
                weights['encoder.signature.fc1.bias'])
        x_beta_1 = F.linear(x, weights['encoder.signature.fc2.weight'],\
                weights['encoder.signature.fc2.bias'])
        x_gamma_2 = F.linear(x, weights['encoder.signature.fc3.weight'],\
                weights['encoder.signature.fc3.bias'])
        x_beta_2 = F.linear(x, weights['encoder.signature.fc4.weight'],\
                weights['encoder.signature.fc4.bias'])
        return torch.tanh(x_gamma_1), torch.tanh(x_beta_1),\
                torch.tanh(x_gamma_2), torch.tanh(x_beta_2)










class LampContextEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(LampContextEncoder, self).__init__()
        self.args = args
        self.conv1 = MetaGCNConv(in_channels, 2 * out_channels,  cached=False)
        self.conv2 = MetaGCNConv(2 * out_channels, out_channels,  cached=False)

        self.signature = GraphContext(args, in_channels, out_channels)
        self.gating_weight1 = Parameter(torch.Tensor(in_channels, 2 * out_channels))
        self.gating_weight2 = Parameter(torch.Tensor(2 * out_channels, out_channels))

        glorot(self.gating_weight1)
        glorot(self.gating_weight2)




    def forward(self, x, edge_index, weights):
        x = F.relu(self.conv1(x, edge_index, \
                weights['encoder.conv1.weight'],weights['encoder.conv1.bias']))
        return self.conv2(x, edge_index,\
                weights['encoder.conv2.weight'],weights['encoder.conv2.bias'])




    def reparameterization(self, x, edge_index, weights, classes = None):

        dic = OrderedDict()

        keys = list(weights.keys())
        sig_keys = [key for key in keys if 'signature' in key]
        gamma1, beta1, gamma2, beta2 = self.signature(x, edge_index, weights, sig_keys,classes)

        x1 = self.conv1(x, edge_index, weights['encoder.conv1.weight'],weights['encoder.conv1.bias'])
        x2 = self.conv2(F.relu(x1), edge_index, weights['encoder.conv2.weight'],weights['encoder.conv2.bias'])

        alpha = torch.sigmoid(torch.mul(x1.mean(0).T, self.gating_weight1))
        gamma = torch.mul(alpha, gamma1) + torch.mul(1 - alpha, torch.ones_like(gamma1))
        beta = torch.mul(alpha, beta1) + torch.mul(1 - alpha, torch.ones_like(beta1))

        weight1 = torch.mul(self.conv1.weight, gamma)
        bias1 = torch.mul(self.conv1.bias, beta)

        dic['encoder.conv1.weight'] = weight1
        dic['encoder.conv1.bias'] = bias1

        alpha = torch.sigmoid(torch.mul(x2.mean(0).T, self.gating_weight2))
        gamma = torch.mul(alpha, gamma2) + torch.mul(1 - alpha, torch.ones_like(gamma2))
        beta = torch.mul(alpha, beta2) + torch.mul(1 - alpha, torch.ones_like(beta2))

        weight2 = torch.mul(self.conv2.weight, gamma)
        bias2 = torch.mul(self.conv2.bias, beta)


        dic['encoder.conv2.weight'] = weight2
        dic['encoder.conv2.bias'] = bias2

        return dic



'''



class LampContextEncoder(torch.nn.Module):  
    def __init__(self, args, in_channels, out_channels):
        super(LampContextEncoder, self).__init__()
        self.args = args
        self.conv1 = MetaGCNConv(in_channels, 2 * out_channels,  cached=False)
        self.conv2 = MetaGCNConv(2 * out_channels, out_channels,  cached=False)

        self.signature = GraphSignature(args, in_channels, out_channels)
        self.gating_weight1 = Parameter(torch.Tensor( 2 * out_channels, 2 * out_channels))
        self.gating_weight2 = Parameter(torch.Tensor(out_channels, out_channels))

        glorot(self.gating_weight1)
        glorot(self.gating_weight2)




    def forward(self, x, edge_index, weights):
        x = F.relu(self.conv1(x, edge_index, \
                weights['encoder.conv1.weight'],weights['encoder.conv1.bias']))
        return self.conv2(x, edge_index,\
                weights['encoder.conv2.weight'],weights['encoder.conv2.bias'])




    def reparameterization(self, x, edge_index, weights,classes = None):

        dic = OrderedDict()


        keys = list(weights.keys())
        sig_keys = [key for key in keys if 'signature' in key]
        gamma1, beta1, gamma2, beta2 = self.signature(x, edge_index, weights, sig_keys,classes)

        x1 = self.conv1(x, edge_index, weights['encoder.conv1.weight'],weights['encoder.conv1.bias'])
        x2 = self.conv2(F.relu(x1), edge_index, weights['encoder.conv2.weight'],weights['encoder.conv2.bias'])

        alpha = torch.sigmoid(torch.matmul(x1, self.gating_weight1).mean(0))
        gamma = torch.mul(alpha, gamma1) + torch.mul(1 - alpha, torch.ones_like(gamma1))
        beta = torch.mul(alpha, beta1) + torch.mul(1 - alpha, torch.ones_like(beta1))

        weight1 = torch.mul(self.conv1.weight, gamma)
        bias1 = torch.mul(self.conv1.bias, beta)

        dic['encoder.conv1.weight'] = weight1
        dic['encoder.conv1.bias'] = bias1

        alpha = torch.sigmoid(torch.matmul(x2, self.gating_weight2).mean(0))
        gamma = torch.mul(alpha, gamma2) + torch.mul(1 - alpha, torch.ones_like(gamma2))
        beta = torch.mul(alpha, beta2) + torch.mul(1 - alpha, torch.ones_like(beta2))

        weight2 = torch.mul(self.conv2.weight, gamma)
        bias2 = torch.mul(self.conv2.bias, beta)


        dic['encoder.conv2.weight'] = weight2
        dic['encoder.conv2.bias'] = bias2

        return dic
 '''    












class MetaEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(MetaEncoder, self).__init__()
        self.args = args
        self.conv1 = MetaGCNConv(in_channels, 2 * out_channels, cached=False)
        if args.model in ['GAE']:
            self.conv2 = MetaGCNConv(2 * out_channels, out_channels, cached=False)
        elif args.model in ['VGAE']:
            self.conv_mu = MetaGCNConv(2 * out_channels, out_channels, cached=False)
            self.conv_logvar = MetaGCNConv(
                2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index, weights, inner_loop=True):
        x = F.relu(self.conv1(x, edge_index, \
                weights['encoder.conv1.weight'],weights['encoder.conv1.bias']))
        if self.args.model in ['GAE']:
            return self.conv2(x, edge_index,\
                    weights['encoder.conv2.weight'],weights['encoder.conv2.bias'])
        elif self.args.model in ['VGAE']:
            return self.conv_mu(x,edge_index,weights['encoder.conv_mu.weight'],\
                    weights['encoder.conv_mu.bias']),\
                self.conv_logvar(x,edge_index,weights['encoder.conv_logvar.weight'],\
                weights['encoder.conv_logvar.bias'])





class Net(torch.nn.Module):
    def __init__(self,train_dataset):
        super(Net, self).__init__()
        self.conv1 = GATConv(train_dataset.num_features, 256, heads=4)
        self.lin1 = torch.nn.Linear(train_dataset.num_features, 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(
            4 * 256, train_dataset.num_classes, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, train_dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index) + self.lin1(x))
        x = F.elu(self.conv2(x, edge_index) + self.lin2(x))
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x




class WDiscriminator(torch.nn.Module):
    def __init__(self, hidden_size, hidden_size2=512):
        super(WDiscriminator, self).__init__()
        self.hidden = MyGCNConv(hidden_size, hidden_size2, cached=False)
        self.hidden2 = torch.nn.Linear(hidden_size2, hidden_size2)
        self.output = torch.nn.Linear(hidden_size2, 1)

    def forward(self, input_embd,edge_index):
        return self.output(F.leaky_relu(self.hidden2(F.leaky_relu(self.hidden(input_embd,edge_index), 0.2, inplace=True)), 0.2, inplace=True))

    def reset_parameters(self):
        #print('reset')
        glorot(self.hidden.weight)
        glorot(self.hidden2.weight)
        glorot(self.output.weight)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_out)
        self.Dropout = nn.Dropout(p=0.0)
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)


    def forward(self, x):

        return (self.fc2(F.relu(self.Dropout(self.fc1(x)))))

    def reset_parameters(self):
        print('reset')
        glorot(self.fc1.weight)
        glorot(self.fc2.weight)


class AE(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(AE, self).__init__()

        self.input = MLP(dim_in, dim_hidden,dim_out)
        self.output = MLP(dim_out,dim_hidden, dim_in)
        self.loss = torch.nn.MSELoss(size_average=True)



    def forward(self, x):

        return self.input(x)
    
    def con_loss(self,x,con_x):
        return self.loss(x,con_x) 

    def reset_parameters(self):
        print('reset')
        self.input.reset_parameters()
        self.output.reset_parameters()


        

class WDiscriminator_old(torch.nn.Module):
    def __init__(self, hidden_size, hidden_size2=512):
        super(WDiscriminator_old, self).__init__()
        self.hidden = torch.nn.Linear(hidden_size, hidden_size2)
        self.hidden2 = torch.nn.Linear(hidden_size2, hidden_size2)
        self.output = torch.nn.Linear(hidden_size2, 1)

    def forward(self, input_embd):
        return self.output(F.leaky_relu(self.hidden2(F.leaky_relu(self.hidden(input_embd), 0.2, inplace=True)), 0.2, inplace=True))

    def reset_parameters(self):
        print('reset')
        glorot(self.hidden.weight)
        glorot(self.hidden2.weight)
        glorot(self.output.weight)
