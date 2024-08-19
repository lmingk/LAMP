
from torch_geometric.nn import GCNConv

from torch import dropout, nn
from .layers import *
import torch.nn.functional as F
from utils import uniform
from collections import OrderedDict


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











class GraphSignature(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(GraphSignature, self).__init__()
        self.args = args

        self.conv1 = MetaGCNConv(in_channels, 2*out_channels, cached=False)
        self.fc1 = nn.Linear(2*out_channels, 2*out_channels, bias=True)
        self.fc2 = nn.Linear(2*out_channels, 2*out_channels, bias=True)
        self.fc3 = nn.Linear(2*out_channels, out_channels, bias=True)
        self.fc4 = nn.Linear(2*out_channels, out_channels, bias=True)
        self.class_weight = nn.Linear(2*out_channels, 2*out_channels, bias=True)


    def forward(self, x, edge_index, weights):

        x = F.relu(self.conv1(x, edge_index, \
                    weights['encoder.signature.conv1.weight'],\
                    weights['encoder.signature.conv1.bias']))

        x = x.mean(0)
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











class LampSignatureEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(LampSignatureEncoder, self).__init__()
        self.args = args
        self.conv1 = MetaGCNConv(in_channels, 2 * out_channels,  cached=False)
        self.conv2 = MetaGCNConv(2 * out_channels, out_channels,  cached=False)

        self.signature = GraphSignature(args, in_channels, out_channels)
        self.gating_weight1 = Parameter(torch.Tensor(in_channels, 2 * out_channels))
        self.gating_weight2 = Parameter(torch.Tensor(2 * out_channels, out_channels))

        glorot(self.gating_weight1)
        glorot(self.gating_weight2)



    def forward(self, x, edge_index, weights):
        x = F.relu(self.conv1(x, edge_index, \
                weights['encoder.conv1.weight'],weights['encoder.conv1.bias']))
        return self.conv2(x, edge_index,\
                weights['encoder.conv2.weight'],weights['encoder.conv2.bias'])



    def modulate(self, x, edge_index, weights):

        dic = OrderedDict()


        keys = list(weights.keys())
        gamma1, beta1, gamma2, beta2 = self.signature(x, edge_index, weights)

        x1 = self.conv1(x, edge_index, weights['encoder.conv1.weight'],weights['encoder.conv1.bias'])
        x2 = self.conv2(F.relu(x1), edge_index, weights['encoder.conv2.weight'],weights['encoder.conv2.bias'])

        alpha = torch.sigmoid(torch.mul(x1.mean(0).T, self.gating_weight1))
        gamma = torch.mul(alpha, gamma1) + torch.mul(1 - alpha, torch.ones_like(gamma1))
        beta = torch.mul(alpha, beta1) + torch.mul(1 - alpha, torch.ones_like(beta1))

        weight1 = torch.mul(self.conv1.weight, gamma)
        bias1 = self.conv1.bias

        dic['encoder.conv1.weight'] = weight1
        dic['encoder.conv1.bias'] = bias1

        alpha = torch.sigmoid(torch.mul(x2.mean(0).T, self.gating_weight2))
        gamma = torch.mul(alpha, gamma2) + torch.mul(1 - alpha, torch.ones_like(gamma2))
        beta = torch.mul(alpha, beta2) + torch.mul(1 - alpha, torch.ones_like(beta2))

        weight2 = torch.mul(self.conv2.weight, gamma)
        bias2 = self.conv2.bias


        dic['encoder.conv2.weight'] = weight2
        dic['encoder.conv2.bias'] = bias2

        return dic


















        




    

        






class MetaEncoder(torch.nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(MetaEncoder, self).__init__()
        self.args = args
        self.conv1 = MetaGCNConv(in_channels, 2 * out_channels, cached=False)
        self.conv2 = MetaGCNConv(2 * out_channels, out_channels, cached=False)


    def forward(self, x, edge_index, weights, inner_loop=True):
        x = F.relu(self.conv1(x, edge_index, \
                weights['encoder.conv1.weight'],weights['encoder.conv1.bias']))
        return self.conv2(x, edge_index,\
                    weights['encoder.conv2.weight'],weights['encoder.conv2.bias'])








class WDiscriminator(torch.nn.Module):
    def __init__(self, hidden_size, hidden_size2=512):
        super(WDiscriminator, self).__init__()
        self.hidden = MyGCNConv(hidden_size, hidden_size2, cached=False)  # R=1 is implemented by a one-layer GCN
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








