
from collections import defaultdict
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import os.path
from pathlib import Path
from random import randint
import random
import math
from torch_geometric.utils import to_undirected
from models.autoencoder import negative_sampling_identify
from models.models import MLP

def load_test_graph(dataset,index):
    dataset_str = './lamp_data/test/' + dataset + '/'
    file = dataset_str+'test_graph{}.npz'.format(index)
    file = np.load(file)
    feats = file['feats']

    scaler = StandardScaler()
    scaler.fit(feats)
    feats = scaler.transform(feats)

    edge_index = file['edge_index']
    labels = file['labels']
    graph = Graph(x = feats,edge_index=edge_index,is_tensor=False,labels=labels)
    return graph



def load_test_graph_share(dataset,index,num):
    dataset_str = './lamp_data/test/' + dataset + '/'
    file = dataset_str+'test_graph{}.npz'.format(index)
    file = np.load(file)
    feats = file['feats']

    mlp = MLP(feats.shape[1], 128, 128)
    mlp.load_state_dict(torch.load('./result/autoencoder_for_{}_{}.pth'.format(dataset,num)), strict=True)


    feats = mlp(torch.from_numpy(feats).to(torch.float)).cpu().detach().numpy()

    scaler = StandardScaler()
    scaler.fit(feats)
    feats = scaler.transform(feats)

    edge_index = file['edge_index']
    labels = file['labels']
    graph = Graph(x = feats,edge_index=edge_index,is_tensor=False,labels=labels)
    return graph


class Graph:
    def __init__(self,x,edge_index,is_tensor = False,labels = None):
        super(Graph, self).__init__()
        self.x = x
        self.edge_index = edge_index
        self.is_tensor = is_tensor
        self.labels = labels


    def to_tensor(self,device):

        if self.is_tensor:return


        edge_index = torch.from_numpy(self.edge_index).to(device)
        x = torch.from_numpy(self.x).to(torch.float).to(device)

        self.x = x
        self.edge_index = edge_index
        self.is_tensor = True


        if not self.labels is None:
            if len(self.labels.shape) == 2:self.labels = torch.FloatTensor(self.labels).to(device)
            else: self.labels = torch.LongTensor(self.labels).to(device)



    def detach(self):
        return Graph(self.x.detach(),self.edge_index,True)


    def transformation(self,MLP):
        return Graph(MLP(self.x).detach(),self.edge_index,True)


    def node_split(self,label_num,instance_num):
        '''
        permutation = np.random.permutation(self.labels.shape[1])
        self.indices = np.random.choice(self.labels.shape[0],label_num,replace = False)
        self.indices.sort()
        labels = self.labels[:,permutation]
        self.train_nodes = np.unique(labels[self.indices,:instance_num].flatten())
        self.test_nodes = np.setdiff1d(np.unique(labels[self.indices].flatten()),self.train_nodes)
        '''
        self.indices = np.random.choice(list(self.labels.keys()),label_num,replace = False)
        self.indices.sort()
        self.train_nodes =[]
        for i in self.indices:self.train_nodes.extend(np.random.choice(np.setdiff1d(self.labels[i],self.indices),instance_num,replace=False))
        self.train_nodes = np.unique(self.train_nodes)
        self.test_nodes = []
        for i in self.indices:self.test_nodes.extend(self.labels[i])


        self.test_nodes = np.setdiff1d(np.unique(self.test_nodes),self.train_nodes)
        self.test_nodes = np.setdiff1d(self.test_nodes,self.indices)
        #if len(self.test_nodes)>5*len(self.train_nodes):self.test_nodes = np.random.choice(self.test_nodes,3*len(self.train_nodes),replace=False)

        #print(len(self.train_nodes),len(self.test_nodes))


        train_pos_link = []
        train_neg_link = []
        test_pos_link = []
        test_neg_link = []
        train_nodes = self.train_nodes
        test_nodes = self.test_nodes
        for i in train_nodes:
            for j in self.indices:
                if i in self.labels[j]:train_pos_link.append((i,j))
                else:train_neg_link.append((i,j))


        ground_truth = []
        for i in test_nodes:
            ground_truth.append([])
            for j in self.indices:
                if i in self.labels[j]:
                    test_pos_link.append((i,j))
                    ground_truth[-1].append(1)
                else:
                    test_neg_link.append((i,j))
                    ground_truth[-1].append(0)

        self.ground_truth = np.array(ground_truth)


        self.train_pos_edge_index = torch.tensor(np.array(train_pos_link).T).to(self.edge_index)
        self.train_neg_edge_index = torch.tensor(np.array(train_neg_link).T).to(self.edge_index)
        self.test_pos_edge_index = torch.tensor(np.array(test_pos_link).T).to(self.edge_index)
        self.test_neg_edge_index = torch.tensor(np.array(test_neg_link).T).to(self.edge_index)

        #print(self.train_pos_edge_index.size(),self.train_neg_edge_index.size(),self.test_pos_edge_index.size(),self.test_neg_edge_index.size())
        #print(len(self.train_nodes),len(self.test_nodes),self.ground_truth.sum(1))






    def attr_split(self,ratio):
        mask_node = np.random.choice(self.x.shape[0],int(self.x.shape[0]*ratio),replace=False)
        self.attr = self.x[mask_node,:].clone().detach()
        self.x[mask_node] = 0
        self.mask = mask_node








    def link_split(self,train_ratio,test_ratio):

        row, col = self.edge_index


    # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]

        n_v = int(math.floor(train_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]

        r, c = row[:n_v], col[:n_v]
        train_pos_edge_index = torch.stack([r, c], dim=0)
        train_pos_edge_index = to_undirected(train_pos_edge_index)
        r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
        test_pos_edge_index = torch.stack([r, c], dim=0)

        r, c = row[n_v + n_t:], col[n_v + n_t:]
        val_pos_edge_index = torch.stack([r, c], dim=0)


        train_pos_edge_index = train_pos_edge_index.detach()
        val_pos_edge_index = val_pos_edge_index.detach()
        test_pos_edge_index = test_pos_edge_index.detach()

        num_nodes = self.x.shape[0]
        neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
        neg_adj_mask = neg_adj_mask.triu(diagonal=1)
        neg_adj_mask[row, col] = 0

        neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
        perm = torch.tensor(random.sample(range(neg_row.size(0)), n_t))
        perm = perm.to(torch.long)
        neg_row, neg_col = neg_row[perm], neg_col[perm]
        test_neg_edge_index = torch.stack([neg_row, neg_col], dim=0)

        self.train_edge_index = train_pos_edge_index
        self.test_edge_index = test_pos_edge_index








    def split_nodes(self,train_ratio,test_ratio):

        #### node split
        n_v = int(math.floor(train_ratio * self.x.size(0)))
        n_t = int(math.floor(test_ratio * self.x.size(0)))
        node_perm = torch.randperm(self.x.size(0))
        self.train_nodes = node_perm[:n_v]
        self.test_nodes = node_perm[n_v:n_v + n_t]


    
    def identify(self,task,train_ratio):
        if task == 'link':
            if self.is_tensor:
                edge_index = self.edge_index.detach()

            row = edge_index[0]
            col = edge_index[1]

            mask = row < col
            row, col = row[mask], col[mask]

            tmp = int(1/train_ratio)
            train_set = []
            tmp_set = []
            val_set = []
            test_set = []

            for i in range(row.size(0)):
                if i%tmp == 0:train_set.append(i)
                else:tmp_set.append(i)

            tmp = int((len(tmp_set)+0.0)/int(0.1*row.size(0)))
            for i,ele in enumerate(tmp_set):
                if i%tmp == 0:val_set.append(ele)
                else:test_set.append(ele)
            

            r, c = row[train_set], col[train_set]
            train_pos_edge_index = to_undirected(torch.stack([r, c], dim=0))
            r, c = row[val_set], col[val_set]
            val_pos_edge_index = to_undirected(torch.stack([r, c], dim=0))
            r, c = row[test_set], col[test_set]
            test_pos_edge_index = torch.stack([r, c], dim=0) #to_undirected(torch.stack([r, c], dim=0))

            train_pos_edge_index = train_pos_edge_index.detach()
            test_pos_edge_index = test_pos_edge_index.detach()
            val_pos_edge_index = val_pos_edge_index.detach()

            self.train_edge_index = train_pos_edge_index
            self.test_edge_index = test_pos_edge_index
            self.val_edge_index = val_pos_edge_index
            self.test_edge_index_negative = negative_sampling_identify(self.test_edge_index,self.x.size(0))

        else:

            train_num = int(self.labels.size(0)*train_ratio)
            val_num = int(self.labels.size(0)*0.1)

            if len(self.labels.shape) == 2:
                num_classes = int(self.labels.shape[1])

                label_arr = defaultdict(list)
                for index in range(num_classes):
                    label_arr[index] = np.where(self.labels[:,index].detach().cpu().numpy())[0].tolist()
                labels = []
                while len(label_arr)!=0:
                    for key in sorted(label_arr.keys()):
                        if len(label_arr[key]) == 0:
                            label_arr.pop(key)
                            continue
                        labels.append(label_arr[key].pop())
                    

                train_set = set()
                while len(train_set)!=train_num:train_set.add(labels.pop(0))
                val_set = set()
                while len(val_set)!=val_num:
                    value = labels.pop(0)
                    if value not in train_set:val_set.add(value)
                test_set = []
                for i in range(self.labels.size(0)):
                    if i not in train_set and i not in val_set:test_set.append(i)

                train_set = list(train_set)
                val_set = list(val_set)



            else:
                num_classes = int(self.labels.max().item()+1)

                label_arr = defaultdict(list)
                for index,label in enumerate(self.labels):
                    label_arr[label.item()].append(index)

                #for i in range(num_classes):print(len(label_arr[i]))
                labels = []
                while len(label_arr)!=0:
                    for key in sorted(label_arr.keys()):
                        if len(label_arr[key]) == 0:
                            label_arr.pop(key)
                            continue
                        labels.append(label_arr[key].pop())
                    

                train_set = labels[:train_num]
                val_set = labels[train_num:train_num+val_num]
                test_set = labels[train_num+val_num:]


            self.train_labels=train_set
            self.test_labels = test_set
            self.val_labels = val_set




            print('train_labels:',len(self.train_labels),'  val_labels:',len(self.val_labels),' test_labels:',len(self.test_labels))

            




class dataGraph:

    def __init__(self,args,mode='train'):
        super(dataGraph, self).__init__()
        self.args = args
        self.train_dataset = args.train_dataset
        self.test_dataset = args.test_dataset[0]
        self.original_datas = {}

        if mode == 'train':self.load_datasets()
        #else: self.load_test_data()

    


    def load_test_data(self):
        dataset_str = './lamp_data/test2/' + self.test_dataset + '/'
        self.test_graphs = {}
        for _ in range(5):
            file = dataset_str+'test_graph{}.npz'.format(_)
            file = np.load(file)
            feats = file['feats']

            scaler = StandardScaler()
            scaler.fit(feats)
            feats = scaler.transform(feats)

            edge_index = file['edge_index']
            labels = file['labels']
            graph = Graph(x = feats,edge_index=edge_index,is_tensor=False,labels=labels)
            self.test_graphs[_] = graph


        






    def load_datasets(self):


        for dataset in self.train_dataset:
            dataset_str = './lamp_data/' + dataset + '/'
            adj_full = sp.load_npz(dataset_str + 'adj_full.npz')
            feat = np.load(dataset_str + 'features.npy',mmap_mode='r')
            self.original_datas[dataset] = Graph(feat,adj_full)




    def fetch_subgraph(self,norm = True):
        '''
        subgraphs = {}
        for dataset in self.train_dataset:
            feats = np.load('./lamp_data/original/{}/feats0.npy'.format(dataset) )
            graph = sp.load_npz('./lamp_data/original/{}/graph0.npz'.format(dataset))
            graph = graph.tocoo()
            graph = np.vstack((graph.row, graph.col)).astype(np.long)
            scaler = StandardScaler()
            scaler.fit(feats)
            feats = scaler.transform(feats)
            subgraphs[dataset] = Graph(feats, graph)
            #split =  np.load('./lamp_data/original/{}/split0.npz'.format(dataset))
            #subgraphs[dataset].train_edge_index = torch.LongTensor(split['train'])
            #subgraphs[dataset].test_edge_index = torch.LongTensor(split['val'])
        return subgraphs


        '''
        subgraphs = {}
        #print(self.train_dataset)
        for dataset in self.train_dataset:
            sampling_graph,_ = self.sampling(dataset,self.args.subgraph_scale,norm)
            sampling_graph.adj = _
            subgraphs[dataset] = sampling_graph
        return subgraphs
        

    def subgraph_to_tensor(self, subgraphs, device):
        for key in subgraphs.keys():
            subgraphs[key].to_tensor(device)


    def sampling(self,dataset, n_samples=2000,norm =  True):
        mat = self.original_datas[dataset].edge_index
        g_vertices = list(range(mat.shape[0]))

        sample = set()
        n_iter = 100 * n_samples

        num_vertices = len(g_vertices)

        current = g_vertices[randint(0, num_vertices-1)]
        sample.add(current)
        count = 0

        while len(sample) < n_samples:
            count += 1
            if count > n_iter: return 0
            if random.random() < 0.0002:
                current = g_vertices[randint(0, num_vertices-1)]
                sample.add(current)
                continue
            else:neighbors = mat[current, :].nonzero()[1]
            if len(neighbors) == 0:
                continue
            current = random.choice(neighbors)
            sample.add(current)

        sample = sorted(sample)
        adj = mat[sample, :][:, sample]
        adj = adj.tolil()
        for i in range(len(sample)):
            adj[i, i] = 0

        adj = adj.tocoo()
        adj_ = np.vstack((adj.row, adj.col)).astype(np.long)

        
        feats = self.original_datas[dataset].x[sample]
        if norm:
            scaler = StandardScaler()
            scaler.fit(feats)
            feats = scaler.transform(feats)

        return Graph(feats, adj_),adj


















