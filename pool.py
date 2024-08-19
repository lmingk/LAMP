from os import device_encoding
from models.models import MLP,glorot
import torch
import numpy as np
import random
from torch.nn import Parameter
from data.load_data import Graph
from collections import defaultdict

class graph_pool():
    def __init__(self,args,dataloader=None,device=None,parameters = True,addr = None):
        super(graph_pool, self).__init__()
        self.syn_graphs = []

        if addr != None:
            graphs  = np.load(addr,allow_pickle=True)['arr_0']
            for graph in graphs:
                node = torch.FloatTensor(graph[0]).to(device)
                edge = torch.from_numpy(graph[1].astype(np.long)).to(device)
                self.syn_graphs.append(Graph(node,edge,True))
        else:
            for _ in range(args.syn_graph_num):
                


                ################# degree base
                
                dataset = args.train_dataset[_ % len(args.train_dataset)]
                subgraph,matrix = dataloader.sampling(dataset,args.subgraph_scale)
                '''
                for i in range(10):
                    subgraph_,matrix_ = dataloader.sampling(dataset,args.subgraph_scale)
                    if subgraph.edge_index.shape[1]<subgraph_.edge_index.shape[1]:
                        subgraph = subgraph_
                        matrix = matrix_
                '''




                edge_index = subgraph.edge_index
                edge = torch.from_numpy(edge_index).to(device)

                if parameters:
                    node = Parameter(torch.FloatTensor(args.subgraph_scale, args.share_dims).to(device))
                    glorot(node)
                else: node = torch.FloatTensor(subgraph.x).to(device)


                degrees = defaultdict(lambda: set())
                for i in range(edge_index.shape[1]):
                    x,y = edge_index[0,i],edge_index[1,i]
                    degrees[x].add(y)
                    degrees[y].add(x)


                '''
                degree_copy = {}
                for key in degrees.keys():degree_copy[key] = np.array(list(degrees[key]))
                degree_copy = sorted(degree_copy.items(),key= lambda a:len(a[1]))
                degree_copy = [list(item) for item in degree_copy]
                label_nodes = set()
                for i in range(1000):
                    n = degree_copy.pop()
                    label_nodes.add(n[0])
                    for item in degree_copy:
                        item[1] = np.setdiff1d(item[1],n[1])
                    degree_copy = sorted(degree_copy,key= lambda a:len(a[1]))
                
                label_nodes = list(label_nodes)
                '''
                



                #deg = sorted(degrees.items(),key = lambda a:len(a[1]),reverse= True)
                #label_nodes = [a[0] for a in deg[:20] ]  #[a[0] for a in deg if len(a[1])>20]  
                
                #labels ={}
                #for n in label_nodes:labels[n] = np.array(list(degrees[n]))
                labels = degrees
                
                g = Graph(node,edge,True)
                g.labels = labels
                self.syn_graphs.append(g)
                

                ################

                '''
                dataset = random.choice(args.train_dataset)
                subgraph,matrix = dataloader.sampling(dataset,args.subgraph_scale)
                edge_index = subgraph.edge_index




                labels = {}
                for i in range(20):labels[args.subgraph_scale+i] = np.random.choice(args.subgraph_scale,100,replace = False)
                new_links = []
                for key in labels.keys():
                    for neighbor in labels[key]:
                        new_links.append((key,neighbor))
                        new_links.append((neighbor,key))
                
                edge = torch.from_numpy(np.hstack([edge_index,np.array(new_links).T])).to(device)
                node = Parameter(torch.FloatTensor(args.subgraph_scale+len(labels), args.share_dims).to(device))
                glorot(node)

                g = Graph(node,edge,True)
                g.labels = labels
                self.syn_graphs.append(g)
                '''







    def get_syn_graphs(self):
        results = []
        for graph in self.syn_graphs:
            x = graph.x.detach().cpu().numpy()
            edge_index = graph.edge_index.cpu().numpy()
            results.append((x,edge_index))
        return results


    def detach(self):
        results = []
        for graph in self.syn_graphs:
            x = graph.x.detach()
            edge_index = graph.edge_index
            results.append(Graph(x,edge_index,True))
        return results


    def transformation(self,mlps):
        results = []
        for i in range(len(mlps)):
            graph = self.syn_graphs[i]
            mlp = mlps[i]
            x = mlp(graph.x).detach()
            edge_index = graph.edge_index
            results.append(Graph(x,edge_index,True))
        return results












