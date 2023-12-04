from models.models import MLP,glorot
import torch
import numpy as np
from torch.nn import Parameter
from load_data import Graph
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
                




    def train_AE(self,x,args,device):

        
        model = MLP(x.shape[1], args.hidden_dims, args.share_dims).to(device)
        model_out = MLP(args.share_dims, args.hidden_dims, x.shape[1]).to(device)          
        optimizer = torch.optim.Adam([{'params':model.parameters(),'lr':0.001},{'params':model_out.parameters(),'lr':0.0005}])
        cnt = 0
        loss_value = 900
        k = 0 
        while True:
            k+=1
            model.train()
            model_out.train()
            loss_AE =torch.nn.MSELoss(size_average=True)(x,model_out(model(x))) 
            optimizer.zero_grad()
            loss_AE.backward()
            optimizer.step()  
            if k% 100 == 1:print(loss_AE.item())

            if loss_AE<loss_value:
                loss_value = loss_AE
                cnt=0
            else:cnt+=1 

            if cnt == 50:
                model.eval()
                return model(x).detach()




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












