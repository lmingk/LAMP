

import torch
seed = 0
torch.manual_seed(seed)

import argparse

from pool import *
from models.models import *
from data.load_data import *
from models.autoencoder import MyGAE
from train import *



parser = argparse.ArgumentParser()


parser.add_argument('--train_dataset', type=list, default=['reddit' ,'academic','yelp'])
parser.add_argument('--test_dataset', type=list, default=['yelp'])
parser.add_argument('--share_dims', type=int, default=128)
parser.add_argument('--hidden_dims', type=int, default=128)
parser.add_argument('--subgraph_scale', type=int, default=2000)
parser.add_argument('--syn_graph_num', default=10, type=int)
parser.add_argument('--model_lr', type=float, default=0.005)
parser.add_argument('--feature_lr', type=float, default=0.0001)
parser.add_argument('--wd_lr', type=float, default=0.005)
parser.add_argument('--inner_lr', type=float, default=0.01)
parser.add_argument('--weight', type=float, default=1)
parser.add_argument('--model', type=str, default='GAE')
parser.add_argument('--output_dims', default=128, type=int)
parser.add_argument('--inner_train_steps', default=10, type=int)
parser.add_argument('--layer_norm', default=False, action='store_true',help='use layer norm')
parser.add_argument('--cuda', type = int,default=-1)





def lamp_gnn(args, device):

    test_dataset = args.test_dataset[0]
    file = open('./result/pretrain_process_for_{}_share{}_lamp5.csv'.format(test_dataset,args.share_dims), 'w')
    dataloader = dataGraph(args)

    MLPs = {}
    for dataset in args.train_dataset:
        MLPs[dataset] = MLP(dataloader.original_datas[dataset].x.shape[1],args.hidden_dims,args.share_dims).to(device)

    syn_graphs = graph_pool(args,dataloader,device,False)
    feature_MLPs = [0]*args.syn_graph_num
    for i in range(args.syn_graph_num):
        feature_MLPs[i] = MLP(syn_graphs.syn_graphs[i].x.shape[1],args.hidden_dims,args.share_dims).to(device)


    wdiscriminator = WDiscriminator(args.share_dims)
    
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.wd_lr, weight_decay=5e-4)
    wdiscriminator.to(device)

    params = []
    for value in MLPs.values():
        params.extend(value.parameters())
    pretrain_model = MyGAE(LampSignatureEncoder(args, args.share_dims, args.output_dims))
    pretrain_model.to(device)

    fake_model = MyGAE(MetaEncoder(args, args.share_dims, args.output_dims))
    fake_model.to(device)

    optimizer_all = torch.optim.Adam([{'params':params,'lr':args.feature_lr}], lr=args.model_lr,weight_decay=5e-4)

    feature_params = []
    for value in feature_MLPs:
        feature_params.extend(value.parameters())

    optimizer_feats = torch.optim.Adam([{'params':feature_params,'lr':args.feature_lr},{'params':pretrain_model.parameters()},{'params':fake_model.parameters()}], lr=args.model_lr, weight_decay=5e-4)


    epoch = 0
    loss_value = 900
    cnt = 0
    optimal_model = pretrain_model
    last_model = pretrain_model

    while True:
        subgraphs = dataloader.fetch_subgraph()
        dataloader.subgraph_to_tensor(subgraphs, device)
        pretrain_model,loss = lamp_gnn_gradient_step(args, pretrain_model, fake_model, wdiscriminator, syn_graphs, subgraphs, MLPs, feature_MLPs, optimizer_wd,
                                                     optimizer_all, optimizer_feats, epoch, device, file)

        print('loss',loss,cnt,epoch)        
        if loss<loss_value:
            optimal_model = last_model
            loss_value = loss
            cnt=0
        else:cnt+=1
        if cnt == 50 :
            torch.save(optimal_model.encoder.state_dict(),'./result/lamp/pretrain_encoder_for_{}_share{}_lamp5.pth'.format(test_dataset,args.share_dims))
            for id,name in enumerate(args.train_dataset):subgraphs[name].x = MLPs[name](subgraphs[name].x).detach()
            results = []
            for graph in subgraphs.values():
                x = graph.x.detach().cpu().numpy()
                edge_index = graph.edge_index.cpu().numpy()
                results.append((x,edge_index))
            
            np.savez('./result/lamp/pretrain_graphs_for_{}_share{}_lamp5.npz'.format(test_dataset,args.share_dims),results) #syn_graphs.get_syn_graphs())
            file.close()
            break
            
        last_model = deepcopy(pretrain_model)

        epoch+=1



def test_graph_lamp(args, device,index,task,ratio):
    dataset = args.test_dataset[0]
    graph = load_test_graph(dataset,index)  # You can extract random graph. It is for reference only.
    graph.to_tensor(device)
    graph.identify(task,ratio)  # You can split randomly. It is for reference only.

    LampEncoder = LampSignatureEncoder(args, args.share_dims, args.output_dims)
    

    model = MyGAE(LampEncoder) 
    model.encoder.load_state_dict(torch.load('./result/lamp/pretrain_encoder_for_{}_share{}_lamp5.pth'.format(dataset,args.share_dims)), strict=True)
    if task == 'node': 
        if len(graph.labels.shape) == 2:model.task = 'node_m'
        else:model.task = 'node'
    else:model.task = 'link'

    syn_graphs = graph_pool(args = args,addr = './result/lamp/pretrain_graphs_for_{}_share{}_lamp5_{}.npz'.format(dataset,args.share_dims),device = device)

    tmp = graph.edge_index
    if task == 'link':graph.edge_index = graph.train_edge_index
    mlp =domain_tansformation(graph, syn_graphs, args, device)
    graph.edge_index = tmp

    mlp.to(device)
    model.to(device)


    weights = OrderedDict(model.named_parameters())
    

    if task == 'link':fast_weights = model.encoder.modulate(mlp(graph.x), graph.train_edge_index, weights)
    else:fast_weights = model.encoder.modulate(mlp(graph.x), graph.edge_index, weights) 

    encoder = MetaEncoder(args, args.share_dims, args.output_dims)
    encoder.to(device)

    with torch.no_grad():

        encoder.conv1.weight.copy_(fast_weights['encoder.conv1.weight'])
        encoder.conv1.bias.copy_(fast_weights['encoder.conv1.bias'])
        encoder.conv2.weight.copy_(fast_weights['encoder.conv2.weight'])
        encoder.conv2.bias.copy_(fast_weights['encoder.conv2.bias'])
    model.encoder = encoder


    if task == 'link':
        mlp.Dropout.p = 0.6
        optimizer = torch.optim.Adam( [{'params':mlp.parameters(),},{'params':model.encoder.parameters()}], lr= 0.0002, weight_decay=5e-4)

        return train_graph_by_link(mlp = mlp,model = model, graph=graph,optimizer=optimizer,device=device)

    else:
        mlp.Dropout.p = 0.8

        if len(graph.labels.shape) == 2:
            task_name = 'node_m'
            num_classes = int(graph.labels.shape[1])
        else:
            num_classes = int(graph.labels.max().item()+1)
            task_name = 'node'


        centroids = []
        pos_link = []
        neg_link = []
        
        with torch.no_grad():
            matrix = mlp(graph.x)

            for i in range(num_classes):
                lab = graph.labels[graph.train_labels]
            
                if task_name == 'node':
                    tmp = np.where((lab == i).detach().cpu().numpy())[0].tolist()
                    nodes = [graph.train_labels[_] for _ in tmp]
 
                    for node in nodes:
                        for j in range(num_classes):
                            if j == i:pos_link.append((node,j+graph.x.size(0)))
                            else: neg_link.append((node,j+graph.x.size(0)))
                    if len(tmp) == 0: centroids.append(matrix[graph.train_labels].mean(0))
                    else: centroids.append(matrix[graph.train_labels][tmp].mean(0))
                if task_name == 'node_m':
                    tmp = np.where((lab[:,i]==1).detach().cpu().numpy())[0].tolist()
                    nodes = [graph.train_labels[_] for _ in tmp]
                    for j in graph.train_labels:
                        if j in nodes:pos_link.append((j,i+graph.x.size(0)))
                        else: neg_link.append((j,i+graph.x.size(0)))
                    if len(tmp) == 0:centroids.append(matrix[graph.train_labels].mean(0))
                    else: centroids.append(matrix[graph.train_labels][lab[:,i]==1].mean(0)) 

        
            graph.pos_edge_index = torch.tensor(np.array(pos_link).T).to(device)
            graph.neg_edge_index = torch.tensor(np.array(neg_link).T).to(device)
            additional_edge_index = torch.tensor(np.array(pos_link+[(j,i) for i,j in pos_link]).T).to(device)
            #graph.edge_index = torch.hstack([graph.edge_index,additional_edge_index])


            centroids  = Parameter(torch.vstack(centroids).to(device))
            #glorot(centroids)
            #centroids = nn.Dropout(0.2)(centroids)


        if dataset in ['academic' ,'yelp']: optimizer = torch.optim.Adam( [{'params':mlp.parameters()},{'params':model.encoder.parameters()},{'params':centroids}], lr= 0.0002 , weight_decay=5e-4)
        else: optimizer = torch.optim.Adam( [{'params':mlp.parameters()},{'params':model.encoder.parameters()},{'params':centroids}], lr= 0.001 ,weight_decay=5e-4)


        return train_graph_by_link(mlp = mlp,model = model, graph=graph,centroids=centroids,optimizer=optimizer,device=device)










args = parser.parse_args()
if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")



lamp_gnn(args, device)
test_graph_lamp(args, device,0,'link',0.05)



