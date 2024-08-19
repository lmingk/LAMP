from turtle import pos
from copy import deepcopy
from unittest import result
import torch
import torch.nn.functional as F
import sklearn.neighbors
import copy
from collections import OrderedDict
import numpy as np
from utils import test,test2,seed_everything
from models.models import MLP,WDiscriminator
from models.autoencoder import negative_sampling
import time
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import roc_auc_score, f1_score
EPS = 1e-15





def train_wdiscriminator(graph_s, graph_t, wdiscriminator, optimizer_d, batch_d_per_iter=20):


    wdiscriminator.reset_parameters()

    for p in wdiscriminator.parameters(): p.requires_grad = True
    if not isinstance(graph_t,list):graph_t = [graph_t]

    for j in range(batch_d_per_iter):
        wdiscriminator.train()

        w1s = []
        for graph in graph_t:
            w1s.append(wdiscriminator(graph.x,graph.edge_index))


        w0 = wdiscriminator(graph_s.x,graph_s.edge_index)
        w1 = torch.vstack(w1s)

        loss = -torch.mean(w1) + torch.mean(w0)

        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()

        for p in wdiscriminator.parameters(): p.data.clamp_(-0.01, 0.01)

    return wdiscriminator






def domain_tansformation(graph_s, graph_syn, args, device):

    model = MLP(graph_s.x.shape[1], args.hidden_dims, args.share_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    wdiscriminator = WDiscriminator(args.share_dims).to(device)
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.wd_lr, weight_decay=5e-4)

    best_value = 1e8
    cnt = 0
    best_model = copy.deepcopy(model).to(device)
    while True:

        model.train()

        loss_all = torch.tensor(0.0).to(device)

        wdiscriminator_copy = copy.deepcopy(train_wdiscriminator(graph_s.transformation(model) , graph_syn.syn_graphs,\
                 wdiscriminator, optimizer_wd, batch_d_per_iter=40))

        for p in wdiscriminator_copy.parameters(): p.requires_grad = False

        wdiscriminator_copy.to(device)
            

        w1s = []
        w0 = wdiscriminator_copy(model(graph_s.x), graph_s.edge_index)

        for graph in graph_syn.syn_graphs:
            w1s.append(wdiscriminator_copy(graph.x, graph.edge_index))
        w1 = torch.vstack(w1s)
        loss_all = torch.mean(w1) - torch.mean(w0)



        value = loss_all.item()
        print('XXXXXXXXXXXXXXXXXXXXX',value)
        if cnt >0:return best_model
        if value < 0:
            cnt+=1
        elif value>=0 and value < best_value:
            best_value = value
            cnt = 0
            best_model = copy.deepcopy(model)
        else: cnt+=1       
        #print(cnt)


        
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()











def train_graph_by_link(mlp,model, graph,optimizer, device,centroids =None):

    mlp.to(device)
    model.to(device)

    value = 0
    count = 0

    if centroids != None: num_classes = centroids.size(0)

    result = []
    for epoch in range(1,1000):
        

        model.train()
        mlp.train()
        
        weights = OrderedDict(model.named_parameters())

        if model.task == 'link':z = model.encode(mlp(graph.x), graph.train_edge_index, weights)
        else:
            matrix = torch.vstack([mlp(graph.x),centroids])
            z = model.encode(matrix, graph.edge_index, weights, inner_loop=True)


        
        if model.task == 'link':loss = model.recon_loss(z,graph.train_edge_index)
        else:
            if epoch<=50:loss = model.class_loss(z, graph.pos_edge_index,graph.neg_edge_index)
            else:loss = model.class_loss2(z, graph.pos_edge_index,graph.neg_edge_index)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            p.data.clamp_(-0.1, 0.1)
        #centroids.data.clamp_(-0.1, 0.1)

        model.eval()
        mlp.eval()
        with torch.no_grad():
            weights = OrderedDict(model.named_parameters())
            

            if model.task == 'link':z = model.encode(mlp(graph.x), graph.train_edge_index, weights)
            else:
                matrix = torch.vstack([mlp(graph.x),centroids])
                z = model.encode(matrix, graph.edge_index, weights, inner_loop=True)


            if model.task == 'link':score = model.test(z,graph.test_edge_index,graph.test_edge_index_negative )[0]
            else: 
                score = model.class_test(z,[_ for _ in range(matrix.size(0)-num_classes,matrix.size(0))],graph.test_labels ,graph.labels[graph.test_labels])
        if epoch in [0,2,4,6,8,10,20,50,100,150,200,300,400,500,600,700,800,900,1000]:print(epoch,value,loss.item())

        if value<score:
            value = score
            count = 0
        else:count+=1
        result.append(value)
    print('the best value: ', value)
    return result








def lamp_gnn_gradient_step(args, pretrain_model, fake_model, wdiscriminator, syn_graphs, subgraphs, MLPs, feature_MLPs, optimizer_wd, optimizer_all, optimizer_feats, epoch, device, file):

    task_losses = []

    torch.autograd.set_detect_anomaly(True)

    graphs = syn_graphs.transformation(feature_MLPs)
    Dis_loss_ = torch.tensor(0.0).to(device)
    for id,name in enumerate(args.train_dataset):
        data_graph = subgraphs[name]
        wdiscriminator_copy = copy.deepcopy(
                train_wdiscriminator(data_graph.transformation(MLPs[name]), graphs, wdiscriminator,
                                     optimizer_wd, batch_d_per_iter=50))
        for p in wdiscriminator_copy.parameters(): p.requires_grad = False
        wdiscriminator_copy.to(device)
        w1s = []
        for graph in graphs:w1s.append(wdiscriminator_copy(graph.x.detach(),graph.edge_index))
        w1 = torch.vstack(w1s)
        w0 = wdiscriminator_copy(MLPs[name](data_graph.x), data_graph.edge_index)
        Dis_loss_ += (torch.mean(w1) - torch.mean(w0))
    optimizer_all.zero_grad()
    Dis_loss_.backward()
    optimizer_all.step()

    pretrain_model.train()
    fake_model.train()
    fake_model.reset_parameters()
    
    aucs = []
    aps = []
    fast_weights = OrderedDict(fake_model.named_parameters())  
    with torch.no_grad():
        for id,name in enumerate(args.train_dataset):
            data_graph = subgraphs[name]
            z_val = fake_model.encode(MLPs[name](data_graph.x), data_graph.edge_index , fast_weights)
            aucb,_ = fake_model.test(z_val,data_graph.edge_index, negative_sampling(data_graph.edge_index, data_graph.x.size(0)))
            aps.append(str(round(aucb,3)))


    
    
    for inner_batch in range(args.inner_train_steps//2):
        #print(inner_batch)
        loss = torch.tensor(0.0).to(device)
        for index in range(args.syn_graph_num):
            syn_graph= syn_graphs.syn_graphs[index]
            mlp = feature_MLPs[index]
            z = fake_model.encode(mlp(syn_graph.x), syn_graph.edge_index,fast_weights)
            loss += fake_model.recon_loss(z,syn_graph.edge_index) 
        print(loss.item())

        gradients = torch.autograd.grad(loss, fast_weights.values(), \
                                            allow_unused=True, create_graph=True)
        gradients = [0 if grad is None else grad for grad in gradients]
        #print(list(zip(fast_weights.keys(), gradients)))

        fast_weights = OrderedDict(
            (name, torch.clamp((param - 0.05 * grad), -0.1, 0.1))
            for ((name, param), grad) in zip(fast_weights.items(), gradients)
        )
    Syn_loss = torch.tensor(0.0).to(device)
    for id,name in enumerate(args.train_dataset):
        data_graph = subgraphs[name]
        z_val = fake_model.encode(MLPs[name](data_graph.x).detach(), data_graph.edge_index , fast_weights)

        Syn_loss+=fake_model.recon_loss(z_val, data_graph.edge_index)

        with torch.no_grad():auc, ap = fake_model.test(z_val,data_graph.edge_index, negative_sampling(data_graph.edge_index, data_graph.x.size(0)))
        aucs.append(str(round(auc,3)))


    Link_loss = torch.tensor(0.0).to(device)
    for index in range(args.syn_graph_num):
        syn_graph= syn_graphs.syn_graphs[index]
        mlp = feature_MLPs[index]
        syn_graph.link_split(0.2,0.2)
        weights = OrderedDict(pretrain_model.named_parameters())
        fast_weights =  pretrain_model.encoder.modulate(mlp(syn_graph.x).detach(), syn_graph.train_edge_index,weights)

        for inner_batch in range(args.inner_train_steps):

            z = pretrain_model.encode(mlp(syn_graph.x).detach(), syn_graph.train_edge_index,fast_weights)
            loss = pretrain_model.recon_loss(z,syn_graph.train_edge_index) #这里计算量可能相当大

            gradients = torch.autograd.grad(loss, fast_weights.values(), \
                                            allow_unused=True, create_graph=True)
            gradients = [0 if grad is None else grad for grad in gradients]

            fast_weights = OrderedDict(
                    (name, torch.clamp((param - args.inner_lr * grad), -0.1, 0.1))
                    for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )

        z_val = pretrain_model.encode(mlp(syn_graph.x), syn_graph.test_edge_index , fast_weights)
        Link_loss+=pretrain_model.recon_loss(z_val, syn_graph.test_edge_index)
    graphs = [subgraphs[name].transformation(MLPs[name]) for name in args.train_dataset]
    Dis_loss = torch.tensor(0.0).to(device)
    for index in range(args.syn_graph_num):
        syn_graph= syn_graphs.syn_graphs[index]

        wdiscriminator_copy = copy.deepcopy(
                train_wdiscriminator(syn_graph.transformation(feature_MLPs[index]), graphs, wdiscriminator,
                                     optimizer_wd, batch_d_per_iter=50))

        for p in wdiscriminator_copy.parameters(): p.requires_grad = False
        wdiscriminator_copy.to(device)


        w1s = []
        for graph in graphs:w1s.append(wdiscriminator_copy(graph.x.detach(),graph.edge_index))
        w1 = torch.vstack(w1s)
        w0 = wdiscriminator_copy(feature_MLPs[index](syn_graph.x), syn_graph.edge_index)

        Dis_loss += (torch.mean(w1) - torch.mean(w0))


    loss =  Link_loss + Dis_loss + Syn_loss
    task_losses.append(loss)

    file.write('{},{},{:.3f},{:.3f},{:.3f},{:.3f},{},{}\n'.format(epoch, '-'.join(args.train_dataset),Syn_loss.item(),\
            Link_loss.item(),Dis_loss.item(),Dis_loss_.item(),'-'.join(aps),'-'.join(aucs)))
    file.flush()

    if len(task_losses) != 0:
        optimizer_feats.zero_grad()
        pretrain_batch_loss = torch.stack(task_losses).mean()
        pretrain_batch_loss.backward()

        optimizer_feats.step()
    for p in pretrain_model.parameters():
        p.data.clamp_(-0.1, 0.1)
    return pretrain_model,Link_loss.item()




