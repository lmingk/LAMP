
import torch
import copy
from collections import OrderedDict
import numpy as np
from models.models import MLP,WDiscriminator,WDiscriminator_old
from models.autoencoder import negative_sampling

EPS = 1e-15

def construct_space(model,edges_num,train_pos_edge_index,graph_x):

    if model is None:z = graph_x
    else:z = model(graph_x)

    node_embs = torch.hstack((z, z))


    if train_pos_edge_index.shape[1] <= edges_num:
        select_edges = train_pos_edge_index
    else:
        perm = np.random.choice(train_pos_edge_index.shape[1], edges_num, replace=False)
        select_edges = train_pos_edge_index[:, perm]

    edge_embs = torch.hstack((z[select_edges[0]], z[select_edges[1]]))
    source_space = torch.vstack((node_embs, edge_embs))
    return source_space





def train_wdiscriminator_old(space_s, space_t,node_num, wdiscriminator, optimizer_d, batch_d_per_iter=20):


    wdiscriminator.reset_parameters()

    for p in wdiscriminator.parameters(): p.requires_grad = True
    

    for j in range(batch_d_per_iter):
        wdiscriminator.train()

        

        w0 = wdiscriminator(space_s[:node_num])
        w1 = wdiscriminator(space_t[:node_num])

        w2 = wdiscriminator(space_s[node_num:])
        w3 = wdiscriminator(space_t[node_num:])

        #loss = -torch.mean(w1) + torch.mean(w0) -torch.mean(w3) + torch.mean(w2)   # 最大化

        l1 = -torch.mean(w1) + torch.mean(w0)
        l2 = -torch.mean(w3) + torch.mean(w2)
        
        loss = l1+l2
        if j% 40 ==0:print(loss.item())

        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()

        for p in wdiscriminator.parameters(): p.data.clamp_(-0.01, 0.01)



    #loss = -torch.mean(w1) + torch.mean(w0)
    #print(loss.item())

    return wdiscriminator











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
        #if j% 40 ==0:print(loss.item())

        optimizer_d.zero_grad()
        loss.backward()
        optimizer_d.step()

        for p in wdiscriminator.parameters(): p.data.clamp_(-0.01, 0.01)


    return wdiscriminator





    
    

def distribution_align_old(graph_s, graph_syn, args, device):

    model = MLP(graph_s.x.shape[1], args.hidden_dims, args.share_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    wdiscriminator = WDiscriminator_old(args.share_dims*2).to(device)
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.wd_lr, weight_decay=5e-4)

    best_value = 1e8
    cnt = 0
    best_model = copy.deepcopy(model).to(device)
    while True:

        model.train()

        loss_all = torch.tensor(0.0).to(device)
        for graph_t in graph_syn.syn_graphs:

            space_s = construct_space(model,graph_s.x.size(0)*2,graph_s.edge_index,graph_s.x)
            space_t = construct_space(None,graph_t.x.size(0)*2,graph_t.edge_index,graph_t.x)
            wdiscriminator_copy = copy.deepcopy(
                train_wdiscriminator_old(space_s.detach(), space_t.detach(),graph_s.x.size(0), wdiscriminator,
                                     optimizer_wd, batch_d_per_iter=80)) 

            for p in wdiscriminator_copy.parameters(): p.requires_grad = False
            wdiscriminator_copy.to(device)


            w0 = wdiscriminator_copy(space_s[:graph_s.x.size(0)])
            w1 = wdiscriminator_copy(space_t[:graph_t.x.size(0)])

            w2 = wdiscriminator_copy(space_s[graph_s.x.size(0):])
            w3 = wdiscriminator_copy(space_t[graph_t.x.size(0):])

            loss = (torch.mean(w1) - torch.mean(w0) + torch.mean(w3) - torch.mean(w2))

            loss_all+=loss

        value = loss_all.item()
        if cnt >0:return best_model
        if value < 0:
            cnt+=1
        elif value>=0 and value < best_value:
            best_value = value
            cnt = 0
            best_model = copy.deepcopy(model)
        else: cnt+=1       
        print(cnt)
        if cnt >0:return best_model

        
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()









def distribution_align2(graph_s, graph_syn, args, device):

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
        for graph_t in graph_syn.syn_graphs:

            wdiscriminator_copy = copy.deepcopy(train_wdiscriminator(graph_s.transformation(model) , graph_t,\
                 wdiscriminator, optimizer_wd, batch_d_per_iter=80))

            for p in wdiscriminator_copy.parameters(): p.requires_grad = False

            wdiscriminator_copy.to(device)
            

            
            w0 = wdiscriminator_copy(model(graph_s.x), graph_s.edge_index)
            w1 = wdiscriminator_copy(graph_t.x, graph_t.edge_index)

            loss = torch.mean(w1) - torch.mean(w0)
            #print((torch.mean(w1) - torch.mean(w0)).item())

            loss_all+=loss

        


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









def distribution_align(graph_s, graph_syn, args, device):

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




def distribution_align3(graph_s, graph_syn, args, device):

    model = MLP(graph_s.x.shape[1], args.hidden_dims, args.share_dims).to(device)
    model_out = MLP(args.share_dims, args.hidden_dims, graph_s.x.shape[1]).to(device)

    optimizer = torch.optim.Adam([{'params':model.parameters(),'lr':0.001},{'params':model_out.parameters(),'lr':0.0005}])

    wdiscriminator = WDiscriminator(args.share_dims).to(device)
    optimizer_wd = torch.optim.Adam(wdiscriminator.parameters(), lr=args.wd_lr, weight_decay=5e-4)

    best_value = 1e8
    cnt = 0
    best_model = copy.deepcopy(model).to(device)
    while True:

        model.train()

        loss_all = torch.tensor(0.0).to(device)

        wdiscriminator_copy = copy.deepcopy(train_wdiscriminator(graph_s.transformation(model) , graph_syn.syn_graphs,\
                 wdiscriminator, optimizer_wd, batch_d_per_iter=80))

        for p in wdiscriminator_copy.parameters(): p.requires_grad = False

        wdiscriminator_copy.to(device)
            

        w1s = []
        w0 = wdiscriminator_copy(model(graph_s.x), graph_s.edge_index)

        for graph in graph_syn.syn_graphs:
            w1s.append(wdiscriminator_copy(graph.x, graph.edge_index))
        w1 = torch.vstack(w1s)
        loss = torch.mean(w1) - torch.mean(w0)

        loss_AE =torch.nn.MSELoss(size_average=True)(graph_s.x,model_out(model(graph_s.x)))

        loss_all = loss_AE #+loss

        value = loss_all.item()
        if cnt >2:return best_model
        if value < 0:
            cnt+=1
        elif value>=0 and value < best_value:
            best_value = value
            cnt = 0
            best_model = copy.deepcopy(model)
        else: cnt+=1       
        print(cnt)

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()







def train_graph_by_link3(mlp,model, graph,optimizer,device,file=None):

    if mlp!= None:mlp.to(device)
    model.to(device)

    value = 0
    count = 0

    if model.task != 'link':
        #graph = copy.deepcopy(graph)
        if len(graph.labels.shape) == 2:
            task_name = 'node_m'
            num_classes = int(graph.labels.shape[1])
        else:
            num_classes = int(graph.labels.max().item()+1)
            task_name = 'node'

        centroids = []
        pos_link = []
        neg_link = []
        for i in range(num_classes):
            lab = graph.labels[graph.train_labels]
            
            if task_name == 'node':
                tmp = np.where((lab == i).detach().cpu().numpy())[0].tolist()
                nodes = [graph.train_labels[_] for _ in tmp] 
 
                for node in nodes:
                    for j in range(num_classes):
                        if j == i:pos_link.append((node,j+graph.x.size(0)))
                        else: neg_link.append((node,j+graph.x.size(0)))
                if len(tmp) == 0: centroids.append(graph.x[graph.train_labels].mean(0))
                else: centroids.append(graph.x[graph.train_labels][tmp].mean(0))
            if task_name == 'node_m':
                tmp = np.where((lab[:,i]==1).detach().cpu().numpy())[0].tolist()
                nodes = [graph.train_labels[_] for _ in tmp]
                for j in graph.train_labels:
                    if j in nodes:pos_link.append((j,i+graph.x.size(0)))
                    else: neg_link.append((j,i+graph.x.size(0)))
                if len(tmp) == 0:centroids.append(graph.x[graph.train_labels].mean(0))
                else: centroids.append(graph.x[graph.train_labels][lab[:,i]==1].mean(0)) 

        #print(centroids)
        graph.x = torch.vstack([graph.x]+centroids)
        pos_edge_index = torch.tensor(np.array(pos_link).T).to(device)
        neg_edge_index = torch.tensor(np.array(neg_link).T).to(device)
        additional_edge_index = torch.tensor(np.array(pos_link+[(j,i) for i,j in pos_link]).T).to(device)
        #graph.edge_index = torch.hstack([graph.edge_index,additional_edge_index])

    result = []
    for epoch in range(1,2000):

        model.train()
        if mlp!= None:mlp.train()
        
        weights = OrderedDict(model.named_parameters())
        if mlp!= None:
            if model.task == 'link':z = model.encode(mlp(graph.x), graph.train_edge_index, weights)
            else:z = model.encode(mlp(graph.x), graph.edge_index, weights, inner_loop=True)
        else:
            if model.task == 'link':z = model.encode(graph.x, graph.train_edge_index, weights)
            else:z = model.encode(graph.x, graph.edge_index, weights, inner_loop=True)


        
        if model.task == 'link':loss = model.recon_loss(z,graph.train_edge_index)
        else:loss = model.class_loss3(z, pos_edge_index,neg_edge_index)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            p.data.clamp_(-0.1, 0.1)

        model.eval()
        if mlp!= None:mlp.eval()


        with torch.no_grad():
            weights = OrderedDict(model.named_parameters())
            
            if mlp!= None:
                if model.task == 'link':z = model.encode(mlp(graph.x), graph.train_edge_index, weights)
                else:z = model.encode(mlp(graph.x), graph.edge_index, weights, inner_loop=True)
            else:
                if model.task == 'link':z = model.encode(graph.x, graph.train_edge_index, weights)
                else:z = model.encode(graph.x, graph.edge_index, weights, inner_loop=True)

            if model.task == 'link':score = model.test(z,graph.test_edge_index,graph.test_edge_index_negative )[0]
            else: 
                score = model.class_test(z,[_ for _ in range(graph.x.size(0)-num_classes,graph.x.size(0))],graph.test_labels ,graph.labels[graph.test_labels])



        result.append(score)
        if file!=None:file.write('{},{},{},{}\n'.format(epoch,count,score))
        if value<score:
            value = score
            count = 0
        else:count+=1
    if file!=None:file.write('the best value:,{}\n'.format(value))
    print('the best value: ', value)
    return result







def train_graph_by_link(mlp, model, graph, optimizer, device, centroids =None, file=None):

    if mlp!= None:mlp.to(device)
    model.to(device)

    value = 0
    count = 0

    num_classes = centroids.size(0)
        

    result = []
    for epoch in range(1,1000):
        


        model.train()
        if mlp!= None:mlp.train()
        
        weights = OrderedDict(model.named_parameters())
        if mlp!= None:
            if model.task == 'link':z = model.encode(mlp(graph.x), graph.train_edge_index, weights, inner_loop=True)
            else:
                matrix = torch.vstack([mlp(graph.x),centroids])
                z = model.encode(matrix, graph.edge_index, weights, inner_loop=True)
        else:
            if model.task == 'link':z = model.encode(graph.x, graph.train_edge_index, weights, inner_loop=True)
            else:z = model.encode(graph.x, graph.edge_index, weights, inner_loop=True)


        
        if model.task == 'link':loss = model.recon_loss(z,graph.train_edge_index)
        else:loss = model.class_loss(z, graph.pos_edge_index, graph.neg_edge_index)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for p in model.parameters():
            p.data.clamp_(-0.1, 0.1)


        model.eval()

        if mlp!= None:mlp.eval()


        with torch.no_grad():
            weights = OrderedDict(model.named_parameters())
            
            if mlp!= None:
                if model.task == 'link':z = model.encode(mlp(graph.x), graph.train_edge_index, weights, inner_loop=True)
                else:
                    matrix = torch.vstack([mlp(graph.x),centroids])
                    z = model.encode(matrix, graph.edge_index, weights, inner_loop=True)
            else:
                if model.task == 'link':z = model.encode(graph.x, graph.train_edge_index, weights, inner_loop=True)
                else:z = model.encode(graph.x, graph.edge_index, weights, inner_loop=True)

            if model.task == 'link':score = model.test(z,graph.test_edge_index,graph.test_edge_index_negative )[0]
            else: 
                score = model.class_test(z,[_ for _ in range(matrix.size(0)-num_classes,matrix.size(0))],graph.test_labels ,graph.labels[graph.test_labels])


        if file!=None:file.write('{},{},{},{}\n'.format(epoch,count,score))
        if value<score:
            value = score
            count = 0
        else:count+=1
        result.append(value)
    if file!=None:file.write('the best value:,{}\n'.format(value))
    print('the best value: ', value)
    return result









def lamp_gnn_gradient_step(args, pretrain_model,fake_model, wdiscriminator, syn_graphs, subgraphs, MLPs,feature_MLPs,optimizer_wd, optimizer_all,optimizer_feats,epoch, device,file):

    task_losses = []

    torch.autograd.set_detect_anomaly(True)


    # The key codes are avaiable when the paper is published



    return pretrain_model,task_losses.mean().item()

