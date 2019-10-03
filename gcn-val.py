import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from reader import *
import sys

#print(torch.cuda.is_available())

Train = np.load('Train.npy')                                         
Val = np.load('Val.npy')

def create_graph(points):
    graph = np.zeros((len(points),len(points)))
    for i, a in enumerate(points):
        for j, b in enumerate(points):
            if i == j:
                continue
            a = np.array(a)
            b = np.array(b)
            graph[i,j] = np.sqrt(np.sum((a-b)**2))
    d = []
    for i in range(len(points)):
        d.append(np.sum(graph[i,:]))
    D = np.diag(d)
    D = np.linalg.cholesky(D)
    D = np.linalg.inv(D)
    I = np.eye(len(points))
    graph_hat = graph + I
    return graph_hat, D

class Scinfaxi(nn.Module):

    def __init__(self):
        super(Scinfaxi,self).__init__()
        self.w1 = Variable(torch.randn(3,3).cuda(),requires_grad=True)
        self.w2 = Variable(torch.randn(3,2).cuda(),requires_grad=True)
        self.w3 = Variable(torch.randn(2,1).cuda(),requires_grad=True)

    def forward(self,A,D,l):
        hidden_layer_1 = F.relu(D.mm(A).mm(l).mm(self.w1))
        hidden_layer_2 = F.relu(D.mm(A).mm(hidden_layer_1).mm(self.w2))
        y_pred = F.relu(D.mm(A).mm(hidden_layer_2).mm(self.w3))
        return y_pred

model = Scinfaxi()
learning_rate = 1e-3
optimizer = optim.Adam([model.w1,model.w2,model.w3],lr=learning_rate)
max_epochs = 10
loss_fn = nn.MSELoss()

import time
for epoch in range(max_epochs):
    epoch_loss = []
    start = time.time()
    for train in Train:
        x,l,y = [],[],[]
        dirs = glob.glob(train)
        data = get_data_bbox_graph(dirs)[0]
        for i in data:
            x.append(i[0])
            l.append(i[1])
            y.append(i[2])
        if (len(x) < 2) or (len(l) < 2) or (len(y) < 2):
            continue 
        A, D = create_graph(x)
        A = Variable(torch.Tensor(A),requires_grad=False).view(len(data),len(data)).cuda()
        D = Variable(torch.Tensor(D),requires_grad=False).view(len(data),len(data)).cuda()
        l = Variable(torch.Tensor(l),requires_grad=False).view(len(data),3).cuda()
        pred = model(A,D,l)
        y = Variable(torch.Tensor(y).cuda(),requires_grad=False).view(len(data),1)
        optimizer.zero_grad()
        loss = loss_fn(pred,y)
        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    
    # Validation                                                                       
    validation_loss = []                                                                 
    for val in Val:                                                           
        val_x,val_l,val_y = [],[],[]
        dirs = glob.glob(val)
        data = get_data_bbox_graph(dirs)[0]
        for t in data:
            val_x.append(t[0])
            val_l.append(t[1])                                         
            val_y.append(t[2])                                         
        if (len(val_x) < 2) or (len(val_l) < 2) or (len(val_y) < 2):
            continue                                                            
        A, D = create_graph(val_x)
        A = Variable(torch.Tensor(A),requires_grad=False).view(len(data),len(data)).cuda()
        D = Variable(torch.Tensor(D),requires_grad=False).view(len(data),len(data)).cuda()
        l = Variable(torch.Tensor(val_l),requires_grad=False).view(len(data),3).cuda()
        val_pred = model.forward(A,D,l)    
        val_y = Variable(torch.Tensor(val_y).cuda(),requires_grad=False).view(len(data),1)
        loss = loss_fn(val_pred,val_y)
        validation_loss.append(loss.item())

    end = time.time()
    torch.save(model.state_dict(), './checkpoints/gcn/gcn-3_layer-epoch_'+str(epoch)+'.model') 
    print('epoch loss: ' + str(np.array(epoch_loss).mean()) + ', Val loss: ' + str(np.array(validation_loss).mean()) + ', Time: ' + str((end-start))) 
