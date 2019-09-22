import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from reader import *
import sys

directories = ['6','7','A','B','C','D']                                         
dataset = []                                                                    
for d in directories:                                                           
    dirs = glob.glob('./../dataset/' + d + '/images_and_csv/*/')                
    data = get_data_bbox_graph(dirs)                                                  
    dataset = dataset + data                                                    
                                                                                
fold = int(sys.argv[1])                                                         
indices = np.load('./../indices.npy')                                                
train_set,test_set = indices[fold][0], indices[fold][1] 

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
        self.w1 = Variable(torch.randn(3,3),requires_grad=True)
        self.w2 = Variable(torch.randn(3,2),requires_grad=True)
        self.w3 = Variable(torch.randn(2,1),requires_grad=True)

    def forward(self,A,D,l):
        hidden_layer_1 = F.relu(D.mm(A).mm(l).mm(self.w1))
        hidden_layer_2 = F.relu(D.mm(A).mm(hidden_layer_1).mm(self.w2))
        y_pred = F.relu(D.mm(A).mm(hidden_layer_2).mm(self.w3))
        return y_pred

model = Scinfaxi()
learning_rate = 1e-3
optimizer = optim.Adam([model.w1,model.w2,model.w3],lr=learning_rate)
max_epochs = int(sys.argv[2])
loss_fn = nn.MSELoss()

import time
for epoch in range(max_epochs):
    for train in train_set:
        x,l,y = [],[],[]
        for i in dataset[train]:
            x.append(i[0])
            l.append(i[1])
            y.append(i[2])
        if (len(x) < 2) or (len(l) < 2) or (len(y) < 2):
            continue 
        A, D = create_graph(x)
        A = Variable(torch.Tensor(A),requires_grad=False).view(len(dataset[train]),len(dataset[train]))
        D = Variable(torch.Tensor(D),requires_grad=False).view(len(dataset[train]),len(dataset[train]))
        l = Variable(torch.Tensor(l),requires_grad=False).view(len(dataset[train]),3)
        pred = model(A,D,l)
        y = Variable(torch.Tensor(y),requires_grad=False).view(len(dataset[train]),1)
        optimizer.zero_grad()
        loss = loss_fn(pred,y)
        loss.backward()
        optimizer.step()

# Testing                                                                       
total_loss = []
total_rel = []                                                                 
for test in test_set:                                                           
    test_x,test_l,test_y = [],[],[]                                                       
    for t in dataset[test]:                                                              
        test_x.append(t[0])
        test_l.append(t[1])                                         
        test_y.append(t[2])                                         
    if (len(test_x) < 2) or (len(test_l) < 2) or (len(test_y) < 2):                        
        continue                                                            
    A, D = create_graph(test_x)                                                  
    A = Variable(torch.Tensor(A),requires_grad=False).view(len(dataset[test]),len(dataset[test]))
    D = Variable(torch.Tensor(D),requires_grad=False).view(len(dataset[test]),len(dataset[test]))
    l = Variable(torch.Tensor(test_l),requires_grad=False).view(len(dataset[test]),3)
    test_pred = model.forward(A,D,l)    
    test_y = Variable(torch.Tensor(test_y),requires_grad=False).view(len(dataset[test]),1)
    loss = (test_pred - test_y) ** 2                                                      
    rel_loss = torch.abs(test_pred - test_y)
    total_rel += rel_loss.view(-1).detach().data.numpy().tolist()                                                      
    total_loss += loss.view(-1).detach().data.numpy().tolist()
print(np.array(total_rel).mean())                                                                
print(np.array(total_loss).mean())  