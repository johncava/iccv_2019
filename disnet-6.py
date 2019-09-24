import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from reader import *
from sklearn.model_selection import KFold
import glob
import sys

directories = ['6','7','A','B','C','D']                                         
dataset = []                                                                    
for d in directories:                                                           
    dirs = glob.glob('./../dataset/' + d + '/images_and_csv/*/')                   
    data = get_data_bbox(dirs)                                                  
    dataset = dataset + data

fold = int(sys.argv[1])
indices = np.load('./../indices.npy')
train_set,test_set = indices[fold][0], indices[fold][1]

class Disnet(nn.Module):
    
    def __init__(self):
        super(Disnet,self).__init__()
        self.disnet = nn.Sequential(
            nn.Linear(7,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,1),
            nn.ReLU()
        )

    def forward(self,x):
        return self.disnet(x)

model = Disnet()                                                              
learning_rate = 1e-3                                                            
optimizer = optim.Adam(model.parameters(),lr=learning_rate)                     
max_epochs = int(sys.argv[2])                                                                  
loss_fn = nn.MSELoss()                                                          
                                                                                
import time                                                                   
for epoch in range(max_epochs):                                                 
    for train in train_set:                                                        
        x,y = [],[]                                                             
        for i in dataset[train]:                                                          
            x.append(i[0])                                                      
            y.append(i[1])                                                      
        x = Variable(torch.Tensor(x),requires_grad=False).view(len(dataset[train]),7)     
        pred = model(x)                                                         
        y = Variable(torch.Tensor(y),requires_grad=False).view(len(dataset[train]),1)     
        optimizer.zero_grad()                                                   
        loss = loss_fn(pred,y)                                                  
        loss.backward()                                                         
        optimizer.step()                                                        
        break

# Testing
total_loss = []
total_rel = []
for test in test_set:
    test_x,test_y = [],[]
    for t in dataset[test]:
        test_x.append(t[0])
        test_y.append(t[1])
    test_x = Variable(torch.Tensor(test_x),requires_grad=False).view(len(dataset[test]),7)
    test_pred = model.forward(test_x)
    test_y = Variable(torch.Tensor(test_y),requires_grad=False).view(len(dataset[test]),1)
    loss = (pred - y) ** 2
    rel_loss = torch.abs(test_pred - test_y)
    total_rel += rel_loss.view(-1).detach().data.numpy().tolist()                                                      
    total_loss += loss.view(-1).detach().data.numpy().tolist()
print(np.array(total_rel).mean())                                                                
print(np.array(total_loss).mean()) 
