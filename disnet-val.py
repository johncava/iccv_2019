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

#print(torch.cuda.is_available())

Train = np.load('Train.npy')                                         
Test = np.load('Val.npy')

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
            nn.Linear(100,1),
            nn.ReLU()
        )

    def forward(self,x):
        return self.disnet(x)

model = Disnet().cuda()                                                              
learning_rate = 1e-3                                                            
optimizer = optim.Adam(model.parameters(),lr=learning_rate)                     
max_epochs = 1                                                                
loss_fn = nn.MSELoss()                                                          
                                                                                
import time                                                                   
for epoch in range(max_epochs):
    epoch_loss = []
    start = time.time()                                                
    for train in Train:                                                        
        x,y = [],[]
        dirs = glob.glob(train)
        data = get_data_bbox(dirs)                                                            
        for i in data:                                                          
            x.append(i[0])                                                      
            y.append(i[1])                                                      
        x = Variable(torch.Tensor(x).cuda(),requires_grad=False).view(len(data),7)     
        pred = model(x)                                                         
        y = Variable(torch.Tensor(y).cuda(),requires_grad=False).view(len(data),1)     
        optimizer.zero_grad()                                                   
        loss = loss_fn(pred,y)
        epoch_loss.append(loss.item())                                                  
        loss.backward()                                                         
        optimizer.step()                                                        
        break

    # Validation
    validation_loss = []
    for test in Test:
        test_x,test_y = [],[]
        dirs = glob.glob(test)
        data = get_data_bbox(dirs) 
        for t in data:
            test_x.append(t[0])
            test_y.append(t[1])
        test_x = Variable(torch.Tensor(test_x).cuda(),requires_grad=False).view(len(data),7)
        test_pred = model.forward(test_x)
        test_y = Variable(torch.Tensor(test_y).cuda(),requires_grad=False).view(len(data),1)
        loss = (pred - y) ** 2                                                     
        validation_loss += loss.view(-1).detach().data.numpy().tolist()
        break
        
    end = time.time()
    torch.save(model.state_dict(), './checkpoints/disnet/disnet-3_layer-epoch_'+str(epoch)+'.model')                                                             
    print('epoch loss: ' + str(sum(epoch_loss)/len(epoch_loss)) + ', Val loss: ' + str(np.array(validation_loss).mean()) + ', Time: ' + str((end-start))) 
