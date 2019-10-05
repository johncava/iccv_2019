import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from reader import *
import glob
import sys

#print(torch.cuda.is_available())

Train = np.load('Train.npy')                                         
Val = np.load('Val.npy')

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
max_epochs = 10                                                             
loss_fn = nn.MSELoss()                                                          
                                                                                
import time                                                                   
for epoch in range(max_epochs):
    epoch_loss = []
    start = time.time()                                                
    for train in Train:                                                        
        x,y = [],[]
        dirs = glob.glob(train)
        data = get_data_bbox(dirs)[0]
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

    # Validation
    validation_loss = []
    for val in Val:
        val_x,val_y = [],[]
        dirs = glob.glob(val)
        data = get_data_bbox(dirs)[0] 
        for t in data:
            val_x.append(t[0])
            val_y.append(t[1])
        val_x = Variable(torch.Tensor(val_x).cuda(),requires_grad=False).view(len(data),7)
        val_pred = model.forward(val_x)
        val_y = Variable(torch.Tensor(val_y).cuda(),requires_grad=False).view(len(data),1)
        val_loss = loss_fn(val_pred,val_y)                                                     
        validation_loss.append(val_loss.item())
        
    end = time.time()
    torch.save(model.state_dict(), './checkpoints/disnet/disnet-3_layer-epoch_'+str(epoch)+'.model')                                                             
    print('epoch loss: ' + str(np.array(epoch_loss).mean()) + ', Val loss: ' + str(np.array(validation_loss).mean()) + ', Time: ' + str((end-start))) 
