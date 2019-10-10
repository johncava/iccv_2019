import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob as glob
from torch.autograd import Variable
from reader import get_depth_data
from PIL import Image

Train = np.load('final_dataset_train.npy')
Val = np.load('final_dataset_val.npy')

class GraphEncoderDecoder(nn.Module):

    def __init__(self):
        super(GraphEncoderDecoder, self).__init__()
        self.cnn1 = nn.Conv2d(3,10,kernel_size=(8,8))
        self.pool1 = nn.MaxPool2d(4,return_indices=True)
        self.cnn2 = nn.Conv2d(10,25,kernel_size=(8,8))
        self.pool2 = nn.MaxPool2d(4,return_indices=True)
        self.cnn3 = nn.Conv2d(25,50,kernel_size=(5,5))
        self.pool3 = nn.MaxPool2d(4,return_indices=True)

        self.unpool1 = nn.MaxUnool2d(4)
        self.decnn1 = nn.ConvTranspose2d(50,25,kernel_size=(5,5))
        self.unpool2 = nn.MaxUnpool2d(4)
        self.decnn2 = nn.ConvTranspose2d(25,10,kernel_size=(8,8))
        self.unpool3 = nn.MaxUnpool2d(4)
        self.decnn3 = nn.ConvTranspose2d(10,1,kernel_size=(8,8))

        self.a = nn.Linear(2*12,1)
        self.W = nn.Linear(12,12)
        self.P = nn.Linear(12,100)
        self.hidden = nn.Linear(2100,2000)

    def forward(self,x,g):
        g = [torch.Tensor([gi]) for gi in g]
        x = F.relu(self.cnn1(x))
        s1 = x.size()
        x, i1 = self.pool1(x)
        x = F.relu(self.cnn2(x))
        s2 = x.size()
        x, i2 = self.pool2(x)
        x = F.relu(self.cnn3(x))
        s3 = x.size()
        x, i3 = self.pool3(x)
        presize = x.size()

        h = []
        for i,hi in enumerate(g):
            a_weights = []
            h_neighbors = []
            for j, hj in enumerate(g):
                a = F.relu(self.a(torch.cat((self.W(hi),self.W(hj)),dim=1)))
                a_weights.append(a)
                h_neighbors.append(hj)
            s = []
            for aw, hn in zip(a_weights,h_neighbors):
                s.append(self.W(aw*hn))
            h.append(sum(s))
        h = torch.cat(h,dim=0)
        latent = self.P(h)
        latent = torch.mean(latent,dim=0)
        x = torch.cat((x.flatten(),latent),dim=0)
        x = self.hidden(x)
        x = x.view(presize)

        x = self.unpool1(x,i3,s3)
        x = F.relu(self.decnn1(x))
        x = self.unpool2(x,i2,s2)
        x = F.relu(self.decnn2(x))
        x = self.unpool3(x,i1,s1)
        x = F.relu(self.decnn3(x))
        return x

model = GraphEncoderDecoder().cuda()
learning_rate = 1e-3
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
max_epochs = 1

import time
for epoch in range(max_epochs):
    epoch_loss = []
    start = time.time()
    train_error = []
    val_error = []
    for train in Train:
        dirs = glob.glob(train)
        gdata,data = get_depth_data(dirs)
        gd = [g[0] for g in gdata]
        x,y = np.array(Image.open(data[0][0])),data[0][1]
        if len(x.shape) < 2:
            train_error.append(train)
            continue
        x,y = x.reshape(1,3,480,640),y.reshape(1,1,480,640)
        x = Variable(torch.Tensor(x).cuda(), requires_grad=True)
        y = Variable(torch.Tensor(y).cuda(), requires_grad=False)
        pred = model(x,gd)
        loss = loss_fn(pred,y)
        epoch_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    validation_loss = []
    for val in Val:
        val_x,val_y = [],[]
        dirs = glob.glob(val)
        g_data, data = get_depth_data(dirs)
        gd = [g[0] for g in g_data]
        val_x, val_y = np.array(Image.open(data[0][0])), data[0][1]
        if len(val_x.shape) < 2:
            val_error.append(val)
            continue
        val_x, val_y = val_x.reshape(1,3,480,640), val_y.reshape(1,1,480,640)
        val_x = Variable(torch.Tensor(val_x).cuda(),requires_grad=False)
        val_pred = model.forward(val_x,gd)
        val_y = Variable(torch.Tensor(val_y).cuda(),requires_grad=False)
        val_loss = loss_fn(val_pred,val_y)
        validation_loss.append(val_loss.item())
    
        
    end = time.time()
    torch.save(model.state_dict(), './checkpoints/GraphEncoderDecoder/GraphEncoderDecoder-3_layer-epoch_'+str(epoch)+'.model') 
    print('epoch loss: ' + str(np.array(epoch_loss).mean()) + ', Val loss: ' + str(np.array(validation_loss).mean()) + ', Time: ' + str((end-start)))
