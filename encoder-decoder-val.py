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

class EncoderDecoder(nn.Module):

    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.cnn1 = nn.Conv2d(3,10,kernel_size=(8,8))
        self.pool1 = nn.MaxPool2d(4,return_indices=True)
        self.cnn2 = nn.Conv2d(10,25,kernel_size=(8,8))
        self.pool2 = nn.MaxPool2d(4,return_indices=True)
        self.cnn3 = nn.Conv2d(25,50,kernel_size=(5,5))
        self.pool3 = nn.MaxPool2d(4,return_indices=True)

        self.unpool1 = nn.MaxUnpool2d(4)
        self.decnn1 = nn.ConvTranspose2d(50,25,kernel_size=(5,5))
        self.unpool2 = nn.MaxUnpool2d(4)
        self.decnn2 = nn.ConvTranspose2d(25,10,kernel_size=(8,8))
        self.unpool3 = nn.MaxUnpool2d(4)
        self.decnn3 = nn.ConvTranspose2d(10,1,kernel_size=(8,8))

    def forward(self,x):
        x = F.relu(self.cnn1(x))
        s1 = x.size()
        x, i1 = self.pool1(x)
        x = F.relu(self.cnn2(x))
        s2 = x.size()
        x, i2 = self.pool2(x)
        x = F.relu(self.cnn3(x))
        s3 = x.size()
        x, i3 = self.pool3(x)

        x = self.unpool1(x,i3,s3)
        x = F.relu(self.decnn1(x))
        x = self.unpool2(x,i2,s2)
        x = F.relu(self.decnn2(x))
        x = self.unpool3(x,i1,s1)
        x = F.relu(self.decnn3(x))
        return x

model = EncoderDecoder().cuda()
learning_rate = 1e-3
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
max_epochs = 10

import time
for epoch in range(max_epochs):
    epoch_loss = []
    start = time.time()
    train_error = []
    val_error = []
    for train in Train:
        dirs = glob.glob(train)
        gdata,data = get_depth_data(dirs)
        x,y = np.array(Image.open(data[0][0])),data[0][1]
        if len(x.shape) < 2:
            train_error.append(train)
            continue
        x,y = x.reshape(1,3,480,640),y.reshape(1,1,480,640)
        x = Variable(torch.Tensor(x).cuda(), requires_grad=True)
        y = Variable(torch.Tensor(y).cuda(), requires_grad=False)
        pred = model(x)
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
        val_x, val_y = np.array(Image.open(data[0][0])), data[0][1]
        if len(val_x.shape) < 2:
            val_error.append(val)
            continue
        val_x, val_y = val_x.reshape(1,3,480,640), val_y.reshape(1,1,480,640)
        val_x = Variable(torch.Tensor(val_x).cuda(),requires_grad=False)
        val_pred = model.forward(val_x)
        val_y = Variable(torch.Tensor(val_y).cuda(),requires_grad=False)
        val_loss = loss_fn(val_pred,val_y)
        validation_loss.append(val_loss.item())
        
    end = time.time()
    torch.save(model.state_dict(), './checkpoints/EncoderDecoder/EncoderDecoder-3_layer-epoch_'+str(epoch)+'.model') 
    print('epoch loss: ' + str(np.array(epoch_loss).mean()) + ', Val loss: ' + str(np.array(validation_loss).mean()) + ', Time: ' + str((end-start)))
