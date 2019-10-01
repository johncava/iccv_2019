import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob as glob
from torch.autograd import Variable
from reader import get_depth_data
from PIL import Image

Train = np.load('Train.npy')                                         
Test = np.load('Val.npy')

class EncoderDecoder(nn.Module):

    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.cnn1 = nn.Conv2d(3,10,kernel_size=(8,8))
        self.cnn2 = nn.Conv2d(10,25,kernel_size=(8,8))
        self.cnn3 = nn.Conv2d(25,50,kernel_size=(5,5))

        self.decnn1 = nn.ConvTranspose2d(50,25,kernel_size=(5,5))
        self.decnn2 = nn.ConvTranspose2d(25,10,kernel_size=(8,8))
        self.decnn3 = nn.ConvTranspose2d(10,1,kernel_size=(8,8))

    def forward(self,x):
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))

        x = F.relu(self.decnn1(x))
        x = F.relu(self.decnn2(x))
        x = F.relu(self.decnn3(x))
        return x
model = EncoderDecoder().cuda()
learning_rate = 1e-3
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
max_epoch = 1

import time
for epoch in range(max_epoch):
    epoch_loss = []
    start = time.time()
    for train in Train:
        dirs = glob.glob(train)
        gdata,data = get_depth_data(dirs)
        x,y = np.array(Image.open(data[0][0])),data[0][1]
        print(x.shape,y.shape)
        x,y = x.reshape(1,3,480,640),y.reshape(1,1,480,640)
        x = Variable(torch.Tensor(x).cuda(), requires_grad=True)
        y = Variable(torch.Tensor(y).cuda(), requires_grad=False)
        pred = model(x)
        loss = loss_fn(pred,y)
        epoch_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break

    # Validation
    validation_loss = []
    for test in Test:
        test_x,test_y = [],[]
        dirs = glob.glob(test)
        g_data, data = get_depth_data(dirs)
        test_x, test_y = np.array(Image.open(data[0][0])), data[0][1]
        test_x, test_y = test_x.reshape(1,3,480,640), test_y.reshape(1,1,480,640)
        test_x = Variable(torch.Tensor(test_x).cuda(),requires_grad=False)
        test_pred = model.forward(test_x)
        test_y = Variable(torch.Tensor(test_y).cuda(),requires_grad=False)
        loss = (pred - y) ** 2
        validation_loss += loss.view(-1).cpu().detach().data.numpy().tolist()
        break
        
    end = time.time()
    torch.save(model.state_dict(), './checkpoints/EncoderDecoder/EncoderDecoder-3_layer-epoch_'+str(epoch)+'.model') 
    print('epoch loss: ' + str(sum(epoch_loss)/len(epoch_loss)) + ', Val loss: ' + str(np.array(validation_loss).mean()) + ', Time: ' + str((end-start)))
