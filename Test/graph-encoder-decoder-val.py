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
Test = np.load('final_dataset_test.npy')

class GraphEncoderDecoder(nn.Module):

    def __init__(self):
        super(GraphEncoderDecoder, self).__init__()
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

        self.a = nn.Linear(2*12,1)
        self.W = nn.Linear(12,12)
        self.P = nn.Linear(12,100)
        self.hidden = nn.Linear(2100,2000)

    def forward(self,x,g):
        g = [torch.Tensor([gi]).cuda() for gi in g]
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
max_epochs = 10

import time
candidate_models = []
validation_losses = []
# Training and Validation
for epoch in range(max_epochs):
    epoch_loss = []
    start = time.time()
    for train in Train:
        dirs = glob.glob(train)
        gdata,data = get_depth_data(dirs)
        gd = [g[0] for g in gdata]
        x,y = np.array(Image.open(data[0][0])),data[0][1]
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
        val_x, val_y = val_x.reshape(1,3,480,640), val_y.reshape(1,1,480,640)
        val_x = Variable(torch.Tensor(val_x).cuda(),requires_grad=False)
        val_pred = model.forward(val_x,gd)
        val_y = Variable(torch.Tensor(val_y).cuda(),requires_grad=False)
        val_loss = loss_fn(val_pred,val_y)
        validation_loss.append(val_loss.item())
        validation_losses.append(val_loss.item())

    end = time.time()
    model_path = path + fold + 'GraphEncoderDecoder-3_layer-epoch_'+str(epoch)+'.model'
    torch.save(model.state_dict(), model_path) 
    candidate_models.append(model_path) 
    print('epoch loss: ' + str(np.array(epoch_loss).mean()) + ', Val loss: ' + str(np.array(validation_loss).mean()) + ', Time: ' + str((end-start)))

    if len(validation_losses) > 1:
        check = (((validation_losses[-2] - validation_losses[-1])/(validation_losses[-2])) * 100)
        if check < 1.0 and check > 0:
            break
        if check < 0:
            candidate_models.pop()

# Test
test_model = GraphEncoderDecoder().cuda()
test_model.load_state_dict(torch.load(candidate_models[-1]))
test_loss = []
for test in Test:
    test_x,test_y = [],[]
    dirs = glob.glob(test)
    g_data, data = get_depth_data(dirs)
    gd = [g[0] for g in g_data]
    test_x, test_y = np.array(Image.open(data[0][0])), data[0][1]
    test_x, test_y = test_x.reshape(1,3,480,640), test_y.reshape(1,1,480,640)
    test_x = Variable(torch.Tensor(test_x).cuda(),requires_grad=False)
    test_pred = test_model.forward(test_x,gd).detach().cpu().numpy()
    test_pred = test_pred.reshape(480,640)
    for box in g_data:
        info,dis = box
        info = [int(i) for i in info[:4]]
        # str(startX),str(startY),str(endX),str(endY)
        startX ,startY, endX, endY = info[0], info[1], info[2], info[3]
        pred_dis = test_pred[startY:endY,startX:endX]/1000.0
        pred_dis = pred_dis[pred_dis > 0]
        dis_error = abs(pred_dis.mean() - dis)
        test_loss.append(dis_error)
print('Test Loss:',np.mean(test_loss))