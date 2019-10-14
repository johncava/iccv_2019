import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from reader import *
import sys

path = './checkpoints/gcn/L9/'

Train = np.load('iccv_dataset_train.npy')
Val = np.load('iccv_dataset_val.npy')
Test = np.load('iccv_dataset_test.npy')

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
        self.w1 = Variable(torch.randn(8,8).cuda(),requires_grad=True)
        self.w2 = Variable(torch.randn(8,8).cuda(),requires_grad=True)
        self.w3 = Variable(torch.randn(8,8).cuda(),requires_grad=True)
        self.w4 = Variable(torch.randn(8,5).cuda(),requires_grad=True)
        self.w5 = Variable(torch.randn(5,5).cuda(),requires_grad=True)
        self.w6 = Variable(torch.randn(5,5).cuda(),requires_grad=True)
        self.w7 = Variable(torch.randn(5,3).cuda(),requires_grad=True)
        self.w8 = Variable(torch.randn(3,3).cuda(),requires_grad=True)
        self.w9 = Variable(torch.randn(3,1).cuda(),requires_grad=True)

    def forward(self,A,D,l):
        hidden_layer_1 = F.relu(D.mm(A).mm(l).mm(self.w1))
        hidden_layer_2 = F.relu(D.mm(A).mm(hidden_layer_1).mm(self.w2))
        hidden_layer_3 = F.relu(D.mm(A).mm(hidden_layer_2).mm(self.w3))
        hidden_layer_4 = F.relu(D.mm(A).mm(hidden_layer_3).mm(self.w4))
        hidden_layer_5 = F.relu(D.mm(A).mm(hidden_layer_4).mm(self.w5))
        hidden_layer_6 = F.relu(D.mm(A).mm(hidden_layer_5).mm(self.w6))
        hidden_layer_7 = F.relu(D.mm(A).mm(hidden_layer_6).mm(self.w7))
        hidden_layer_8 = F.relu(D.mm(A).mm(hidden_layer_7).mm(self.w8))
        y_pred = F.relu(D.mm(A).mm(hidden_layer_8).mm(self.w9))
        return torch.clamp(y_pred,max=200)

model = Scinfaxi()
learning_rate = 1e-3
optimizer = optim.Adam([model.w1,model.w2,model.w3,model.w4,model.w5,model.w6,model.w7,model.w8,model.w9],lr=learning_rate)
max_epochs = 15
loss_fn = nn.MSELoss()

import time
candidate_models = []
validation_losses = []
# Training and Validation
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
        A, D = create_graph(x)
        A = Variable(torch.Tensor(A),requires_grad=False).view(len(data),len(data)).cuda()
        D = Variable(torch.Tensor(D),requires_grad=False).view(len(data),len(data)).cuda()
        l = Variable(torch.Tensor(l),requires_grad=False).view(len(data),8).cuda()
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
        A, D = create_graph(val_x)
        A = Variable(torch.Tensor(A),requires_grad=False).view(len(data),len(data)).cuda()
        D = Variable(torch.Tensor(D),requires_grad=False).view(len(data),len(data)).cuda()
        l = Variable(torch.Tensor(val_l),requires_grad=False).view(len(data),8).cuda()
        val_pred = model.forward(A,D,l)    
        val_y = Variable(torch.Tensor(val_y).cuda(),requires_grad=False).view(len(data),1)
        val_loss = loss_fn(val_pred,val_y)
        validation_loss.append(val_loss.item())

    validation_losses.append(np.array(validation_loss).mean())

    end = time.time()
    model_path = path + 'gcn-9_layer-epoch_'+str(epoch)+'.model'
    torch.save(model.state_dict(), model_path) 
    candidate_models.append(model_path) 
    print('epoch loss: ' + str(np.array(epoch_loss).mean()) + ', Val loss: ' + str(np.array(validation_loss).mean()) + ', Time: ' + str((end-start))) 

    if len(validation_losses) > 1:
        check = (((validation_losses[-2] - validation_losses[-1])/(validation_losses[-2])) * 100)
        if check < 1.0 and check > 0:
            break
        if check < 0:
            candidate_models.pop()
            break
# Test
test_model = Scinfaxi().cuda()
test_model.load_state_dict(torch.load(candidate_models[-1]))
test_loss = []                                                               
for test in Test:                                                           
    test_x,test_l,test_y = [],[],[]
    dirs = glob.glob(test)
    data = get_data_bbox_graph(dirs)[0]
    for t in data:
        test_x.append(t[0])
        test_l.append(t[1])                                         
        test_y.append(t[2])                                                     
    A, D = create_graph(test_x)
    A = Variable(torch.Tensor(A),requires_grad=False).view(len(data),len(data)).cuda()
    D = Variable(torch.Tensor(D),requires_grad=False).view(len(data),len(data)).cuda()
    l = Variable(torch.Tensor(test_l),requires_grad=False).view(len(data),8).cuda()
    test_pred = test_model.forward(A,D,l).detach().cpu().numpy().squeeze().tolist()
    for i,j in zip(test_pred,test_y):
        test_loss.append(abs(i-j))
np.save(path+'test_loss.npy',test_loss)
print('Test Loss:',np.mean(test_loss))
