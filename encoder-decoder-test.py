import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob as glob
from torch.autograd import Variable
from reader import get_depth_data
from PIL import Image
import sys

Test = np.load('final_dataset_test.npy')

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


# Test
candidate_models = glob.glob('./checkpoints/EncoderDecoder/*.model')
test_model = EncoderDecoder()
test_model.load_state_dict(torch.load(candidate_models[-1])['state_dict'])

test_loss = []
for test in Test:
    test_x,test_y = [],[]
    dirs = glob.glob(test)
    g_data, data = get_depth_data(dirs)
    test_x, test_y = np.array(Image.open(data[0][0])), data[0][1]
    test_x, test_y = test_x.reshape(1,3,480,640), test_y.reshape(1,1,480,640)
    test_x = Variable(torch.Tensor(test_x).cuda(),requires_grad=False)
    test_pred = model.forward(test_x).detach().numpy()
    for box in g_data:
        info,dis = box
        info = [int(i) for i in info[:4]]
        # str(startX),str(startY),str(endX),str(endY)
        startX ,startY, endX, endY = info[0], info[1], info[2], info[3]
        pred_dis = test_pred[startY:endY,startX:endX].min()/1000.0
        dis_error = pred_dis - dis
        test_loss.append(dis_error)
    break
print('Test Loss:',np.mean(test_loss))