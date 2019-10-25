import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from get_data import get_depth

dataset = get_depth()

class EncoderDecoder(nn.Module):

    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.cnn1 = nn.Conv2d(3,10)
        self.cnn2 = nn.Conv2d(10,25)
        self.cnn3 = nn.Conv2d(25,50)

        self.decnn1 = nn.ConvTranspose2d(50,25)
        self.decnn2 = nn.ConvTranspose2d(25,10)
        self.decnn3 = nn.ConvTranspose2d(10,1)

    def forward(self,x):
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))

        x = F.relu(self.decnn1(x))
        x = F.relu(self.decnn2(x))
        x = F.relu(self.decnn3(x))

model = EncoderDecoder()
learning_rate = 1e-3
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
max_epochs = 1

for epoch in range(max_epoch):
    for data in dataset:
        x,y = data
        x = Variable(torch.Tensor(x), requires_grad=True)
        y = Variable(torch.Tensor(y), requires_grad=False)
        pred = model(x)
        loss = loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break