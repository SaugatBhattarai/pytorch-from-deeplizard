import torch 
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

#Converting into pytorch class CNN
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features = 12 * 4* 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10 )
    
    def forward(self, t):
        #(1) input layer
        t = t
        
        #(2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        #(3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        #(4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        print('... t ...', t)
        t = self.fc1(t)
        t = F.relu(t)
        
        #(5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)
        
        #(6) output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1)
        
        return t

# turn off gradient tracking features
# torch.set_grad_enabled(False)


network = Network()

# data_loader = torch.utils.data.DataLoader(
#     train_set,
#     batch_size=10
# )

train_set = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

sample =  next(iter(train_set))
image,label = sample

output = network(image.unsqueeze(0))
print(output)

#CNN output size formula (square)
#- suppose we have  n * n input.
#- suppose we have f * f filter.
#- suppose we have a padding of p and a stride of s.
# 
# The output size O is given by this formula:
# O = (n - f + 2p)/s + 1 