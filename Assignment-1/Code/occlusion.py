import torch, os
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sb
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim

torch.manual_seed(0)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5, 0.5, 0.5])])


########################################################################
# Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).



# <<<<<<<<<<<<<<<<<<<<< EDIT THE MODEL DEFINITION >>>>>>>>>>>>>>>>>>>>>>>>>>
# Try experimenting by changing the following:
# 1. number of feature maps in conv layer
# 2. Number of conv layers
# 3. Kernel size
# etc etc.,

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 =  nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(kernel_size=4, stride=2)
        self.fc1 = nn.Linear(in_features=256, out_features=33) # 5 is the number of classes here (for batch 3,4,5 out_features is 33)

    def forward(self, x): 

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = F.relu(self.conv4(x))      
        x = self.pool(x)
        
        x = F.avg_pool2d(x, kernel_size=x.shape[2:])
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        
        return x  

################### DO NOT EDIT THE BELOW CODE!!! #######################

net=Net()
net.load_state_dict(torch.load("../models/bestmodel.pth"))
net.eval()

# transfer the model to GPU
if torch.cuda.is_available():
    net = net.cuda()


num_params = np.sum([p.nelement() for p in net.parameters()])
print(num_params, ' parameters')

occlusion_set = torchvision.datasets.ImageFolder(root="../occlusion",transform=transform)
occlusion_loader = torch.utils.data.DataLoader(occlusion_set, batch_size=1,shuffle=True, num_workers=2)

Label_map=
Temp_map=occlusion_set.class_to_idx
labels_list=list(Label_map)
A=[0]*10
for e in Temp_map:
    A[Temp_map[e]]=Label_map[e]

    

if torch.cuda.is_available():
    net = net.cuda()

os.makedirs('./plots', exist_ok=True)
n=7
for (image,label) in tqdm(occlusion_loader):
    if torch.cuda.is_available():
        image= image.cuda()
        label=label.cuda()
        width=image.shape[-2]
        height=image.shape[-1]
        w_n=int(width/n)
        h_n=int(height/n)
        confidence_map=torch.zeros([h_n,w_n])
        for i in range(0,h_n):
            for j in range(0,w_n):
                h_start=i*n
                w_start=j*n
                h_end=(i+1)*n
                w_end=(j+1)*n
                new_image=image.clone().detach()
                new_image[:,:,w_start:w_end,h_start:h_end]=0.5 #grayscale
                new_output=F.softmax(net(new_image.cuda()),dim=1)
                prob=new_output.tolist()[0][A[label]]
                confidence_map[i,j]=prob
        ax=sb.heatmap(confidence_map, xticklabels=False, yticklabels=False)
        plt.savefig('./plots/'+labels_list[A[label]]+'.png')
        plt.close()

