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
# net.eval()

# transfer the model to GPU
if torch.cuda.is_available():
    net = net.cuda()


num_params = np.sum([p.nelement() for p in net.parameters()])
print(num_params, ' parameters')


Label_map={'african_hunting_dog': 0, 'ant': 1, 'ashcan': 2, 'black_footed_ferret': 3, 'bookshop': 4, 'carousel': 5,
           'catamaran': 6, 'cocktail_shaker': 7, 'combination_lock': 8, 'consomme': 9, 'coral_reef': 10, 
           'dalmatian': 11, 'dishrag': 12, 'fire_screen': 13, 'goose': 14, 'green_mamba': 15, 
           'king_crab': 16, 'ladybug': 17, 'lion': 18, 'lipstick': 19, 'miniature_poodle': 20,
           'orange': 21, 'organ': 22, 'parallel_bars': 23, 'photocopier': 24, 'rhinoceros_beetle': 25,
           'slot': 26, 'snorkel': 27, 'spider_web': 28, 'toucan': 29, 'triceratops': 30, 'unicycle': 31, 'vase': 32}

labels_list=list(Label_map)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
    
def classwise_test(testloader, model):
########################################################################
# class-wise accuracy

    classes, _ = find_classes("../5/val")
    n_class = len(classes) # number of classes

    class_correct = list(0. for i in range(n_class))
    class_total = list(0. for i in range(n_class))
    with torch.no_grad():
        for data in tqdm(testloader):
            images, labels = data
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()        
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(n_class):
        print('Accuracy of %10s : %2f %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
        
val_set = torchvision.datasets.ImageFolder(root="../5/val",transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1,shuffle=False, num_workers=2)


if torch.cuda.is_available():
    net = net.cuda()

os.makedirs("./filters",exist_ok = True) 
activation = {}
def get_activation(name):
    def hook_fn(model, inp, out):
        activation[name] = out.detach()
    return hook_fn

net.conv1.register_forward_hook(get_activation('conv1'))
net.conv2.register_forward_hook(get_activation('conv2'))
net.conv3.register_forward_hook(get_activation('conv3'))
net.conv4.register_forward_hook(get_activation('conv4'))

def img_show(img, ax, title=None):
     
    npimg = img.cpu().numpy()
    ax.axis("off")
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    if title!=None:
        ax.set_title(title)
        
        
def get_activations(model,loader,filter_num,layer):
    L=None
    for data in tqdm(loader):
        images, labels = data
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()        
        output = net(images)
        act=activation[layer][:,filter_num]
        if(L is None):
            L=act
        else:
            L=torch.cat((L,act),dim=0)
    return L

def get_plots(Layers,Filter_map,model,loader):
    for l in Layers:
        for f in Filter_map[l]:
            A=get_activations(model,loader,f,l)
            act_idx = torch.argsort(torch.Tensor([A[i].norm() for i in range(A.shape[0])]),descending=False)
            fig,axes = plt.subplots(nrows=2, ncols=5, figsize=(14,5))
            for i in range(5):
                img, label = val_loader.dataset[act_idx[i]]
                img_show((A[act_idx[i]])[None,:,:],axes[0,i])
                img_show(img,axes[1,i],labels_list[label])
            plot_name = l + '_filter' + str(f)
            plt.savefig('./filters/Filter_'+plot_name+'.png')

    
Layers=['conv1','conv2','conv3','conv4']
Filter_map={}
Filter_map['conv1']=[44,45]
Filter_map['conv2']=[37,117]
Filter_map['conv3']=[209,254]
Filter_map['conv4']=[62,227]
get_plots(Layers,Filter_map,net,val_loader)
