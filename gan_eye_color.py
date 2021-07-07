#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.autograd import Variable

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, sampler
from torchvision.datasets import MNIST
import torchvision.datasets as dset
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from torchsummary import summary


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # 设置画图的尺寸
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# In[3]:


path='eyes'
bach=16


# In[4]:


data=dset.ImageFolder(root=path,transform=transforms.Compose([transforms.Resize([28, 28]),
                                                              transforms.ToTensor(),
                                                        transforms.Normalize(
                                                            mean=(0.5,0.5,0.5),
                                                            std=(0.5,0.5,0.5))
                                                    ]))


# In[5]:


data_train=DataLoader(dataset=data, batch_size=bach, shuffle=4,drop_last =True)


# In[6]:


imgs,_=next(iter(data_train))


# In[7]:


class DNET(nn.Module):
    def __init__(self):
        super(DNET,self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(3,32,5,1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,5,1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2,2))
        self.fc=nn.Sequential(
        nn.Linear(1024,1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024,1))
    def forward(self,x):
        x=self.conv(x)
        x=x.view(x.shape[0],-1)
        x=self.fc(x)
        return x


# In[8]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[9]:


from torchsummary import summary
model =DNET().to(device)
summary(model,input_size=(3,28,28))


# In[11]:


class GNET(nn.Module):
    def __init__(self,noise_dim=96):
        super(GNET,self).__init__()
        self.fc=nn.Sequential(
            nn.Linear(noise_dim,1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,7*7*128),
            nn.ReLU(True),
            nn.BatchNorm1d(7*7*128))
        self.conv=nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64,3,4,2,padding=1),
            
            nn.Tanh())
    def forward(self,x):
        x=self.fc(x)
        x=x.view(x.shape[0],128,7,7)
        x=self.conv(x)
        return x


# In[12]:


model


# In[13]:


bce_loss=nn.BCEWithLogitsLoss()

def D_loss(real_labels,fake_labels):
    size=real_labels.shape[0]
    true_label=torch.ones(size,1).float()
    false_label=torch.zeros(size,1).float()
    loss=bce_loss(real_labels,true_label)+bce_loss(fake_labels,false_label)
    return loss

def G_loss(fake_labels):
    size=fake_labels.shape[0]
    false_label=torch.ones(size,1).float()
    loss=bce_loss(fake_labels,false_label)
    return loss


# In[14]:


def get_optimizer(net):
    return  torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.5, 0.999))


# In[15]:


def imshow(img):
    #img = (img+0.5)/2 # unnormalize
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# In[16]:


def train_gan(D_net, G_net, D_optimizer, G_optimizer, D_loss, G_loss, show_every=250, 
                noise_size=96, num_epochs=10):
    iter_count=0
    for epoch in range(num_epochs):
        for x,_ in data_train:
            batch_size=x.shape[0]
            
            # 1 判别器训练
            # 真数据数据预测
            # 假数据预测
            # 判别器损失计算
            # 判别器反向传播
            
            # 2 生成器训练
            ## 要不要再次生成假数据？如果不重新生成，因为判别器已进化，第一阶段假数据已可识别
            # 判别器预测
            # 生成器损失计算
            # 生成器反向传播,第一个反向传播需添加参数 loss.backward(retain_graph=True)
        
            #3 可视化
            
            # 判别器训练
            logits_real=D_net(x)
            sample_noise=(torch.rand(batch_size,noise_size)-0.5)/0.5
            fake_images=G_net(sample_noise)
            logits_fake=D_net(fake_images)
            d_loss=D_loss(logits_real,logits_fake)
            D_optimizer.zero_grad()
            d_loss.backward(retain_graph=True)
            D_optimizer.step()
            
            # 生成器优化
            logits_fake=D_net(fake_images)
            
            g_loss=G_loss(logits_fake)
            G_optimizer.zero_grad()
            g_loss.backward()
            G_optimizer.step()
            
            if (iter_count % show_every == 0):
                imshow(torchvision.utils.make_grid(x,normalize=True))
                imshow(torchvision.utils.make_grid(fake_images,normalize=True))
            iter_count+=1



D_DC = DNET()
G_DC = GNET()

D_DC_optim = get_optimizer(D_DC)
G_DC_optim = get_optimizer(G_DC)

train_gan(D_DC, G_DC, D_DC_optim, G_DC_optim, D_loss, G_loss, num_epochs=100)




