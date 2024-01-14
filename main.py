import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange

import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader,sampler,Dataset
import torchvision.transforms as T
import os
import numpy as np
import scipy.io
import random
from torchvision import models

from dataset import MRIDataset
from model import SwinTransformer3D_UNet

torch.cuda.set_device(1)

class ToTensor_3d(object):
    def __init__(self):
        self.transform_img = T.Compose([            
            T.RandomResizedCrop(40, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),                        
        ])
    def __call__(self,pic):
        img = torch.from_numpy(pic)        
        return img

class Charbonnier_loss(nn.Module):
    def __init__(self):
        super(Charbonnier_loss,self).__init__()
        self.eps = 1e-6
    
    def forward(self,X,Y):
        diff = torch.add(X,-Y)
        L2 = torch.sqrt(diff*diff + self.eps)
        loss = torch.mean(L2)
        return loss

def train(model, loss_fn, optimizer,dataloader, num_epochs = 1):
    model.train()
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))    
        loss_sum = 0
        count = 0
        for t, sample in enumerate(dataloader):
            x_var = Variable(sample['image'].float().cuda())                  
            y_var = Variable(sample['label'].float().cuda())#.long())            
            
            flow  = model(x_var)            
            loss_all = loss_fn(y_var,flow)            
            
            loss = loss_all
            loss_sum += loss
            count += 1
                       
            if (t + 1) % print_every == 0:                
                print('t = %d, loss = %.6f' % (t + 1,loss.cpu())) 
            
            optimizer.zero_grad()               
            loss.backward()
            optimizer.step()                            
        print('epoch = %d, train loss = %.6f' % (epoch+1, loss_sum.cpu() / count))

def test(model, loss_fn, loader):    
    model.eval() 
    with torch.no_grad():
        for t, sample in enumerate(loader):
            x_var = Variable(sample['image'].float().cuda())                  
            y_var = Variable(sample['label'].float().cuda())
            
            flow = model(x_var)
            
            loss = loss_fn(y_var,flow)            
            
            flow = flow.data.cpu().numpy()
            label = y_var.data.cpu().numpy()    
                        
    print('Test Loss:(%.6f)' % (loss))
    scipy.io.savemat('results.mat',{'flow':flow,'label':label})


if __name__== "__main__":
    ToTensor_3d = ToTensor_3d()
    
    image_dataset_train=MRIDataset(root_dir='training_dataset_folder',\
                                   labels=True,transform=ToTensor_3d,param = 0.5)
    
    image_dataloader_train = DataLoader(image_dataset_train, batch_size=15,
                            shuffle=True, num_workers=4)
    image_dataset_test=MRIDataset(root_dir='testing_dataset_folder',\
                                  labels=True,transform=ToTensor_3d,param = 0)
    
    image_dataloader_test = DataLoader(image_dataset_test, batch_size=106,
                            shuffle=False, num_workers=1)
    dtype = torch.cuda.FloatTensor # the GPU datatype
    # Constant to control how frequently we print train loss
    print_every = 10    

    model = SwinTransformer3D_UNet().cuda()
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = Charbonnier_loss().type(dtype).cuda()
    
    model.train()
    train(model,loss_fn,optimizer,image_dataloader_train,num_epochs=4)
    
    model.eval()
    test(model, loss_fn, image_dataloader_test)