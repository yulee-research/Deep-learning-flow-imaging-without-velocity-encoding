import os
import numpy as np
import scipy.io
import random
from torch.utils.data import DataLoader,sampler,Dataset



def random_hflip(raw_data, mapping, mask):
    raw_data = np.flip(raw_data,axis=0).copy()    
    mapping = np.flip(mapping,axis=0).copy()    
    mask = np.flip(mask,axis=0).copy()    
    return raw_data, mapping, mask

def random_vflip(raw_data, mapping, mask):
    raw_data = np.flip(raw_data,axis=1).copy()
    mapping = np.flip(mapping,axis=1).copy()   
    mask = np.flip(mask,axis=1).copy()   
    return raw_data, mapping, mask

def random_rot(image,label, mask):
    times = np.random.randint(1,4)
    image = np.rot90(image,times,(0,1)).copy()
    label = np.rot90(label,times).copy()
    mask = np.rot90(mask,times).copy()
    return image,label, mask

class MRIDataset(Dataset):    

    def __init__(self,  root_dir,labels=True, transform=None, param = 0):
        """
        Args:
            root_dir (string): Directory with all the images.
            labels(list): labels if images.
            transform (callable, optional): Optional transform to be applied on a sample.
            param (int): Used for data augmentation.
        """
        self.root_dir = root_dir
        self.transform = transform    
        self.length=len(os.listdir(self.root_dir))
        self.labels=labels
        self.param = param
    def __len__(self):
        return self.length
    
    def transform(raw_data,recon_image,label):
        raw_data,recon_image,label = radom_hflip(raw_data,recon_image,label)
        return raw_data,recon_image,label
    
    def __getitem__(self, idx):
        datapath = str(idx+1)+'.mat'
        image_path = os.path.join(self.root_dir,datapath)
        data_mat = scipy.io.loadmat(image_path)
        image = data_mat['Reg']        
        label = data_mat['Vec']
        label_mask = data_mat['Mask']
                
        image = np.concatenate((image,image[:,:,28:30]),2)
        label = np.concatenate((label,label[:,:,28:30]),2)
        label_mask = np.concatenate((label_mask,label_mask[:,:,28:30]),2)
        
        if random.random() < self.param:            
            image, label,label_mask = random_hflip(image,label,label_mask)            
        if random.random() < self.param:            
            image, label,label_mask = random_vflip(image,label,label_mask)            
        if random.random() < self.param:
            image, label, label_mask = random_rot(image,label,label_mask)
        
        if self.transform:            
            image = image.transpose([2,0,1])
            label = label.transpose([2,0,1])
            label_mask = label_mask.transpose([2,0,1])
                        
            image = self.transform(image)
            label = self.transform(label)
            label_mask = self.transform(label_mask)
            
            # Take odd or even frames
            
            if random.random() < self.param:
                image = image[0:32:2,:,:]
                label = label[0:32:2,:,:]
                label_mask = label_mask[0:32:2,:,:]
            else:
                image = image[1:32:2,:,:]
                label = label[1:32:2,:,:]
                label_mask = label_mask[1:32:2,:,:]
                        
            image = image.unsqueeze(0)
            label = label.unsqueeze(0)
            label_mask = label_mask.unsqueeze(0)
                        
        if self.labels:
            sample={'image':image,'img_path':image_path,'label':label,'label_mask':label_mask}
        else:
            sample={'image':image,'img_path':image_path}
        return sample  