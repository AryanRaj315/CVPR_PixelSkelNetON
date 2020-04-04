import cv2
from torch.utils.data import Dataset, DataLoader, sampler
from tfs import Shift, Compose, HorizontalFlip, VerticalFlip, RotateAndCenterCrop, RandomRotate90, Recenter, ToTensor
import numpy as np
import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split

class CVPRDataset(Dataset):
    def __init__(self, phase, shape = 256, crop_type=0):
        self.transforms = get_transforms(phase, crop_type=crop_type, size=shape)
        self.phase = phase
        self.df_train = pd.read_csv("df_train.csv")
        self.df_val = pd.read_csv("df_val.csv")
        self.shape = shape
        self.train_size = len(self.df_train)
        self.val_size = len(self.df_val)

    def __getitem__(self, idx):
        if self.phase == 'train':
            img = cv2.imread('./CVPR_dataset/train/images/'+self.df_train.Name[idx])
            mask = cv2.imread('./CVPR_dataset/train/masks/'+self.df_train.Name[idx], 0)
            label = np.asarray(self.df_train.iloc[idx, 2:]).astype('int').argmax()
            
            try:
                shape = img.shape
            except:
                print("EXCEPTION RAISED! Path of Train Val Image", self.df.Name[idx])
            
        else:
            img = cv2.imread('./CVPR_dataset/val/images/'+self.df_val.Name[idx])
            mask = cv2.imread('./CVPR_dataset/val/masks/'+self.df_val.Name[idx], 0)
            label = np.asarray(self.df_val.iloc[idx, 2:]).argmax()
#             label = label.reshape(1,)
#             label = label.argmax()
            try:
                shape = img.shape
            except:
                print("EXCEPTION RAISED! Path of Image", self.path_val[idx])
#         print(img.shape)
#         img = img.reshape(256, 256, 1)
        augmented = self.transforms(image=img, mask=mask)
        img = augmented[0]
        mask = augmented[1]
#         print(img.shape, mask.shape)
#         mask = mask[0].permute(2, 0, 1)
        return img, mask, label

    def __len__(self):
        if self.phase == 'train':
            return self.train_size
        else:
            return self.val_size

def get_transforms(phase, crop_type=0, size=256):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                 Recenter(prob=0.7),
                 Shift(prob = 0.5),
                 VerticalFlip(prob=0.5),
                 RotateAndCenterCrop(prob=0.5, limit=90),
            ]
        )
    elif phase == "val":
        list_transforms.extend(
            [
                 Recenter(prob=1),
            ]
    )
    list_transforms.extend([ToTensor()])    
    list_trfms = Compose(list_transforms)
    return list_trfms

def provider(phase, shape = 256, crop_type = 0, batch_size=8, num_workers=4):
    '''Returns dataloader for the model training'''
    if phase == 'train':
        image_dataset = CVPRDataset(phase, shape=shape, crop_type=crop_type)
        pin = False 
    else:
        image_dataset = CVPRDataset(phase, shape=shape, crop_type=crop_type)
        pin = False
        
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=pin,
        shuffle=True,   
    )

    return dataloader