import cv2
import tifffile as tiff
from torch.utils.data import Dataset, DataLoader, sampler
import albumentations as aug
from albumentations import (HorizontalFlip, VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose,Cutout, GaussNoise, RandomRotate90, Transpose, RandomBrightnessContrast, RandomCrop)
from albumentations.pytorch import ToTensor
from albumentations.augmentations.transforms import CropNonEmptyMaskIfExists
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
            mask = cv2.imread('./CVPR_dataset/train/masks/'+self.df_train.Name[idx])
        else:
            img = cv2.imread('./CVPR_dataset/val/images/'+self.df_val.Name[idx])
            mask = cv2.imread('./CVPR_dataset/val/masks/'+self.df_val.Name[idx])
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
#         print(mask.shape)
        mask = mask[0].permute(2, 0, 1)
        return img, mask

    def __len__(self):
        if self.phase == 'train':
            return self.train_size
        else:
            return self.val_size

def get_transforms(phase, crop_type=0, size=256):
    list_transforms = []
    if phase == "train":
        list_transforms.extend([
             aug.Flip(),
             aug.RandomRotate90()
            ])
        
    list_transforms.extend(
        [
            ToTensor(),
        ]
    )
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