# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from utils import *
from dataset import *
from meter import *

import os
import numpy as np
import time
import cv2
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import pandas as pd
import torch.optim as optim
import random
import sys
sys.path.insert(0, 'optimizers')
from ralamb import Ralamb
from radam import RAdam
from ranger import Ranger
from lookahead import LookaheadAdam
from over9000 import Over9000

from tqdm import tqdm_notebook as tqdm

class Trainer(object):
    '''This class takes care of training and validation of our model'''
    def __init__(self,model, optim, loss, lr, bs, name, shape=512, crop_type=0):
        self.num_workers = 4
        self.batch_size = {"train": bs, "val": 1}
        self.accumulation_steps = bs // self.batch_size['train']
        self.lr = lr
        self.loss = loss
        self.optim = optim
        self.num_epochs = 0
        self.best_dice = 0.
        self.best_val_acc = 0
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.name = name
        self.do_cutmix = True
        self.loss_classification = torch.nn.CrossEntropyLoss()
        if self.loss == 'BCE':
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif self.loss == 'BCE+DICE':
            self.criterion = BCEDiceLoss(threshold=None)  #MODIFIED
        elif self.loss == 'TVERSKY':
            self.criterion = Tversky()
        elif self.loss == 'Dice' or self.loss == 'DICE':
            self.criterion = DiceLoss()
        elif self.loss == 'BCE+DICE+JACCARD':
            self.criterion = BCEDiceJaccardLoss(threshold=None)
        else:
            raise(Exception(f'{self.loss} is not recognized. Please provide a valid loss function.'))

        # Optimizers
        if self.optim == 'Over9000':
            self.optimizer = Over9000(self.net.parameters(),lr=self.lr)
        elif self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(),lr=self.lr)
        elif self.optim == 'RAdam':
            self.optimizer = Radam(self.net.parameters(),lr=self.lr)
        elif self.optim == 'Ralamb':
            self.optimizer = Ralamb(self.net.parameters(),lr=self.lr)
        elif self.optim == 'Ranger':
            self.optimizer = Ranger(self.net.parameters(),lr=self.lr)
        elif self.optim == 'LookaheadAdam':
            self.optimizer = LookaheadAdam(self.net.parameters(),lr=self.lr)
        else:
            raise(Exception(f'{self.optim} is not recognized. Please provide a valid optimizer function.'))
            
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, mode="min", patience=4, verbose=True, min_lr = 1e-5)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        
        self.dataloaders = {
            phase: provider(
                phase=phase,
                shape=shape,
                crop_type=crop_type,
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers if phase=='train' else 0,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.F2_scores = {phase: [] for phase in self.phases}
        self.acc_scores = {phase: [] for phase in self.phases}
    
    def change_loader(self, crop_type=0, shape=512):
        '''
        crop_type -- 0 (CropNonEmptyMaskIfExists)
                  -- 1 (RandomResizedCrop)
        shape     -- 512 (default)
        '''
        self.dataloaders = {
            phase: provider(
                phase=phase,
                shape=shape,
                crop_type=crop_type,
                batch_size=self.batch_size[phase],
                num_workers= 0,
            )
            for phase in self.phases
        }

    def freeze(self):
        for  name, param in self.net.encoder.named_parameters():
            if name.find('bn') != -1:
                param.requires_grad=True
            else:
                param.requires_grad=False
                
    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def load_model(self, name, path='models/'):
        state = torch.load(path+name, map_location=lambda storage, loc: storage)
        self.net.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        print("Loaded model with dice: ", state['best_dice'])
            
    def unfreeze(self):
        for param in self.net.parameters():
            param.requires_grad=True
     
    def bold(self, masks):
#         print(masks.type())
        masks = masks.numpy()
        for i in range(masks.shape[0]):
            mask = masks[i]
            kernel = np.ones((3,3),np.uint8)
#             print(mask.shape)
            mask = cv2.dilate(mask,kernel,iterations = 1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            masks[i] = mask
        return torch.Tensor(masks).type(torch.FloatTensor)
    
    def forward(self, images, targets, labels):
        images = images.to(self.device)
#         print(images.shape)
        masks = targets.to(self.device)
#         masks.resize_((masks.shape[0],1,256, 256))
        labels = labels.to(self.device)
        print(images.shape, masks.shape, labels.shape)
#         images = torch.cat((images, images, images), 1)
        outputs, preds = self.net(images.type(torch.cuda.FloatTensor))
        cls_loss = self.loss_classification(preds, labels)
        seg_loss = self.criterion(outputs, masks)
        return cls_loss, seg_loss, outputs, preds
    
    def cutmix(self,batch, alpha):
        data, targets, labels = batch
        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]
        shuffled_targets = targets[indices]
        lam = np.random.beta(alpha, alpha)
        image_h, image_w = data.shape[2:]
        cx = np.random.uniform(0, image_w)
        cy = np.random.uniform(0, image_h)
        w = image_w * np.sqrt(1 - lam)
        h = image_h * np.sqrt(1 - lam)
        x0 = int(np.round(max(cx - w / 2, 0)))
        x1 = int(np.round(min(cx + w / 2, image_w)))
        y0 = int(np.round(max(cy - h / 2, 0)))
        y1 = int(np.round(min(cy + h / 2, image_h)))
        data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
        targets[:, :, y0:y1, x0:x1] = shuffled_targets[:, :, y0:y1, x0:x1]
        return data, targets, labels

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(tk0):
            if phase == "train" and self.do_cutmix:
                images, targets = self.cutmix(batch, 0.5)
            else:
                images, targets, labels = batch
            if phase == 'train':
                targets = self.bold(targets)
            cls_loss, seg_loss, outputs, preds = self.forward(images, targets, labels)
            loss = (0.2*cls_loss + seg_loss) / self.accumulation_steps
            if phase == "train":
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            preds = preds.detach().cpu()
            labels = labels.detach().cpu()
            
#             if phase == 'train':
            meter.update(targets, outputs, labels,  preds)
#             else:
#                 meter.update(targets[:,:,:-pad_h,:-pad_w], outputs[:,:,:-pad_h,:-pad_w])
            tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou, f2, acc = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        self.F2_scores[phase].append(f2)
        self.acc_scores[phase].append(acc)
        torch.cuda.empty_cache()
        return epoch_loss, dice, acc

    def train_end(self):
        train_dice = self.dice_scores["train"]
        train_loss = self.losses["train"]
        train_f2 = self.F2_scores["train"]
        train_iou = self.iou_scores["train"]
        train_acc = self.acc_scores["train"]
        
        val_dice = self.dice_scores["val"]
        val_loss = self.losses["val"]
        val_f2 = self.F2_scores["val"]
        val_iou = self.iou_scores["val"]
        val_acc = self.acc_scores["val"]

        df_data=np.array([train_loss,train_dice,train_iou,train_f2, train_acc, val_loss,val_dice,val_iou,val_f2, val_acc]).T
        df = pd.DataFrame(df_data,columns = ['train_loss','train_dice','train_iou','train_f2', 'train_acc', 'val_loss','val_dice','val_iou','val_f2', 'val_acc'])
        df.to_csv('logs/'+self.name+'.csv')

    def fit(self, epochs):
        self.num_epochs+=epochs
        for epoch in range(self.num_epochs-epochs, self.num_epochs):
            self.net.train()
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_dice": self.best_dice,
                "best_acc": self.best_val_acc,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            self.net.eval()
            with torch.no_grad():
                val_loss, val_dice, val_acc = self.iterate(epoch, "val")
                self.scheduler.step(val_loss)
            if val_dice > self.best_dice:
                print("* New optimal found according to dice, saving state *")
                state["best_dice"] = self.best_dice = val_dice
                os.makedirs('models/', exist_ok=True)
                torch.save(state, 'models/'+self.name+'_best_dice.pth')
            if val_acc > self.best_val_acc:
                print("* New optimal found according to accuracy, saving state *")
                state["best_acc"] = self.best_val_acc = val_acc
                os.makedirs('models/', exist_ok=True)
                torch.save(state, 'models/'+self.name+'_best_acc.pth')
            print()
            self.train_end()