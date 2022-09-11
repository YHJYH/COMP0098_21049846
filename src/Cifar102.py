# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 21:18:16 2022

@author: Yuheng

This file tests model performance on OOD cifar-10.2

https://github.com/modestyachts/cifar-10.2
"""
import numpy as np
import os
import argparse
import time
from PIL import Image
from typing import Any, Optional, Callable, Tuple

import torch
import torchvision
import torch.nn as nn

import model


class Cifar102TrainDataset(torch.utils.data.Dataset):
    
    def __init__(self, file_loc: str, transform: Optional[Callable] = None) -> None:
        self.ood_train_set = np.load(file_loc)
        self.transform = transform 
        
        self.data: Any = []
        self.targets = []
        
        for i in range(self.ood_train_set['images'].shape[0]):
            self.data.append(self.ood_train_set['images'][i])
            self.targets.append(self.ood_train_set['labels'][i])
          
        self.data = np.vstack(self.data).reshape(-1, 32, 32, 3)
        
        #self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        #self.data = self.data.transpose((0, 2, 3, 1)) ###
        
    def __len__(self):
        return len(self.ood_train_set['images'])
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, target = self.data[idx], self.targets[idx]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target
        

class Cifar102TestDataset(torch.utils.data.Dataset):
    
    def __init__(self, file_loc: str, transform: Optional[Callable] = None) -> None:
        self.ood_test_set = np.load(file_loc)
        self.transform = transform
        
        self.data: Any = []
        self.targets = []
        
        for i in range(self.ood_test_set['images'].shape[0]):
            self.data.append(self.ood_test_set['images'][i])
            self.targets.append(self.ood_test_set['labels'][i])
        
        self.data = np.vstack(self.data).reshape(-1, 32, 32, 3)
        
        #self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        #self.data = self.data.transpose((0, 2, 3, 1))
        
    def __len__(self):
        return len(self.ood_test_set['images'])
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, target = self.data[idx], self.targets[idx]
        
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target

'''
def test(args):
    
    print(args)
    
    model_file = 'train_model'
    os.makedirs('output/%s/' % args.data_type, exist_ok=True)
    #output_file = 'output/%s/feature_maps.txt' % args.data_type
    
    print('Loading dataset...')
    
    batch_size_test = 256
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    (0.4914, 0.4822, 0.4465), 
                                                    (0.2023, 0.1994, 0.2010))])
    
    if args.data_type == 'cifar10':
        # https://www.cs.toronto.edu/~kriz/cifar.html
        
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('/files/', train=False, download=True,
                                         transform=transform),
            batch_size=batch_size_test, shuffle=True)
    
    elif args.data_type == 'cifar102':
        test_loader = torch.utils.data.DataLoader(Cifar102TestDataset(file_loc='cifar102_test.npz', 
                                                                  transform=transform),
                                                  batch_size=batch_size_test, shuffle=True)
    
    num_classes = 10
    input_height = 32
    input_width = 32
    input_channel = 3
        
    critierion = nn.CrossEntropyLoss(size_average=False)
    
    print('Initializing model...')
    if args.model_type == 'vgg16':
        MODEL = model.VGG16(num_classes, input_height, input_width, input_channel)
    elif args.model_type == 'vgg16_N':
        MODEL = model.VGG16_N(num_classes, input_height, input_width, input_channel) 
    #MODEL.to(device)  
    
    print('Loading model...')
    best_model = torch.load(model_file)
    MODEL.load_state_dict(best_model['MODEL'])
    
    start = time.time()
    loss = 0
    acc = 0
    losses = []
    feature_maps = {}
    with torch.no_grad():
        for layer, (img, label) in enumerate(test_loader):
            #img, label = img.to(device), label.to(device)
            pred = MODEL(img)[1]
            feature = MODEL(img)[0]
            feature_maps[layer] = feature
            loss += critierion(pred, label).item()
            pred_label = pred.data.max(1, keepdim=True)[1]
            acc += pred_label.eq(label.data.view_as(pred_label)).sum()
            
        loss /= len(test_loader.dataset)
        losses.append(loss)
        print('\nTest average loss: %.4f, acc: %.4f' % (loss, acc/len(test_loader.dataset)))
        end = time.time()
        print('Test time: %.4f s' % (end-start))
        print('-'*10)
        print('Saving feature maps...')
        torch.save(feature_maps, 'output/%s/OOD_features.pt' % args.data_type)
        print('Feature maps saved!')
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-data_type', default='cifar102', type=str)
    parser.add_argument('-model_type', default='vgg16_N', type=str)
    
    args = parser.parse_args()
    test(args)

'''    
