# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 01:13:52 2022

@author: Yuheng
"""
import argparse
import os
#import gzip
import numpy as np
import time

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import model
from utils import weights_init
from Cifar102 import Cifar102TrainDataset, Cifar102TestDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    
    print(args)
    
    os.makedirs('model/%s/' % args.data_type, exist_ok=True)
    model_file = 'model/%s/CL_train_model' % args.data_type
    base_model_file = 'model/%s/train_model' % (args.data_type)
    
    print('Loading dataset...')
    if args.data_type == 'mnist':
        batch_size_train = 64
        
        dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ]))
        
        train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
        train_loader = torch.utils.data.DataLoader(train_set,
            batch_size=batch_size_train, shuffle=True)
        
        dev_loader = torch.utils.data.DataLoader(val_set,
                                                 batch_size=batch_size_train, shuffle=True)
        
        num_classes = 10
        input_height = 28
        input_width = 28
        input_channel = 1
        
        critierion = F.nll_loss()
        
    if args. data_type == 'cifar10':
        # https://www.cs.toronto.edu/~kriz/cifar.html
        batch_size_train = 64
        
        dataset = torchvision.datasets.CIFAR10('/files/', train=True, download=True, 
                                               transform = torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(
                                                       (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # https://medium.com/@buiminhhien2k/solving-cifar10-dataset-with-vgg16-pre-trained-architect-using-pytorch-validation-accuracy-over-3f9596942861
                                                   ]))
        
        train_set, val_set = torch.utils.data.random_split(dataset, [40000, 10000])
        train_loader = torch.utils.data.DataLoader(train_set, 
                                                   batch_size = batch_size_train, shuffle=True)
        
        dev_loader = torch.utils.data.DataLoader(val_set, 
                                                 batch_size=batch_size_train, shuffle=True)
        
        num_classes = 10
        input_height = 32
        input_width = 32
        input_channel = 3
        
        critierion = nn.CrossEntropyLoss()
    
    print('Initializing model...')
    if args.model_type == 'lenet5_raw':
        MODEL = model.LeNet_5_raw(num_classes, input_height, input_width, input_channel)
    elif args.model_type == 'lenet5_N':
        MODEL = model.LeNet_5_N(num_classes, input_height, input_width, input_channel)
    elif args.model_type == 'lenet300100_raw':
        MODEL = model.LeNet_300_100_raw(num_classes, input_height, input_width, input_channel)
    elif args.model_type == 'vgg16':
        MODEL = model.VGG16(num_classes, input_height, input_width, input_channel)
    elif args.model_type == 'vgg16_N':
        MODEL = model.VGG16_N(num_classes, input_height, input_width, input_channel)
    
    # Create prune mask to initial weights
    print('Loading %s model state_dict...' % args.model_type)
    if os.path.exists(base_model_file):
        print('Loading base model file...')
        base_MODEL = torch.load(base_model_file)
        MODEL.load_state_dict(base_MODEL['MODEL'])
        
    print('Creating initial mask...')
    mask_init = make_mask(MODEL)
    
    # Pruning mask by percentage
    mask_on, num_params = prune_by_percentile(MODEL, mask_init, args.prune_percentage)
    print('Number of parameters remaining: %d' % num_params)
    
    # Create pruned model   
    torch.manual_seed(args.seed)
    MODEL.apply(weights_init)
    
    # Apply mask to the model
    step = 0
    for name, param in MODEL.named_parameters():
        if 'weight' in name:
            param.data = torch.from_numpy(param.data.cpu().numpy() * mask_on[step])
            step += 1
    
    MODEL.to(device)
    
    optimizer = torch.optim.SGD(MODEL.parameters(), lr=args.learning_rate, momentum=args.momentum)
    
    best_vloss = 100
    if os.path.exists(model_file):
        print('Loading existing model...')
        best_model = torch.load(model_file)
        MODEL.load_state_dict(best_model['MODEL'])
        best_vloss = best_model['dev_loss']
        optimizer.load_state_dict(best_model['optimizer'])
        
    print('Training begin...')
    patience = args.patience
    trigger_times = 0
    
    for epoch in range(args.num_epoch):
        
        #train_losses = []
        #dev_losses = []
        running_loss = 0.
        last_loss = 0.
        
        MODEL.train(True)
        print('----------------------------------------------')
        for batch_index, (img, label) in enumerate(train_loader):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            
            apply_mask(MODEL, mask_on)
            
            pred = MODEL(img)[1]
            loss = critierion(pred, label)
            loss.backward()
            optimizer.step()
            
            # applied this since 3_2
            apply_mask(MODEL, mask_on)
            
            running_loss += loss.item()
            if batch_index % 100 == 0:
                last_loss = running_loss / 100. # loss per batch
                print('Epoch: %d, Batch: %d. (%.0f %%)' % (epoch+1, batch_index, 100.*batch_index/len(train_loader)))
                print('Train loss: %.6f' % last_loss)
                #train_losses.append(loss.item())
                torch.save({
                    'MODEL': MODEL.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, model_file)
                running_loss = 0.
                               
        print('----------------------------------------------')
        MODEL.train(False)
        running_vloss = 0.0
        for batch_index, (img, label) in enumerate(dev_loader):
            
            img, label = img.to(device), label.to(device)
            #MODEL.eval()
            
            pred = MODEL(img)[1]
            running_vloss += critierion(pred, label).item()
            
            #if batch_index % 64 == 0:
                #print('Epoch: %d, Batch: %d. (%.0f %%)' % (epoch+1, batch_index, 100*batch_index/len(dev_loader)))
                #print('Dev loss: %.6f' % loss.item())
                #dev_losses.append(loss.item())
        
        avg_vloss = running_vloss / (batch_index+1)
        print('Train loss: %.6f, Valid loss: %.6f' % (last_loss, avg_vloss))
        if avg_vloss <= best_vloss:
            trigger_times = 0
            print('Updating model file...')
            best_vloss = avg_vloss
            torch.save({
                'MODEL': MODEL.state_dict(),
                'optimizer': optimizer.state_dict(),
                'dev_loss': best_vloss
                }, model_file)
            cur_epoch = epoch
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print('Early stopping at: %d' % (cur_epoch+1))    
                break
        
        print('Early stopping at: %d' % (cur_epoch+1)) 
                
    print('----------------------------------------------')
    

def test(args):
    
    print(args)
    torch.cuda.empty_cache()
    # torch.cuda.memory_summary(device=None, abbreviated=False)
    
    model_file = 'model/%s/CL_train_model' % args.data_type
    os.makedirs('output/%s/' % args.data_type, exist_ok=True)
    #output_file = 'output/%s/feature_maps.txt' % args.data_type
    
    print('Loading dataset...')
    if args.data_type == 'mnist':
        batch_size_test = 1000
        
        test_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST('/files/', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))
                                     ])),
          batch_size=batch_size_test, shuffle=True) 
        
        num_classes = 10
        input_height = 28
        input_width = 28
        input_channel = 1
        
        critierion = F.nll_loss(size_average=False)
        
    if args.data_type == 'cifar10':
        # https://www.cs.toronto.edu/~kriz/cifar.html
        batch_size_test = 256
        
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10('/files/', train=False, download=True,
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                 (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                             ])),
            batch_size=batch_size_test, shuffle=True)
        
        num_classes = 10
        input_height = 32
        input_width = 32
        input_channel = 3
        
        critierion = nn.CrossEntropyLoss(size_average=False)
    
    elif args.data_type == 'cifar102':
        batch_size_test = 256
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize(
                                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                                                        )])
        test_loader = torch.utils.data.DataLoader(Cifar102TestDataset(file_loc='cifar102_test.npz', 
                                                                  transform=transform),
                                                  batch_size=batch_size_test, shuffle=True)
        num_classes = 10
        input_height = 32
        input_width = 32
        input_channel = 3
            
        critierion = nn.CrossEntropyLoss(size_average=False)
    
    print('Initializing model...')
    if args.model_type == 'lenet5_raw':
        MODEL = model.LeNet_5_raw(num_classes, input_height, input_width, input_channel)
    elif args.model_type == 'lenet5_N':
        MODEL = model.LeNet_5_N(num_classes, input_height, input_width, input_channel)
    elif args.model_type == 'lenet300100_raw':
        MODEL = model.LeNet_300_100_raw(num_classes, input_height, input_width, input_channel)
    elif args.model_type == 'vgg16':
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
        torch.save(feature_maps, 'output/%s/CL_features.pt' % args.data_type)
        print('Feature maps saved!')
        
        

def make_mask(model):
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            step += 1
    mask = [None] * step
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            mask[step] = np.ones_like(tensor)
            step += 1
    return mask

def prune_by_percentile(model, mask, percent, reinit=False):
    
    step = 0
    #num_params = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu().numpy()
            #alive = tensor[np.nonzero(tensor)]
            percentile_val = np.percentile(abs(tensor), percent)
            
            #weight_device = param.device
            layer_mask = np.where(abs(tensor) < percentile_val, 0, mask[step])
            
            #param.data = torch.from_numpy(tensor * layer_mask).to(weight_device)
            mask[step] = layer_mask
            step += 1
            #num_params += np.sum(layer_mask)
            
    num_params = sum([mask[i].sum() for i in range(len(mask))])
            
    return mask, num_params

def apply_mask(model, mask):
    step = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_device = param.device
            param.data = torch.from_numpy(param.data.cpu().numpy() * mask[step]).to(weight_device)
            step += 1
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-mode', default='train', type=str)
    parser.add_argument('-data_type', default='cifar10', type=str)
    parser.add_argument('-model_type', default='vgg16', type=str)
    #parser.add_argument('-model_code', default='2_1', type=str)
    parser.add_argument('-seed', default=13, type=int)
    
    parser.add_argument('-learning_rate', default=0.001, type=float) 
    parser.add_argument('-momentum', default=0.9, type=float)
    
    #parser.add_argument('-prune_iterations', default=6, type=int)
    parser.add_argument('-prune_percentage', default=70, type=int) # 0 for minimum (no prune), 50 for median (half pruned), 100 for maximum (all prune)
    #parser.add_argument('-prune_type', default='lt', type=str)
    parser.add_argument('-num_epoch', default=50, type=int)
    parser.add_argument('-patience', default=5, type=int)
    
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    
