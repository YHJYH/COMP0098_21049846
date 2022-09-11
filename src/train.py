# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 01:28:56 2022

@author: Yuheng
"""

import argparse
import os
#import gzip
import time

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import model
from Cifar102 import Cifar102TrainDataset, Cifar102TestDataset
from utils import weights_init

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(args):
    
    print(args)
    
    os.makedirs('model/%s/' % args.data_type, exist_ok=True)
    model_file = 'model/%s/train_model' % args.data_type
    
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
    
    if args. data_type == 'cifar102':
        
        batch_size_train = 64
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        dataset = Cifar102TrainDataset(file_loc='cifar102_train.npz', transform=transform)
        
        train_set, val_set = torch.utils.data.random_split(dataset, [8000, 2000])
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
    torch.manual_seed(args.seed)
    MODEL.apply(weights_init)
    MODEL.to(device)
    
    optimizer = torch.optim.SGD(MODEL.parameters(), lr=args.learning_rate, momentum=args.momentum)
    
    best_vloss = 100
    if os.path.exists(model_file):
        print('Loading existing model...')
        best_model = torch.load(model_file)
        MODEL.load_state_dict(best_model['MODEL'])
        best_vloss = best_model['dev_loss']
        #best_vloss.load_state_dict(best_model['dev_loss'])
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
            pred = MODEL(img)[1]
            loss = critierion(pred, label)
            loss.backward()
            optimizer.step()
            
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
    
    model_file = 'model/%s/train_model' % args.data_type
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
        torch.save(feature_maps, 'output/%s/features.pt' % args.data_type)
        print('Feature maps saved!')
        
            
    
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-mode', default='train', type=str) # train or test
    
    parser.add_argument('-data_type', default='cifar10', type=str) # mnist, cifar10, cifar102
    parser.add_argument('-model_type', default='vgg16', type=str) # options: lenet5_raw, lenet5_N, lenet300100_raw, vgg16
    parser.add_argument('-seed', default=13, type=int) # 13: fixed initialization; other number: random initialization
    parser.add_argument('-learning_rate', default=0.001, type=float) # lenet=0.01 ##vgg=0.001 (used same for vgg)
    parser.add_argument('-momentum', default=0.9, type=float) # lenet=0.5 ##vgg=0.9 (used same for vgg)
    
    parser.add_argument('-num_epoch', default=50, type=int) #default=50
    parser.add_argument('-patience', default=5, type=int) #same as num_epoch
    
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)