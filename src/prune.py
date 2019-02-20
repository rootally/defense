"""Pruning the weights away"""
from __future__ import print_fucntion
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torch.backends.cudnn as cudnn 

import os 
import argparse
from utils import progress_bar 
from train import create_model 
from load_data import get_data 
import numpy as np

parser = argparse.ArgumentParser(description='Pruning parameters')
parser.add_argument('--p_epochs', default=20, type=int, help='Number of training epochs')
parser.add_argument('--prune', '-p', default=0.9, dest='prune', help='fraction of parameters prunes')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate') 

prune_args = parser.parse_args()

#setup device 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0 
start_epoch = 0 
prune = float(prune_args.prune)   #percentage to be pruned away.

#load the data
trainloader, testloader = get_data()

#load the model with checkpoints 
model, criterion, optimizer = create_model()
checkpoint = torch.load('./checkpoint/ckpt.t7')
model.load_state_dict(checkpoint['net'])
epoch = checkpoint['epoch']
model.eval()

model = model.to(device) 
if device == 'cuda':
    model = torch.nn.DataParallel(model) 
    cudnn.benchmark = True

"""function to prune the weights"""
def prune_weights(torchweights):
    """Take the models weihts, and convert them into numpy array 
    then reshape the weight vector into a one dimensional vector 
    Arrange the weights according to their magnitude and set the 
    mask value for lower P weights to 0 and rest as 1 
    store this in the adressbook and maskbook 
    """
    weights=np.abs(torchweights.cpu().numpy());
    weightshape=weights.shape
    rankedweights=weights.reshape(weights.size).argsort()#.reshape(weightshape)
    
    num = weights.size
    prune_num = int(np.round(num*prune))
    count=0
    
    masks = np.zeros(rankedweights.shape)
    for n, rankedweight in enumerate(rankedweights):
        if rankedweight > prune_num:
            masks[n]=1
        else: count+=1
    print("total weights:", num)
    print("weights pruned:",count)
    
    masks=masks.reshape(weightshape)
    weights=masks*weights
    
    return torch.from_numpy(weights).cuda(), masks

"""The pruned weights location must be saved in addressbook and maskbook"""
addressbook = []
maskbook = []
for k,v in model.state_dict().items():
    if "conv2" in k:
        addressbook.append(k) 
        print("pruning layer ", k) 
        weights = v 
        weights, masks = prune_weights(weights) 
        maskbook.append(masks) 
        checkpoint['net'][k] = weights 

checkpoint['address'] = addressbook 
checkpoint['mask'] = maskbook

#training 
def prune_train(epoch, trainloader, net, criterion, optimizer):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets= inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs) 
        loss = criterion(outputs, targets) 
        loss.backward() 

        #mask pruned weights 
        checkpoint['net'] = model.state_dict()
        for address, mask in zip(addressbook, maskbook):
            checkpoint['net'][address] = torch.from_numpy(checkpoint['net'][address].cpu().numpy() * mask) 

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1) 
        total += targets.size(0) 
        correct += predicted.eq(targets).sum().item() 

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def prune_test(epoch, testloader, net, criterion, optimizer):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/prune_ckpt.t7')
        best_acc = acc


if __name__ == '__main__':
    for epoch in range (start_epoch, start_epoch+prune_args.p_epochs):
        prune_train(epoch, trainloader, model, criterion, optimizer) 
        prune_test(epoch, testloader, model, criterion, optimizer)
        with open("prune-results.txt", "a") as f:
            f.write(str(epoch) + "\n")
            f.write(str(best_acc)+"\n")