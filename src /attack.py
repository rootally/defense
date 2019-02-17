import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import torch.backends.cudnn as cudnn 
from torch.autograd import Variable 
import torchvision
import torchvision.transforms as transforms 
import pickle as pkl 
import numpy as np 
import matplotlib.pyplot as plt 
from train import create_model
from load_data import get_data
from utils import progress_bar
import argparse 


parser = argparse.ArgumentParser(description='Adversarial attacks')
parser.add_argument('--attack', default='fgsm', type=str, help='Adversarial attack')
parser.add_argument('--epsilon', default=0.031, type=float, help='Epsilon value')

arg = parser.parse_args()

#set defualt device to gpu 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#load the saved model and define epsilon values 
epsilons = [0, .05, .1, .15, .2, .25, .3]
model, criterion, optimizer = create_model()
checkpoint = torch.load('./checkpoint/ckpt.t7')
model.load_state_dict(checkpoint['net'])
epoch = checkpoint['epoch']
optimizer = checkpoint['optimizer']
model.eval()

#data
trainloader, testloader = get_data()

def FGSM(test_loader,epsilon = 0.1,min_val = 0,max_val = 1):
    correct = 0                   # Fast gradient sign method
    adv_correct = 0
    misclassified = 0
    total = 0
    adv_noise =0 
    adverserial_images = []
    y_preds = []
    y_preds_adv = []
    test_images = []
    test_label = []

    for i, (images,labels) in enumerate(testloader):
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        images = Variable(images,requires_grad = True)
        labels = Variable(labels)
        
        outputs = model(images)
        loss =criterion(outputs,labels)

        model.zero_grad()
        if images.grad is not None:
            images.grad.data.fill_(0)
        loss.backward()
        
        grad = torch.sign(images.grad.data) # Take the sign of the gradient.
        images_adv = images.data + epsilon*grad     # x_adv = x + epsilon*grad
        
        adv_output = model(Variable(images_adv)) # output by the model after adding adverserial noise
        
        _,predicted = torch.max(outputs.data,1)      # Prediction on the clean image
        _,adv_predicted = torch.max(adv_output.data,1) # Prediction on the image after adding adverserial noise
        total += labels.size(0)
        adv_correct += (adv_predicted == labels).sum().item()
        misclassified += (predicted != adv_predicted).sum().item()
        
        y_preds_adv.extend(adv_predicted.cpu().data.numpy())
        test_images.extend(images.cpu().data.numpy())
        test_label.extend(labels.cpu().data.numpy())

    print('Accuracy with epsion ' + str(epsilon) + ' is   '+ str(100*adv_correct/total))

#fucntion to test without any epsilon 
def test(testloader):
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

if __name__ == "__main__":
    if (arg.attack == 'fgsm'):
        FGSM(testloader, arg.epsilon)

