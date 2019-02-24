"""Targeted dropout implementation"""
from __future__ import print_function 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import torch.backends.cudnn as cudnn 
from torch.autograd import Variable 
import numpy as np 
from train import create_model 
from scipy.stats import truncnorm

#Make sure cuda is used 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def t_dropout(weights, p, d):
    """function to implement T-D
    args - input (tensor) 
    input containing output of some layer in the neural network 
    p - percentage of weights to be considered 
    d - dropout probability 

    returns - tensor 
    output shape = same as input shape
    take the input vector, then rank the weights based on their magnitudes
    take the bottom p fraction of the weights of each cloumn(neuron) and masks them 
    Then creates a random dropout mask for entire column and does logical AND between them  
    with dropout probability d.
    """
    mu, sigma = 0, 0.1  #mean and standard deviation of normal distribution
    num_dropped = (int) (p* weights.shape[0])
    
    size = weights.size()                           #return a tuple
    output_channels = weights.shape[0]
    weights = torch.reshape(weights, (-1, output_channels))     #reshape weight matrix into (-1, number of output channels)
    
    mask = torch.ones_like(weights, deivce = device, requires_grad = False)  
    dropout_mask = torch.empty_like(weights, device = device, requires_grad = False).uniform(0, 1)
    ones_matrix = torch.ones_like(weights, device = device, requires_grad=False) 

    sorted_weights, indices = torch.sort(weights, dim =1, descending=True)  #sort the weights along the column  

    for i in range(weights.shape[0]):         #loop through each row 
        mask[i, :num_dropped] = 0            #make the mask value of required values 0
    
    dropout_mask = torch.bernoulli(dropout_mask) #generated dropout mask from bernoulli distribution

    and_mask = mask*dropout_mask      #logical AND targeted mask and dropout mask
    final_mask = ones_matrix - and_mask                 # (1 - p*d) 
    masked_weights = final_mask * sorted_weights
    
    masked_weights = torch.reshape(masked_weights, size)

    return masked_weights
        
        
        

    



