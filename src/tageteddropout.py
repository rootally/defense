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

def get_truncated_normal(mean=0, sd=1, low=0, upp=1):   #truncated normal distribution 
    return truncnorm(                                   #upper bound of the random variable 
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def t_dropout(input, p, d):
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
    weights = np.abs(input.cpu().numpy())   #takes the inputs and convert into numpy array 
    num_dropped = (int) (p* weights.shape[0])
    mask = np.ones_like(weights)  
    dropout_mask = np.ones_like(weights)   
    ones_matrix = np.ones_like(weights) 
    sorted_weights = np.sort(weights, axis =1)  #sort the weights along the column  

    for i in range(weights.shape[0]):         #loop through each row 
        mask[i, :num_dropped] = 0            #make the mask value of required values 0
    
    for i in range(weights.shape[0]):         #generate dropout mask 
        for j in range(weights.shape[1]):
            prob = get_truncated_normal() 
            if (prob.rvs(1) < d):
                dropout_mask[i][j] = 1
            else:
                dropout_mask[i][j] = 0

    and_mask = mask*dropout_mask      #logical AND targeted mask and dropout mask
    final_mask = ones_matrix - and_mask                 # (1 - p*d) 
    masked_weights = final_mask * sorted_weights 

    return torch.from_numpy(masked_weights).cuda()
        
        
        

    



