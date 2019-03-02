import torch

"""
References:
Targeted Dropout by Aidan N. Gomez, Ivan Zhang, Kevin Swersky, Yarin Gal, and Geoffrey E. Hintonhe
The code release for the same.
"""

def targeted_weight_droput(w, drop_rate,targ_perc, is_training):
    w_shape = list(w.size())
    
#    print( w_shape)
    w = w.view(-1,w_shape[-1])
#    print(w.size())
    norm = torch.abs(w)
#    print(norm.size())
    idx = int(targ_perc * w_shape[0])
    threshold = (torch.sort(norm, dim=0)[0])[idx]
    mask = norm < (threshold[0])
#    print(mask.size())

    if not is_training:
        # mask = mask.float()
        w = (1. - mask.float())*w
        w = w.view(w_shape)
#        print(w)
    return w

def targeted_unit_dropout(w, drop_rate, targ_rate, is_training):
    w_shape = list(w.size())
    w = w.view(-1, w_shape[-1])
    norm = torch.norm(w, dim=0)
    idx = int(targ_rate * int(w.shape[1]))
    sorted_norms = torch.sort(norm)
    threshold = (sorted_norms[0])[idx]
    mask = (norm < threshold)[None,:]
    mask = mask.repeat(w.shape[0],1)
    mask =  torch.where(((1. - drop_rate) < torch.empty(list(w.size())).uniform_(0, 1)) & mask,
                        torch.ones(w.shape, dtype=torch.float32),
                        torch.zeros(w.shape, dtype = torch.float32))
    x = (1-mask) * w
    x = x.view(w_shape)
    return x