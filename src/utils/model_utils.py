import torch
from src.utils.dropout import targeted_weight_dropout, targeted_unit_dropout

def conv(weight, params):

  #TODO - actually pass the is_training mode here. Not just True/False
  if params.targeted_weight:
    weight = torch.nn.Parameter(targeted_weight_dropout(weight, params, True))
  elif params.targeted_unit:
    weight = torch.nn.Parameter(targeted_weight_dropout(weight, params, True))

  return weight
