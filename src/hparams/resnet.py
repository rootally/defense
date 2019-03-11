from .registry import register

class HParams():
  def __init__(self):
    self.model = "ResNet18"
    self.batch_size = 128
    self.targeted_weight = False
    self.targeted_unit = False
    self.targ_perc = 0.0
    self.drop_rate = 0.0
    self.learning_rate = 0.1
    self.momentum = 0.9
    self.weight_decay = 5e-4
    self.num_epochs = 256

def resnet18_default():
  hps = HParams()
  return hps

def resnet18_targ_weight_075_drop_033():
  hps = resnet18_default()
  hps.targeted_weight = True
  hps.targ_perc = 0.75
  hps.drop_rate = 0.33

  return hps

def resnet18_targ_weight_075_drop_050():
  hps = resnet18_targ_weight_075_drop_033()
  hps.drop_rate  = 0.5

  return hps

def resnet18_targ_weight_075_drop_066():
  hps = resnet18_targ_weight_075_drop_050()
  hps.drop_rate  = 0.66

  return hps  

def resnet18_targ_weight_050_drop_050():
  hps = resnet18_targ_weight_075_drop_050()
  hps.targ_perc = 0.50

  return hps  

def resnet34_default():
  hps = HParams()
  hps.model = "ResNet34"
  return hps

def resnet34_targ_weight_075_drop_033():
  hps = resnet34_default()
  hps.targeted_weight = True
  hps.targ_perc = 0.75
  hps.drop_rate = 0.33

  return hps

def resnet34_targ_weight_075_drop_050():
  hps = resnet34_targ_weight_075_drop_033()
  hps.drop_rate  = 0.5

  return hps

def resnet34_targ_weight_075_drop_066():
  hps = resnet34_targ_weight_075_drop_050()
  hps.drop_rate  = 0.66

  return hps  

def resnet34_targ_weight_050_drop_050():
  hps = resnet34_targ_weight_075_drop_050()
  hps.targ_perc = 0.50

  return hps    