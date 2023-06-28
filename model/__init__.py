from .resnet import *
from .densenet import *
from .densenet3 import *

def load_model(name, pretrained=False, **kwargs):
  model_dict = globals()
  model = model_dict[name](pretrained=pretrained, **kwargs)
  return model