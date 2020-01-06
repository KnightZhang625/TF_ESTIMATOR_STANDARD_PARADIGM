# coding:utf-8

import sys
from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

from utils.log import log_info as _info
from utils.log import log_error as _error

def forbid_new_attributes(wrapped_setatrr):
    def __setattr__(self, name, value):
        if hasattr(self, name):
            wrapped_setatrr(self, name, value)
        else:
            _error('Add new {} is forbidden'.format(name))
            raise AttributeError
    return __setattr__

class NoNewAttrs(object):
    """forbid to add new attributes"""
    __setattr__ = forbid_new_attributes(object.__setattr__)
    class __metaclass__(type):
        __setattr__ = forbid_new_attributes(type.__setattr__)

class Config(NoNewAttrs):
  # MAIN
  save_model_path = str(PROJECT_PATH / 'models')
  infer_pb_path = str(PROJECT_PATH / 'models_deployed')
  print_info_interval = 10
  keep_checkpoint_max = 1
  save_checkpoints_steps = 1000
  
  # data relevant
  buffer_size = 100
  batch_size = 32

  # train relevant
  learning_rate = 1e-1
  train_steps = 1000
  end_learning_rate = 2e-1
  colocate_gradients_with_ops = True

config = Config()
