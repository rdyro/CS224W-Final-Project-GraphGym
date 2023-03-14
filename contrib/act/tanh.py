import torch
import torch.nn as nn

from graphgym.config import cfg
from graphgym.register import register_act

register_act('tanh', torch.nn.Tanh())
