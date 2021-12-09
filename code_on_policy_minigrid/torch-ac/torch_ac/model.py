from abc import abstractmethod, abstractproperty
import torch.nn as nn
import torch.nn.functional as F

class ACModel_Block:
    recurrent = False

    @abstractmethod
    def __init__(self, obs_space, action_space):
        pass

    @abstractmethod
    def forward(self, obs):
        pass

class RecurrentACModel_Block(ACModel_Block):
    recurrent = True

    @abstractmethod
    def forward(self, obs, memory, block_memory):
        pass

    @property
    @abstractmethod
    def memory_size(self):
        pass