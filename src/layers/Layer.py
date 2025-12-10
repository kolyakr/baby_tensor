import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
  def __init__(self, name: str):
    self.name = name
    self.params = {}
    self.grads = {}
    self.cache = {}
  
  @abstractmethod
  def initialize_parameters(self, input_dim: int, seed: int = 0):
    pass
  
  @abstractmethod
  def forward_pass(self, A_prev: np.ndarray):  
    pass

  @abstractmethod
  def backward_pass(self, dA: np.ndarray):    
    pass