import numpy as np
from src.optimizers import Optimizer
from src.layers import Layer

class Momentum(Optimizer):
  def __init__(self, b: float = 0.9):
    self.b = b
    self.m = {}
    
  def update_layers(
    self,
    layers: list[Layer],
    learning_rate: float
  ):
    for layer in layers:
      for key in layer.params.keys():
        
        momentum_key = f"{layer.name}_{key}"
        old_momentum = self.m.get(momentum_key, np.zeros_like(layer.params[key]))
        
        momentum = self.b * old_momentum + (1 - self.b) * layer.grads[f"d{key}"]
        
        layer.params[key] -= learning_rate * momentum
        
        self.m[momentum_key] = momentum
              