from src.layers import Layer
from src.optimizers import Optimizer
import numpy as np

class Adam(Optimizer):
  def __init__(self, b: float = 0.9, g: float = 0.999):
    self.b = b
    self.g = g
    self.eps = 1e-10
    self.iter = 1
    
    self.m1 = {}
    self.m2 = {}
    
  def update_layers(
    self,
    layers: list[Layer],
    learning_rate
  ):
    for layer in layers:
      for key in layer.params.keys():
        
        m_key = f"{layer.name}_{key}"
        
        old_m1 = self.m1.get(m_key, np.zeros_like(layer.params[key]))
        old_m2 = self.m2.get(m_key, np.zeros_like(layer.params[key]))
        
        m1 = self.b * old_m1 + (1 - self.b) * layer.grads[f"d{key}"]
        m2 = self.g * old_m2 + (1 - self.g) * np.power(layer.grads[f"d{key}"], 2)
        
        corrected_m1 = m1 / (1 - np.power(self.b, self.iter))
        corrected_m2 = m2 / (1 - np.power(self.g, self.iter))
        
        layer.params[key] -= learning_rate * (corrected_m1 / (np.sqrt(corrected_m2 + 1) + self.eps))
        
        self.m1[m_key] = m1
        self.m2[m_key] = m2
    
    self.iter += 1