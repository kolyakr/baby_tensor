from src.layers import Layer
import numpy as np

class FlattenLayer(Layer):
  def __init__(self, name):
    super().__init__(name)
    
  def initialize_parameters(self, input_shape: tuple, seed: int = 0):
    np.random.seed(seed)
    
    c, h, w = input_shape
    
    output_dim = c * h * w
    
    return output_dim
    
  def forward_pass(self, A_prev):
    
    self.cache["A_prev"] = A_prev
    
    b, c, h, w = A_prev.shape
    m = c * h * w
    
    A = A_prev.reshape(b, m)
    
    A_transposed = A.T
    
    return A_transposed
    
  def backward_pass(self, dA):
    dA_transposed = dA.T
    
    self.grads["dA"] = dA_transposed
    
    dA_prev = dA_transposed.reshape(self.cache["A_prev"].shape)
    
    return dA_prev 
  