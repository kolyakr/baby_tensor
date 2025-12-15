from src.layers import Layer
from src.utils.activations import get_activation

class SoftmaxLayer(Layer):
  def __init__(self, name):
    super().__init__(name)
    
    self.act_func = get_activation("softmax")[0]
    
  def initialize_parameters(self, input_dim, seed = 0):
    return input_dim
  
  def forward_pass(self, Z):
    self.cache["Z"] = Z
    
    A = self.act_func(Z)
    
    return A

  def backward_pass(self, dA):
    self.grads["dA"] = dA 
    
    dZ = dA
    
    return dZ 