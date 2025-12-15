from src.layers import Layer
from src.utils.activations import get_activation

class SigmoidLayer(Layer):
  def __init__(self, name):
    super().__init__(name)
    
    self.act_func = get_activation("sigmoid")[0]
    self.act_deriv = get_activation("sigmoid")[1]
    
  def initialize_parameters(self, input_dim, seed = 0):
    return input_dim
  
  def forward_pass(self, Z):
    self.cache["Z"] = Z
    
    A = self.act_func(Z)
    
    return A

  def backward_pass(self, dA):
    self.grads["dA"] = dA 
    
    dZ = dA * self.act_deriv(self.cache["Z"])
    
    return dZ 