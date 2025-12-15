from src.layers.Layer import Layer
from src.utils.initialization import get_initialization
from src.utils.activations import ActivationType, get_activation
import numpy as np

class DenseLayer(Layer):
  def __init__(self, output_dim: int, next_act_layer: ActivationType, name: str = "Dense"):
    super().__init__(name)
    self.output_dim = output_dim
    self.next_act_layer = next_act_layer
    
    
  def initialize_parameters(self, input_dim: int, seed: int = 0):
    np.random.seed(seed)
    
    init_name = "he" if self.next_act_layer == "relu" else "xavier"
    initialization = get_initialization(init_name)
    
    W_shape = (self.output_dim, input_dim)
    
    self.params["W"] = initialization(W_shape)
    self.params["b"] = np.zeros((self.output_dim, 1))
    
    self.grads["dW"] = np.zeros(W_shape)
    self.grads["db"] = np.zeros((self.output_dim, 1))
    
    # return the input dimension of for the next layer
    return self.output_dim
    
  def forward_pass(self, A_prev):
    self.cache["A_prev"] = A_prev
    
    Z = np.matmul(self.params["W"], A_prev) + np.matmul(self.params["b"], np.ones((1, A_prev.shape[1])))
    
    return Z
    
  def backward_pass(self, dZ):
      
    self.grads["dZ"] = dZ
    self.grads["dW"] = np.matmul(dZ, self.cache["A_prev"].transpose())
    self.grads["db"] = np.sum(dZ, axis=1, keepdims=True)
    
    dA_prev = np.matmul(self.params["W"].transpose(), dZ)
    
    return dA_prev