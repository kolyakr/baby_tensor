from src.layers.Layer import Layer
from src.utils.initialization import get_initialization
from src.utils.activations import ActivationType, get_activation
import numpy as np

class DenseLayer(Layer):
  def __init__(self, output_dim: int, activation_name: ActivationType, name: str = "Dense"):
    super().__init__(name)
    self.output_dim = output_dim
    
    if(activation_name not in ActivationType.__args__):
      return ValueError(f"Activation name '{activation_name}' does not exist")
    
    self.activation_name = activation_name
    self.activation_fnc = get_activation(activation_name)[0]
    self.activation_derivative = get_activation(activation_name)[1]
    
    
  def initialize_parameters(self, input_dim: int, seed: int = 0):
    np.random.seed(seed)
    
    init_name = "he" if self.activation_name == "relu" else "xavier"
    initialization = get_initialization(init_name)
    
    W_shape = (self.output_dim, input_dim)
    
    self.params["W"] = initialization(W_shape)
    self.params["b"] = np.zeros((self.output_dim, 1))
    
    self.grads["dW"] = np.zeros(W_shape)
    self.grads["db"] = np.zeros((self.output_dim, 1))
    
  def forward_pass(self, A_prev):
    Z = np.matmul(self.params["W"], A_prev) + np.matmul(self.params["b"], np.ones((1, A_prev.shape[1])))
    A = self.activation_fnc(Z)
    
    self.cache["Z"] = Z
    self.cache["A_prev"] = A_prev
    
    return A
    
  def backward_pass(self, dA):
    dL_dZ = self.activation_derivative(self.cache["Z"]) * dA
    dL_dA_prev = np.matmul(self.params["W"].transpose(), dL_dZ)
    
    self.grads["dA"] = dA
    self.grads["dZ"] = dL_dZ
    
    dL_dW = np.matmul(dL_dZ, self.cache["A_prev"].transpose())
    dL_db = dL_dZ
    
    self.grads["dW"] = dL_dW
    self.grads["db"] = dL_db
    
    return dL_dA_prev