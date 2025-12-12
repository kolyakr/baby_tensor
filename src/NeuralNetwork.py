import numpy as np
from src.layers import Layer
from src.utils.losses import LossType, get_loss
from src.optimizers import OptimizerType, get_optimizer

class NeuralNetwork:
  def __init__(
    self, 
    layers: list[Layer],
    loss_type: LossType,
    optimizer_type: OptimizerType,
    seed: int
  ):
    self.layers = layers
    self.loss_type = loss_type
    self.optimizer = get_optimizer(optimizer_type)
    self.seed = seed
    
  def build(self, input_dim: int):
    temp_input_dim = input_dim
    
    for layer in self.layers:
      temp_input_dim = layer.initialize_parameters(temp_input_dim, self.seed)
  
  def forward_pass(self, X: np.ndarray):
    
    A = X
      
    for layer in self.layers:
      A = layer.forward_pass(A)
      
    return A

  def backward_pass(self, y_true: np.ndarray, y_hat: np.ndarray):
    
    _, loss_derivative = get_loss(self.loss_type)
    
    dA = loss_derivative(y_hat, y_true)
    
    for layer in self.layers[::-1]:
      dA = layer.backward_pass(dA)
      
  def train(
    self, 
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 100,
    batch_size: int = 34,
    learning_rate: int = 0.01
  ):
    
    n = X_train.shape[1]
    idx = range(n)
    loss_fnc, _ = get_loss(self.loss_type)
    
    for i in range(epochs):     
      shuffled_idx = np.random.permutation(idx)
      
      for start_idx in range(0, n, batch_size):
        
        end_index = start_idx + batch_size
        batch_idx = shuffled_idx[start_idx:end_index]
        
        X_train_batch = X_train[:, batch_idx]
        y_train_batch = y_train[:, batch_idx]
        
        y_hat = self.forward_pass(X_train_batch)
        self.backward_pass(y_train_batch, y_hat)
        self.optimizer.update_layers(self.layers, learning_rate)
        
        curr_iter = n // batch_size
        
        if curr_iter % 100 == 0:
          loss = loss_fnc(y_hat, y_train_batch)
          print(f"Epoch {i}, Iteration {curr_iter}: Loss = {loss:.4f}")