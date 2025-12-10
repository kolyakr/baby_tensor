import numpy as np
from src.layers import Layer
from src.utils.losses import LossType
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
      
      
      
    
  
  
    
    