import numpy as np
from src.layers import Layer, DenseLayer
from src.utils.losses import LossType

class NeuralNetwork:
  def __init__(
    self, 
    layers: list[Layer],
    loss_type: LossType,
    seed: int
  ):
    self.layers = layers
    self.loss_type = loss_type
    self.seed = seed
    
    