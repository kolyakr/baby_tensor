from abc import ABC, abstractmethod
from src.layers import Layer

class Optimizer(ABC):
  def __init__(self):
    pass
  
  @abstractmethod
  def update_layers(
    self,
    layers: list[Layer],
    learning_rate: float,
  ):
    pass
    