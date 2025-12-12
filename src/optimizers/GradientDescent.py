from src.optimizers import Optimizer
from src.layers import Layer

class GradientDescent(Optimizer):
  def update_layers(
    self,
    layers: list[Layer],
    learning_rate: float
  ):
    for layer in layers:
      for key in layer.params.keys():
        layer.params[key] -= learning_rate * layer.grads[f"d{key}"]