from .Optimizer import Optimizer
from .GradientDescent import GradientDescent
from .Momentum import Momentum
from .Adam import Adam
from typing import Literal

OptimizerType = Literal["gd", "momentum", "adam"]

def get_optimizer(name: OptimizerType):
  if(name == "gd"):
    return GradientDescent()
  if(name == "momentum"):
    return Momentum()
  if(name == "adam"):
    return Adam()
  else:
    return ValueError(f"Optimizer '{name}' does not exist")