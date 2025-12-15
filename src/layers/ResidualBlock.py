from src.layers import Layer

class ResidualBlock(Layer):
  def __init__(self, name):
    super().__init__(name)
    
    self.sub_layers : list[Layer] = []
  
  def initialize_parameters(self, input_dim, seed = 0):
    return input_dim
  
  def forward_pass(self, A_prev):
    
    self.cache["A_prev_skip"] = A_prev
    
    residual_path_output = A_prev
    for sublayer in self.sub_layers:
      residual_path_output = sublayer.forward_pass(residual_path_output)
      
    Z = residual_path_output + self.cache["A_prev_skip"]
    
    return Z
    
  def backward_pass(self, dZ):
    self.grads["dZ"] = dZ
    
    # dA_prev = dA_prev(main_residual_path) + dA_prev(skip)
    
    dA_prev_skip = dZ
    
    dA_prev_main_residual_path = dZ
    for sublayer in reversed(self.sub_layers):
      dA_prev_main_residual_path = sublayer.backward_pass(dA_prev_main_residual_path)
      
    dA_prev = dA_prev_main_residual_path + dA_prev_skip
    
    return dA_prev    