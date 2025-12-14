from src.layers import Layer
import numpy as np

class MaxPoolingLayer(Layer):
  def __init__(self, name, pool_size: tuple = (2, 2), stride: int = 2):
    super().__init__(name)
    self.pool_size = pool_size
    self.stride = stride
    self.index_map = {}
    
  def initialize_parameters(self, input_dim: tuple, seed = 0):
    
    c, in_h, in_w = input_dim
    
    out_h = int(np.floor((in_h - self.pool_size[0]) / self.stride)) + 1
    out_w = int(np.floor((in_w - self.pool_size[1]) / self.stride)) + 1
    
    return (c, out_h, out_w)
  
  def forward_pass(self, A_prev):
    
    self.cache["A_prev"] = A_prev
    
    batch_size, channels, in_h, in_w = A_prev.shape
    
    out_h = int(np.floor((in_h - self.pool_size[0]) / self.stride)) + 1
    out_w = int(np.floor((in_w - self.pool_size[1]) / self.stride)) + 1
    
    A = np.zeros((
      batch_size,
      channels,
      out_h,
      out_w
    ))
  
    for b in range(batch_size):
      for c in range(channels):
        for out_x in range(out_h):
          for out_y in range(out_w):
            map_key = f"{b}_{c}_{out_x}_{out_y}"
            
            max_val = -np.inf
            max_indices = (0, 0)
            
            for pool_f_x in range(self.pool_size[0]):
              for pool_f_y in range(self.pool_size[1]):
                
                in_x = out_x * self.stride + pool_f_x
                in_y = out_y * self.stride + pool_f_y
                
                curr_val = A_prev[b, c, in_x, in_y]
                
                if curr_val >= max_val:
                  max_val = curr_val
                  max_indices = (in_x, in_y)    

            A[b, c, out_x, out_y] = max_val
            self.index_map[map_key] = max_indices
    
    return A
  
  def backward_pass(self, dA):
    
    self.grads["dA"] = dA
    
    dA_prev = np.zeros_like(self.cache["A_prev"])
    
    batch_size, channels, out_h, out_w = dA.shape
    
    for b in range(batch_size):
      for c in range(channels):
        for out_x in range(out_h):
          for out_y in range(out_w):
            
            map_key = f"{b}_{c}_{out_x}_{out_y}"
            max_in_x, max_in_y = self.index_map[map_key]
            contribution = dA[b, c, out_x, out_y]
            
            # we do not need to iterate over filter
            #
            #for pool_f_x in range(self.pool_size[0]):
            #  for pool_f_y in range(self.pool_size[1]):
            #
            # because all elements in dA_prev are zeros already and we need to change only
            # in max_in_x, max_in_y posistions
            
            dA_prev[b, c, max_in_x, max_in_y] += contribution
    
    self.index_map = {}
    return dA_prev  