from src.layers import Layer
from src.utils.activations import ActivationType, get_activation
from src.utils.initialization import get_initialization
import numpy as np

class ConvolutionalLayer(Layer):
  def __init__(
    self,
    name: str,
    activation_name: ActivationType,
    output_channels_dim: int,
    kernel_size: tuple = (3, 3),
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1
  ):
    super().__init__(name)
    
    self.activation_name = activation_name
    self.activation_fnc = get_activation(activation_name)[0]
    self.activation_derivative = get_activation(activation_name)[1]
    self.output_channels_dim = output_channels_dim
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    
  def initialize_parameters(self, input_dim, seed = 0):
    
    self.params["W"] = np.zeros((
      self.output_channels_dim,
      input_dim, 
      self.kernel_size[0], 
      self.kernel_size[1]
    ))
    self.grads["dW"] = np.zeros((
      self.output_channels_dim,
      input_dim,
      self.kernel_size[0], 
      self.kernel_size[1]
    ))
    
    self.params["b"] = np.zeros(self.output_channels_dim)
    self.grads["db"] = np.zeros((1, self.output_channels_dim))
    
    init_name = "he" if self.activation_name == "relu" else "xavier"
    initialize = get_initialization(init_name)
    
    for i in range(self.output_channels_dim):
      for j in range(input_dim):
        self.params["W"][i][j] = initialize(self.kernel_size)
      
    return self.output_channels_dim
  
  def forward_pass(self, A_prev):
    # A_prev.shape = (batch, channels, height, width)
    
    batch_size, input_channels_dim, input_h, input_w = A_prev.shape
    
    # we need to pad input channels in order to simplify computations
    A_padded = np.pad(
      array=A_prev,
      pad_width=((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
    )
    
    eff_kernel_size = (   # we used effective kernel size ONLY in order to compute output size
      self.kernel_size[0] + (self.kernel_size[0] - 1) * (self.dilation - 1),
      self.kernel_size[1] + (self.kernel_size[1] - 1) * (self.dilation - 1)
    )
    
    output_h = int((np.floor(input_h - eff_kernel_size[0] + 2 * self.padding) / self.stride)) + 1 
    output_w = int((np.floor(input_w - eff_kernel_size[1] + 2 * self.padding) / self.stride)) + 1 
    
    Z = np.zeros((
      batch_size,
      self.output_channels_dim,
      output_h,
      output_w
    ))
    
    for b in range(Z.shape[0]):
      for out_c in range(Z.shape[1]):
        for out_x in range(Z.shape[2]):
          for out_y in range(Z.shape[3]):
            Z_value = 0
            
            for in_c in range(input_channels_dim):  
              for kernel_x in range(self.kernel_size[0]):
                for kernel_y in range(self.kernel_size[1]):
                  
                  in_x = out_x * self.stride + kernel_x * self.dilation
                  in_y = out_y * self.stride + kernel_y * self.dilation
                  
                  Z_value += self.params["W"][out_c, in_c, kernel_x, kernel_y] * \
                             A_padded[b, in_c, in_x, in_y]
              
            Z[b, out_c, out_x, out_y] = Z_value
        
        Z[b, out_c, :, :] += self.params["b"][out_c]
              
    A = self.activation_fnc(Z)
        
    self.cache["A_prev"] = A_prev
    self.cache["Z"] = Z
    
    return A
    
  def backward_pass(self, dA):
    
    self.grads["dA"] = dA
    self.grads["dZ"] = self.grads["dA"] * self.activation_derivative(self.cache["Z"])
    self.grads["db"] = np.sum(self.grads["dZ"], axis=(0, 2, 3))
    
    # calculate gradient for W
    
    A_padded = np.pad(
      array=self.cache["A_prev"],
      pad_width=((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
    )
    
    for c_out in range(self.params["W"].shape[0]):
      for c_in in range(self.params["W"].shape[1]):
        for kernel_x in range(self.params["W"].shape[2]):
          for kernel_y in range(self.params["W"].shape[3]):
            
            dW = 0
            
            for b in range(self.cache["Z"].shape[0]):
              for out_x in range(self.cache["Z"].shape[2]):
                for out_y in range(self.cache["Z"].shape[3]):
                  
                  in_x = out_x * self.stride + kernel_x * self.dilation
                  in_y = out_y * self.stride + kernel_y * self.dilation
                  
                  dW += self.grads["dZ"][b, c_out, out_x, out_y] * A_padded[b, c_in, in_x, in_y]
            
            self.grads["dW"][c_out, c_in, kernel_x, kernel_y] = dW
            
    dA_prev = np.zeros_like(self.cache["A_prev"])

    dZ = self.grads["dZ"]  
    W = self.params["W"]  
    
    K_H, K_W = self.kernel_size
    B, C_out, H_out, W_out = dZ.shape
    
    s = self.stride
    d = self.dilation
    
    for b in range(B):
        for out_c in range(C_out):
            for in_c in range(self.cache["A_prev"].shape[1]): 

                for out_x in range(H_out):
                    for out_y in range(W_out):
                        
                        dZ_val = dZ[b, out_c, out_x, out_y]

                        for kernel_x in range(K_H):
                            for kernel_y in range(K_W):

                                W_val = W[out_c, in_c, kernel_x, kernel_y]
                                
                                in_x = out_x * s + kernel_x * d
                                in_y = out_y * s + kernel_y * d
                                
                                H_in = self.cache["A_prev"].shape[2]
                                W_in = self.cache["A_prev"].shape[3]
                                
                                if (in_x >= self.padding and in_x < H_in + self.padding and 
                                    in_y >= self.padding and in_y < W_in + self.padding):
                                    
                                    target_in_x = in_x - self.padding
                                    target_in_y = in_y - self.padding
                                    
                                    dA_prev[b, in_c, target_in_x, target_in_y] += dZ_val * W_val

    return dA_prev
              
    
    
    