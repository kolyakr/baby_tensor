"""
ConvolutionalLayer - Documented Version

This layer implements 2D convolution with support for stride, padding, and dilation.
Uses Numba JIT compilation for performance optimization of computationally intensive operations.

Note: Numba functions are included for performance optimization. The core mathematical
operations are conceptually simple but computationally expensive, hence the JIT compilation.
"""

import numpy as np
from numba import njit
from src.layers import Layer
from src.utils.initialization import get_initialization
from src.utils.activations import ActivationType

# ============================================================================
# NUMBA-OPTIMIZED HELPER FUNCTIONS
# (Written for performance optimization of computationally intensive operations)
# ============================================================================

@njit(cache=True)
def convolve_forward_numba(
    A_padded: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    stride: int,
    dilation: int,
    Z: np.ndarray
) -> np.ndarray:
    """
    Numba-optimized forward convolution operation.
    
    Performs the core convolution operation: slides filters over the input,
    computing dot products at each position. This is the computationally
    intensive part of the forward pass, hence the Numba optimization.
    
    Parameters:
        A_padded (np.ndarray): Padded input activations
            Shape: (batch_size, in_channels, padded_height, padded_width)
        W (np.ndarray): Convolution filters/kernels
            Shape: (out_channels, in_channels, kernel_height, kernel_width)
        b (np.ndarray): Bias terms
            Shape: (out_channels,)
        stride (int): Step size for sliding the kernel
        dilation (int): Spacing between kernel elements (for dilated convolution)
        Z (np.ndarray): Pre-allocated output array (modified in-place)
            Shape: (batch_size, out_channels, out_height, out_width)
            
    Returns:
        np.ndarray: Convolution output (same as Z input, modified in-place)
            Shape: (batch_size, out_channels, out_height, out_width)
    """
    batch_size, out_c_dim, out_h, out_w = Z.shape
    in_c_dim, k_h, k_w = W.shape[1], W.shape[2], W.shape[3]
    
    # Nested loops for batch, output channels, output spatial dimensions,
    # input channels, and kernel dimensions
    for b_idx in range(batch_size):
        for oc in range(out_c_dim):
            for oh in range(out_h):
                for ow in range(out_w):
                    val = 0.0
                    # Compute convolution at this output position
                    for ic in range(in_c_dim):
                        for kh in range(k_h):
                            for kw in range(k_w):
                                # Calculate input position with stride and dilation
                                ix = oh * stride + kh * dilation
                                iy = ow * stride + kw * dilation
                                val += W[oc, ic, kh, kw] * A_padded[b_idx, ic, ix, iy]
                    Z[b_idx, oc, oh, ow] = val + b[oc]
    return Z


@njit(cache=True)
def convolve_backward_dA_numba(
    dZ: np.ndarray,
    W: np.ndarray,
    padding: int,
    stride: int,
    dilation: int,
    dA_prev_shape: tuple
) -> np.ndarray:
    """
    Numba-optimized backward pass for computing input gradient.
    
    Computes the gradient of the loss with respect to the input activations.
    This operation is also computationally intensive (similar to forward pass),
    hence the Numba optimization.
    
    The gradient flows backwards through the convolution operation, accumulating
    contributions from all output positions that were influenced by each input position.
    
    Parameters:
        dZ (np.ndarray): Gradient of loss with respect to convolution output
            Shape: (batch_size, out_channels, out_height, out_width)
        W (np.ndarray): Convolution filters/kernels
            Shape: (out_channels, in_channels, kernel_height, kernel_width)
        padding (int): Amount of padding used in forward pass
        stride (int): Stride used in forward pass
        dilation (int): Dilation used in forward pass
        dA_prev_shape (tuple): Shape of the original input (before padding)
            Format: (batch_size, in_channels, in_height, in_width)
            
    Returns:
        np.ndarray: Gradient with respect to input activations
            Shape: Same as dA_prev_shape
    """
    B, C_out, H_out, W_out = dZ.shape
    C_in, k_h, k_w = W.shape[1], W.shape[2], W.shape[3]
    dA_prev = np.zeros(dA_prev_shape)
    
    for b in range(B):
        for oc in range(C_out):
            for ic in range(C_in):
                for oh in range(H_out):
                    for ow in range(W_out):
                        dz_val = dZ[b, oc, oh, ow]
                        # Propagate gradient through each kernel position
                        for kh in range(k_h):
                            for kw in range(k_w):
                                # Calculate corresponding input position
                                ix = oh * stride + kh * dilation
                                iy = ow * stride + kw * dilation
                                
                                # Bounds check accounting for padding
                                if (padding <= ix < dA_prev_shape[2] + padding and 
                                    padding <= iy < dA_prev_shape[3] + padding):
                                    # Convert from padded to unpadded coordinates
                                    tx = ix - padding
                                    ty = iy - padding
                                    dA_prev[b, ic, tx, ty] += dz_val * W[oc, ic, kh, kw]
    return dA_prev


# ============================================================================
# MAIN CONVOLUTIONAL LAYER CLASS
# ============================================================================

class ConvolutionalLayer(Layer):
    """
    2D Convolutional Layer
    
    Applies 2D convolution over input activations with support for:
    - Multiple input/output channels
    - Configurable kernel size
    - Stride (subsampling)
    - Padding (zero-padding)
    - Dilation (atrous convolution)
    
    Attributes:
        name (str): Layer identifier
        output_channels_dim (int): Number of output channels (number of filters)
        next_act_layer (ActivationType): Type of activation function in next layer
                                         (used for weight initialization)
        kernel_size (tuple): Size of convolution kernel (height, width)
        stride (int): Step size for sliding the kernel
        padding (int): Amount of zero-padding added to input borders
        dilation (int): Spacing between kernel elements (1 = standard convolution)
        params (dict): Layer parameters
            - W (np.ndarray): Convolution kernels/filters
            - b (np.ndarray): Bias terms
        cache (dict): Cached values for backward pass
            - A_prev (np.ndarray): Input activations
        grads (dict): Gradients computed during backward pass
            - dW (np.ndarray): Gradient of W
            - db (np.ndarray): Gradient of b
    """
    
    def __init__(
        self,
        name: str,
        output_channels_dim: int,
        next_act_layer: ActivationType,
        kernel_size: tuple = (3, 3),
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ) -> None:
        """
        Initialize the Convolutional Layer.
        
        Parameters:
            name (str): Layer identifier
            output_channels_dim (int): Number of output channels (filters)
            next_act_layer (ActivationType): Activation type for next layer
                                             ("relu" uses He init, others use Xavier)
            kernel_size (tuple, optional): Kernel dimensions (height, width)
                                          Default: (3, 3)
            stride (int, optional): Sliding window step size. Default: 1
            padding (int, optional): Zero-padding amount. Default: 1
            dilation (int, optional): Kernel element spacing. Default: 1
                                     (1 = standard, >1 = dilated/atrous convolution)
                                     
        Returns:
            None
        """
        super().__init__(name)
        self.output_channels_dim = output_channels_dim
        self.next_act_layer = next_act_layer
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def initialize_parameters(self, input_shape: tuple, seed: int = 0) -> tuple:
        """
        Initialize convolution weights and biases.
        
        Weights are initialized using He initialization (for ReLU) or Xavier
        initialization (for other activations). Biases are initialized to zero.
        
        Also computes and returns the output spatial dimensions after convolution.
        
        Parameters:
            input_shape (tuple): Input dimensions (channels, height, width)
            seed (int, optional): Random seed for reproducibility. Default: 0
            
        Returns:
            tuple: Output shape (out_channels, out_height, out_width)
                   where spatial dimensions are computed as:
                   out_h = (in_h + 2*padding - effective_kernel_h) // stride + 1
                   out_w = (in_w + 2*padding - effective_kernel_w) // stride + 1
                   effective_kernel = kernel + (kernel - 1) * (dilation - 1)
        """
        np.random.seed(seed)
        input_c, input_h, input_w = input_shape
        
        # Initialize weights and biases
        self.params["W"] = np.zeros((self.output_channels_dim, input_c, *self.kernel_size))
        self.params["b"] = np.zeros(self.output_channels_dim)
        
        # Select initialization based on next activation layer
        init_name = "he" if self.next_act_layer == "relu" else "xavier"
        initialize = get_initialization(init_name)
        
        # Initialize each filter
        for i in range(self.output_channels_dim):
            for j in range(input_c):
                self.params["W"][i][j] = initialize(self.kernel_size)
                
        # Calculate output dimensions considering dilation
        # Effective kernel size accounts for dilation: k_eff = k + (k-1)*(d-1)
        eff_kh = self.kernel_size[0] + (self.kernel_size[0] - 1) * (self.dilation - 1)
        eff_kw = self.kernel_size[1] + (self.kernel_size[1] - 1) * (self.dilation - 1)
        
        output_h = (input_h + 2 * self.padding - eff_kh) // self.stride + 1
        output_w = (input_w + 2 * self.padding - eff_kw) // self.stride + 1
        
        return (self.output_channels_dim, output_h, output_w)

    def forward_pass(self, A_prev: np.ndarray) -> np.ndarray:
        """
        Perform forward pass of convolution operation.
        
        Steps:
            1. Cache input for backward pass
            2. Apply zero-padding to input
            3. Compute output dimensions
            4. Perform convolution using Numba-optimized function
        
        The convolution operation computes:
            Z[b, oc, oh, ow] = sum over (ic, kh, kw) of:
                W[oc, ic, kh, kw] * A_padded[b, ic, oh*stride + kh*dilation, ow*stride + kw*dilation]
            + b[oc]
        
        Parameters:
            A_prev (np.ndarray): Input activations
                Shape: (batch_size, in_channels, in_height, in_width)
                
        Returns:
            np.ndarray: Convolution output
                Shape: (batch_size, out_channels, out_height, out_width)
        """
        self.cache["A_prev"] = A_prev
        batch_size, in_c, in_h, in_w = A_prev.shape
        
        # Apply zero-padding
        A_padded = np.pad(
            A_prev, 
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
        )
        
        # Calculate output spatial dimensions
        eff_kh = self.kernel_size[0] + (self.kernel_size[0] - 1) * (self.dilation - 1)
        eff_kw = self.kernel_size[1] + (self.kernel_size[1] - 1) * (self.dilation - 1)
        out_h = (in_h + 2 * self.padding - eff_kh) // self.stride + 1
        out_w = (in_w + 2 * self.padding - eff_kw) // self.stride + 1
        
        # Pre-allocate output array
        Z = np.zeros((batch_size, self.output_channels_dim, out_h, out_w))
        
        # Execute fast Numba-optimized convolution
        Z = convolve_forward_numba(
            A_padded, 
            self.params["W"], 
            self.params["b"], 
            self.stride, 
            self.dilation, 
            Z
        )
        
        return Z

    def backward_pass(self, dZ: np.ndarray) -> np.ndarray:
        """
        Perform backward pass to compute gradients.
        
        Computes three gradients:
            1. db: gradient with respect to biases (simple sum over batch and spatial dims)
            2. dW: gradient with respect to weights (convolution of input with dZ)
            3. dA_prev: gradient with respect to input (transposed convolution of dZ with W)
        
        The weight gradient is computed by treating dZ as filters and A_prev as input.
        The input gradient uses Numba optimization due to computational intensity.
        
        All gradients are averaged over the batch dimension.
        
        Parameters:
            dZ (np.ndarray): Gradient of loss with respect to convolution output
                Shape: (batch_size, out_channels, out_height, out_width)
                
        Returns:
            np.ndarray: Gradient of loss with respect to input (dA_prev)
                Shape: (batch_size, in_channels, in_height, in_width)
        """
        A_prev = self.cache["A_prev"]
        batch_size = dZ.shape[0]
        
        # 1. Gradient for bias: sum over all dimensions except output channels
        self.grads["db"] = np.sum(dZ, axis=(0, 2, 3)) / batch_size
        
        # 2. Gradient for Weights
        # Pad input for proper alignment
        A_padded = np.pad(
            A_prev, 
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
        )
        dW = np.zeros_like(self.params["W"])
        
        # Compute dW by correlating A_padded with dZ
        for oc in range(self.output_channels_dim):
            for ic in range(A_prev.shape[1]):
                for kh in range(self.kernel_size[0]):
                    for kw in range(self.kernel_size[1]):
                        # Extract relevant slice of input using stride and dilation
                        h_start = kh * self.dilation
                        h_end = h_start + dZ.shape[2] * self.stride
                        w_start = kw * self.dilation
                        w_end = w_start + dZ.shape[3] * self.stride
                        
                        # Vectorized computation over batch dimension
                        a_slice = A_padded[:, ic, h_start:h_end:self.stride, w_start:w_end:self.stride]
                        dW[oc, ic, kh, kw] = np.sum(dZ[:, oc, :, :] * a_slice) / batch_size
        
        self.grads["dW"] = dW
        
        # 3. Gradient for input (using Numba for performance)
        dA_prev = convolve_backward_dA_numba(
            dZ, 
            self.params["W"], 
            self.padding, 
            self.stride, 
            self.dilation, 
            A_prev.shape
        )
        
        return dA_prev