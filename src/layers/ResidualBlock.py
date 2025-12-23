"""
ResidualBlock - Documented Version

This layer implements a residual connection (skip connection), a key innovation
from ResNet that enables training of very deep neural networks.

The residual block adds the input directly to the output of a series of layers:
    output = F(input) + input

where F represents the transformations applied by the sub-layers.

Benefits:
    1. Mitigates vanishing gradient problem in deep networks
    2. Enables gradient flow directly to earlier layers via skip connection
    3. Allows network to learn identity mapping easily (just set F ≈ 0)
    4. Enables training of networks with 100+ layers
    5. Often improves convergence speed and final performance
"""

from src.layers import Layer
import numpy as np


class ResidualBlock(Layer):
    """
    Residual Block (Skip Connection)
    
    Implements a residual connection that adds the input to the output of
    a sequence of sub-layers. This is the fundamental building block of
    ResNet architectures.
    
    Architecture:
        Input (A_prev)
            |
            |-------------- skip connection ------------|
            |                                            |
            v                                            v
        [sub_layer_1] -> [sub_layer_2] -> ... -> [sub_layer_n]
            |                                            |
            v                                            v
        F(A_prev)                                    A_prev
            |                                            |
            └----------------> (+) <--------------------┘
                                |
                                v
                            Output (Z)
    
    Mathematical operation:
        Forward:  Z = F(A_prev) + A_prev
        Backward: dA_prev = dF + dA_skip
                  where dF flows through sub-layers and dA_skip flows directly
    
    The skip connection provides a direct gradient path, allowing gradients
    to flow backwards without being diminished by multiple layer operations.
    
    Attributes:
        name (str): Layer identifier
        sub_layers (list[Layer]): Sequence of layers forming the residual path
                                  These layers compute F(x)
        params (dict): Empty at block level - parameters stored in sub-layers
        cache (dict): Cached values for backward pass
            - A_prev_skip (np.ndarray): Input saved for skip connection
        grads (dict): Cached gradients
            - dZ (np.ndarray): Gradient w.r.t. block output
    
    Note: Input and output dimensions must match for the addition operation.
          If dimensions differ, a projection layer (1×1 conv or dense) may be
          needed on the skip connection.
    """
    
    def __init__(self, name: str, sub_layers: list[Layer]) -> None:
        """
        Initialize the Residual Block.
        
        Parameters:
            name (str): Block identifier
            sub_layers (list[Layer]): Ordered list of layers in the residual path
                                      These layers compute F(x) in Z = F(x) + x
                                      Can include conv layers, batch norm, activations, etc.
                                      
        Returns:
            None
            
        Example:
            >>> from src.layers import ConvolutionalLayer, BatchNormalizationLayer, ReLULayer
            >>> sub_layers = [
            ...     ConvolutionalLayer("conv1", 64, "relu"),
            ...     BatchNormalizationLayer("bn1"),
            ...     ReLULayer("relu1"),
            ...     ConvolutionalLayer("conv2", 64, "linear")
            ... ]
            >>> residual = ResidualBlock("res_block_1", sub_layers)
        """
        super().__init__(name)
        
        self.sub_layers = sub_layers
    
    def initialize_parameters(self, input_dim: int | tuple, seed: int = 0) -> int | tuple:
        """
        Initialize parameters for all sub-layers in the residual block.
        
        Chains the initialization through all sub-layers sequentially.
        Each sub-layer's output dimension becomes the next sub-layer's input dimension.
        
        The final output dimension should match the input dimension for the
        skip connection to work (element-wise addition requires matching shapes).
        If dimensions don't match, the architecture needs adjustment (e.g., projection).
        
        Parameters:
            input_dim (int | tuple): Input dimensions to the residual block
                                     Format depends on data type:
                                     - int: for 1D data (features)
                                     - tuple: for multi-dimensional data (C, H, W)
            seed (int, optional): Random seed for reproducible initialization
                                 Default: 0
                                 
        Returns:
            int | tuple: Output dimensions of the residual block
                        Should typically match input_dim for standard residual connections
                        
        Example:
            >>> input_shape = (64, 28, 28)  # 64 channels, 28×28 spatial
            >>> output_shape = residual.initialize_parameters(input_shape, seed=42)
            >>> print(output_shape)  # Should be (64, 28, 28) for proper residual connection
        """
        curr_input_dim = input_dim
        
        # Chain initialization through all sub-layers
        for sublayer in self.sub_layers:
            curr_input_dim = sublayer.initialize_parameters(curr_input_dim, seed)
        
        return curr_input_dim
    
    def forward_pass(self, A_prev: np.ndarray) -> np.ndarray:
        """
        Perform forward pass through the residual block.
        
        Computes: Z = F(A_prev) + A_prev
        where F is the composition of all sub-layers.
        
        Steps:
            1. Cache the input (needed for skip connection)
            2. Pass input through all sub-layers sequentially (compute F(A_prev))
            3. Add the original input to the sub-layers' output (skip connection)
            4. Return the sum
        
        The skip connection allows the network to learn residual functions,
        which are often easier to optimize than learning the full transformation.
        
        Parameters:
            A_prev (np.ndarray): Input to the residual block
                                Shape depends on architecture:
                                - (features, batch_size) for dense networks
                                - (batch_size, channels, height, width) for CNNs
                                
        Returns:
            np.ndarray: Output of residual block Z = F(A_prev) + A_prev
                       Same shape as A_prev
                       
        Example:
            >>> A_prev = np.random.randn(32, 64, 28, 28)  # 32 samples, 64 channels
            >>> Z = residual.forward_pass(A_prev)
            >>> print(Z.shape)  # (32, 64, 28, 28) - same as input
        """
        # Cache input for skip connection
        self.cache["A_prev_skip"] = A_prev
        
        # Compute F(A_prev) by passing through all sub-layers
        residual_path_output = A_prev
        for sublayer in self.sub_layers:
            residual_path_output = sublayer.forward_pass(residual_path_output)
        
        # Add skip connection: Z = F(A_prev) + A_prev
        Z = residual_path_output + self.cache["A_prev_skip"]
        
        return Z
    
    def backward_pass(self, dZ: np.ndarray) -> np.ndarray:
        """
        Perform backward pass through the residual block.
        
        The gradient splits into two paths:
            1. Main path: flows backwards through all sub-layers
            2. Skip path: flows directly to the input (unchanged)
        
        Then these gradients are summed: dA_prev = dF + dA_skip
        
        This dual-path gradient flow is the key benefit of residual connections:
            - The skip path ensures gradients can flow directly to earlier layers
            - The main path allows learning of residual transformations
            - Combined, they mitigate vanishing gradients in deep networks
        
        Mathematical derivation:
            Z = F(A_prev) + A_prev
            
            ∂L/∂A_prev = ∂L/∂Z * ∂Z/∂A_prev
                       = dZ * (∂F/∂A_prev + ∂A_prev/∂A_prev)
                       = dZ * (∂F/∂A_prev + I)
                       = dZ * ∂F/∂A_prev + dZ
                       = dF + dA_skip
        
        Parameters:
            dZ (np.ndarray): Gradient of loss with respect to block output
                            Shape matches the output from forward_pass
                            
        Returns:
            np.ndarray: Gradient of loss with respect to block input (dA_prev)
                       dA_prev = gradient through main path + gradient through skip
                       Same shape as A_prev from forward_pass
                       
        Example:
            >>> dZ = np.random.randn(32, 64, 28, 28)
            >>> dA_prev = residual.backward_pass(dZ)
            >>> print(dA_prev.shape)  # (32, 64, 28, 28)
            >>> # dA_prev contains gradients from both paths
        """
        self.grads["dZ"] = dZ
        
        # Gradient flow through residual block:
        # dA_prev = dA_prev(main_residual_path) + dA_prev(skip)
        
        # Skip path: gradient flows directly (identity derivative = 1)
        dA_prev_skip = dZ
        
        # Main residual path: gradient flows backwards through all sub-layers
        dA_prev_main_residual_path = dZ
        for sublayer in reversed(self.sub_layers):
            dA_prev_main_residual_path = sublayer.backward_pass(dA_prev_main_residual_path)
        
        # Combine gradients from both paths
        dA_prev = dA_prev_main_residual_path + dA_prev_skip
        
        return dA_prev