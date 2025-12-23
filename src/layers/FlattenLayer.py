"""
FlattenLayer - Documented Version

This layer flattens multi-dimensional convolutional feature maps into a 1D vector
for each sample in the batch. This is typically used as a transition layer between
convolutional layers and dense (fully connected) layers.

Example transformation:
    Input:  (batch_size, channels, height, width)
    Output: (channels * height * width, batch_size)
"""

from src.layers import Layer
import numpy as np


class FlattenLayer(Layer):
    """
    Flatten Layer
    
    Reshapes multi-dimensional input (typically from convolutional layers)
    into a 2D matrix suitable for dense layers. Each sample's spatial features
    are flattened into a single vector.
    
    This layer has no learnable parameters - it only performs reshaping operations.
    
    Transformation:
        Forward:  (B, C, H, W) → (C*H*W, B)
        Backward: (C*H*W, B) → (B, C, H, W)
    
    where:
        B = batch size
        C = number of channels
        H = height
        W = width
    
    Attributes:
        name (str): Layer identifier
        params (dict): Empty - no learnable parameters
        cache (dict): Cached values for backward pass
            - A_prev (np.ndarray): Original input shape before flattening
        grads (dict): Empty - no learnable parameters to compute gradients for
    """
    
    def __init__(self, name: str) -> None:
        """
        Initialize the Flatten Layer.
        
        Parameters:
            name (str): Layer identifier
            
        Returns:
            None
        """
        super().__init__(name)
    
    def initialize_parameters(self, input_shape: tuple, seed: int = 0) -> int:
        """
        Calculate output dimension after flattening.
        
        No actual parameters to initialize since this is a reshaping layer.
        Simply computes the flattened dimension size.
        
        Parameters:
            input_shape (tuple): Input dimensions (channels, height, width)
                Format: (C, H, W)
            seed (int, optional): Random seed (not used, included for API consistency)
                                 Default: 0
            
        Returns:
            int: Output dimension after flattening (C * H * W)
                 This value is used as input_dim for the next layer
        """
        np.random.seed(seed)
        
        c, h, w = input_shape
        
        # Calculate total number of features after flattening
        output_dim = c * h * w
        
        return output_dim
    
    def forward_pass(self, A_prev: np.ndarray) -> np.ndarray:
        """
        Perform forward pass: flatten spatial dimensions.
        
        Reshapes 4D convolutional feature maps into 2D matrix where each column
        represents one flattened sample.
        
        The transformation preserves all information - no data is lost, only
        the shape changes.
        
        Steps:
            1. Cache original shape (needed for backward pass)
            2. Reshape from (B, C, H, W) to (B, C*H*W)
            3. Transpose to (C*H*W, B) for compatibility with dense layers
        
        Parameters:
            A_prev (np.ndarray): Input feature maps from convolutional layer
                Shape: (batch_size, channels, height, width)
                Example: (32, 64, 7, 7) for 32 samples, 64 feature maps of size 7×7
                
        Returns:
            np.ndarray: Flattened output
                Shape: (channels * height * width, batch_size)
                Example: (3136, 32) for the above input
                Each column is one flattened sample
        """
        # Cache original shape for backward pass
        self.cache["A_prev"] = A_prev
        
        b, c, h, w = A_prev.shape
        
        # Reshape: (B, C, H, W) → (B, C*H*W) → transpose → (C*H*W, B)
        A = A_prev.reshape(b, -1).T
        
        return A
    
    def backward_pass(self, dA: np.ndarray) -> np.ndarray:
        """
        Perform backward pass: unflatten gradient back to original shape.
        
        This is the inverse operation of forward_pass. The gradient flows
        backwards through the reshape operation, restoring the original
        multi-dimensional structure.
        
        Since reshaping is a linear operation with no parameters, the backward
        pass simply reshapes the gradient to match the input shape.
        
        Steps:
            1. Retrieve original input shape from cache
            2. Transpose from (C*H*W, B) to (B, C*H*W)
            3. Reshape to (B, C, H, W)
        
        Parameters:
            dA (np.ndarray): Gradient of loss with respect to flattened output
                Shape: (channels * height * width, batch_size)
                
        Returns:
            np.ndarray: Gradient of loss with respect to input (dA_prev)
                Shape: (batch_size, channels, height, width)
                Same shape as the original input to forward_pass
        """
        # Retrieve original shape from cache
        b, c, h, w = self.cache["A_prev"].shape
        
        # Reshape gradient back to original multi-dimensional format
        # (C*H*W, B) → transpose → (B, C*H*W) → reshape → (B, C, H, W)
        dA_prev = dA.T.reshape(b, c, h, w)
        
        return dA_prev