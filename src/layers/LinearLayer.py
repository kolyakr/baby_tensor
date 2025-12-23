"""
LinearLayer - Documented Version

This layer implements the linear (identity) activation function.
The linear activation simply returns its input unchanged: f(x) = x

This is typically used:
    1. In output layers for regression tasks (predicting continuous values)
    2. As a pass-through layer when no activation is desired
    3. For debugging or architectural experimentation

Note: This is an activation layer, not a dense (fully connected) layer.
"""

from src.layers import Layer
from src.utils.activations import get_activation
import numpy as np


class LinearLayer(Layer):
    """
    Linear Activation Layer
    
    Applies the linear (identity) activation function: f(x) = x
    
    Mathematical definition:
        Forward:  A = Z  (output equals input)
        Backward: dZ = dA * 1 = dA  (gradient passes through unchanged)
    
    This layer has no learnable parameters - it only applies an activation function.
    The linear activation is effectively a no-op, but is included for:
        - API consistency with other activation layers
        - Explicit specification in model architectures
        - Flexibility in model design
    
    Attributes:
        name (str): Layer identifier
        act_func (callable): Linear activation function
                             Signature: (Z: np.ndarray) -> np.ndarray
                             Returns Z unchanged
        act_deriv (callable): Derivative of linear activation
                              Signature: (Z: np.ndarray) -> np.ndarray
                              Returns ones (gradient = 1 everywhere)
        params (dict): Empty - no learnable parameters
        cache (dict): Cached values for backward pass
            - Z (np.ndarray): Linear input (pre-activation)
        grads (dict): Cached gradients
            - dA (np.ndarray): Gradient w.r.t. activation output
    """
    
    def __init__(self, name: str) -> None:
        """
        Initialize the Linear Activation Layer.
        
        Retrieves the linear activation function and its derivative from
        the activation utilities.
        
        Parameters:
            name (str): Layer identifier
            
        Returns:
            None
        """
        super().__init__(name)
        
        # Get linear activation function and its derivative
        # get_activation returns (function, derivative)
        self.act_func = get_activation("linear")[0]
        self.act_deriv = get_activation("linear")[1]
    
    def initialize_parameters(self, input_dim: int | tuple, seed: int = 0) -> int | tuple:
        """
        Initialize parameters (no-op for activation layers).
        
        The linear activation layer has no learnable parameters to initialize.
        The input dimension passes through unchanged since linear activation
        doesn't alter the shape or size of the data.
        
        Parameters:
            input_dim (int | tuple): Input dimensions
                                     - int: for 1D data (features, batch_size)
                                     - tuple: for multi-dimensional data
                                       (batch_size, channels, height, width)
            seed (int, optional): Random seed (not used, included for API consistency)
                                 Default: 0
            
        Returns:
            int | tuple: Output dimensions (same as input_dim)
                        Linear activation preserves shape
        """
        return input_dim
    
    def forward_pass(self, Z: np.ndarray) -> np.ndarray:
        """
        Perform forward pass: apply linear activation.
        
        For linear activation: A = f(Z) = Z
        The output is identical to the input - the function is the identity.
        
        Despite being a no-op mathematically, this layer still:
            1. Caches Z for the backward pass
            2. Applies the activation function (for consistency)
            3. Returns the activated output
        
        Parameters:
            Z (np.ndarray): Linear input from previous layer (pre-activation values)
                           Shape can be:
                           - (output_dim, batch_size) for dense layers
                           - (batch_size, channels, height, width) for conv layers
                           
        Returns:
            np.ndarray: Activated output A = Z
                       Same shape as input Z
                       
        Example:
            >>> Z = np.array([[1.5, -2.0], [3.0, 0.5]])
            >>> linear = LinearLayer("linear1")
            >>> A = linear.forward_pass(Z)
            >>> print(A)
            [[1.5, -2.0], [3.0, 0.5]]  # Unchanged
        """
        # Cache input for backward pass
        self.cache["Z"] = Z
        
        # Apply linear activation: A = Z
        A = self.act_func(Z)
        
        return A
    
    def backward_pass(self, dA: np.ndarray) -> np.ndarray:
        """
        Perform backward pass: compute gradient through linear activation.
        
        For linear activation: f(x) = x, so f'(x) = 1
        Therefore: dZ = dA * f'(Z) = dA * 1 = dA
        
        The gradient passes through unchanged since the derivative of the
        identity function is 1 everywhere.
        
        Parameters:
            dA (np.ndarray): Gradient of loss with respect to activation output
                            Shape matches the output from forward_pass
                            
        Returns:
            np.ndarray: Gradient of loss with respect to linear input (dZ)
                       dZ = dA (gradient unchanged)
                       Same shape as dA
                       
        Example:
            >>> dA = np.array([[0.5, -0.3], [0.2, 0.8]])
            >>> dZ = linear.backward_pass(dA)
            >>> print(dZ)
            [[0.5, -0.3], [0.2, 0.8]]  # Unchanged
        """
        # Cache dA (optional, for debugging/analysis)
        self.grads["dA"] = dA
        
        # Compute gradient: dZ = dA * derivative(Z)
        # For linear: derivative = 1, so dZ = dA
        dZ = dA * self.act_deriv(self.cache["Z"])
        
        return dZ