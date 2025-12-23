"""
SigmoidLayer - Documented Version

This layer implements the sigmoid (logistic) activation function.
Sigmoid squashes input values to the range (0, 1), making it useful for:
    1. Binary classification output layers (probability interpretation)
    2. Gates in LSTM/GRU networks
    3. Any scenario requiring output bounded between 0 and 1

Mathematical definition:
    f(x) = 1 / (1 + e^(-x)) = σ(x)
    
    f'(x) = σ(x) * (1 - σ(x))

Properties:
    - Output range: (0, 1)
    - S-shaped curve
    - Smooth and differentiable everywhere
    - Can be interpreted as probability
"""

from src.layers import Layer
from src.utils.activations import get_activation
import numpy as np


class SigmoidLayer(Layer):
    """
    Sigmoid Activation Layer
    
    Applies the sigmoid (logistic) activation function element-wise: f(x) = 1/(1 + e^(-x))
    
    Mathematical properties:
        Forward:  A = σ(Z) = 1 / (1 + e^(-Z))
                  - Maps input to range (0, 1)
                  - Output can be interpreted as probability
        
        Backward: dZ = dA * σ'(Z)
                  where σ'(Z) = σ(Z) * (1 - σ(Z))
    
    Characteristics:
        - Smooth, continuous, differentiable
        - Monotonically increasing
        - Symmetric around 0.5 (σ(0) = 0.5)
        - Asymptotic behavior: σ(-∞) → 0, σ(+∞) → 1
    
    Advantages:
        - Clear probability interpretation (0 to 1 range)
        - Smooth gradients
        - Historically important activation function
    
    Disadvantages:
        - Vanishing gradient problem for extreme values (|x| >> 0)
        - Not zero-centered (outputs always positive)
        - Computationally expensive (exponential operation)
        - Saturates for large |x|, causing slow learning
    
    Common uses:
        - Binary classification output layer
        - Gates in recurrent neural networks (LSTM/GRU)
        - Less common in hidden layers (ReLU preferred for deep networks)
    
    This layer has no learnable parameters - it only applies an activation function.
    
    Attributes:
        name (str): Layer identifier
        act_func (callable): Sigmoid activation function
                             Signature: (Z: np.ndarray) -> np.ndarray
                             Returns 1/(1 + exp(-Z))
        act_deriv (callable): Derivative of sigmoid activation
                              Signature: (Z: np.ndarray) -> np.ndarray
                              Returns σ(Z) * (1 - σ(Z))
        params (dict): Empty - no learnable parameters
        cache (dict): Cached values for backward pass
            - Z (np.ndarray): Linear input (pre-activation)
        grads (dict): Cached gradients
            - dA (np.ndarray): Gradient w.r.t. activation output
    """
    
    def __init__(self, name: str) -> None:
        """
        Initialize the Sigmoid Activation Layer.
        
        Retrieves the sigmoid activation function and its derivative from
        the activation utilities.
        
        Parameters:
            name (str): Layer identifier
            
        Returns:
            None
        """
        super().__init__(name)
        
        # Get sigmoid activation function and its derivative
        # get_activation returns (function, derivative)
        self.act_func = get_activation("sigmoid")[0]
        self.act_deriv = get_activation("sigmoid")[1]
    
    def initialize_parameters(self, input_dim: int | tuple, seed: int = 0) -> int | tuple:
        """
        Initialize parameters (no-op for activation layers).
        
        The sigmoid activation layer has no learnable parameters to initialize.
        The input dimension passes through unchanged since sigmoid activation
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
                        Sigmoid activation preserves shape
        """
        return input_dim
    
    def forward_pass(self, Z: np.ndarray) -> np.ndarray:
        """
        Perform forward pass: apply sigmoid activation.
        
        For sigmoid activation: A = σ(Z) = 1 / (1 + e^(-Z))
        
        The function squashes any real-valued input to the range (0, 1):
            - Large positive values → close to 1
            - Large negative values → close to 0
            - Z = 0 → exactly 0.5
        
        This transformation is smooth and differentiable, making it suitable
        for gradient-based optimization.
        
        Steps:
            1. Cache Z for the backward pass (needed to compute derivative)
            2. Apply sigmoid: compute 1/(1 + exp(-Z))
            3. Return activated output
        
        Parameters:
            Z (np.ndarray): Linear input from previous layer (pre-activation values)
                           Shape can be:
                           - (output_dim, batch_size) for dense layers
                           - (batch_size, channels, height, width) for conv layers
                           
        Returns:
            np.ndarray: Activated output A = σ(Z)
                       Same shape as input Z
                       All values in range (0, 1)
                       
        Example:
            >>> Z = np.array([[2.0, -2.0], [0.0, 5.0]])
            >>> sigmoid = SigmoidLayer("sigmoid1")
            >>> A = sigmoid.forward_pass(Z)
            >>> print(A)
            [[0.8808, 0.1192], [0.5000, 0.9933]]  # All values between 0 and 1
        """
        # Cache input for backward pass (needed to compute derivative)
        self.cache["Z"] = Z
        
        # Apply sigmoid activation: A = 1 / (1 + e^(-Z))
        A = self.act_func(Z)
        
        return A
    
    def backward_pass(self, dA: np.ndarray) -> np.ndarray:
        """
        Perform backward pass: compute gradient through sigmoid activation.
        
        For sigmoid activation:
            f(x) = σ(x) = 1 / (1 + e^(-x))
            f'(x) = σ(x) * (1 - σ(x))
        
        This derivative has an elegant form that can be computed using the
        forward pass output. The derivative is maximum at x=0 (where σ(0)=0.5),
        giving f'(0) = 0.25, and approaches 0 for large |x|.
        
        Therefore: dZ = dA * σ(Z) * (1 - σ(Z))
        
        The vanishing gradient problem occurs because:
            - When |Z| is large, σ(Z) approaches 0 or 1
            - This makes σ(Z) * (1 - σ(Z)) very small
            - Small gradients slow down or stop learning
        
        This is why sigmoid is rarely used in hidden layers of deep networks
        (ReLU doesn't have this problem).
        
        Parameters:
            dA (np.ndarray): Gradient of loss with respect to activation output
                            Shape matches the output from forward_pass
                            
        Returns:
            np.ndarray: Gradient of loss with respect to linear input (dZ)
                       dZ = dA * σ(Z) * (1 - σ(Z))
                       Same shape as dA
                       Gradient is small for extreme values of Z
                       
        Example:
            >>> # Forward pass had Z = [[2.0, -2.0], [0.0, 5.0]]
            >>> # σ(Z) = [[0.88, 0.12], [0.50, 0.99]]
            >>> # σ'(Z) = [[0.10, 0.10], [0.25, 0.01]] (approximately)
            >>> dA = np.array([[1.0, 1.0], [1.0, 1.0]])
            >>> dZ = sigmoid.backward_pass(dA)
            >>> print(dZ)
            [[0.10, 0.10], [0.25, 0.01]]  # Small gradients at extremes
        """
        # Cache dA (optional, for debugging/analysis)
        self.grads["dA"] = dA
        
        # Compute gradient: dZ = dA * derivative(Z)
        # derivative(Z) = σ(Z) * (1 - σ(Z))
        dZ = dA * self.act_deriv(self.cache["Z"])
        
        return dZ