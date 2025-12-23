"""
ReLULayer - Documented Version

This layer implements the Rectified Linear Unit (ReLU) activation function.
ReLU is one of the most popular activation functions in deep learning due to:
    1. Computational efficiency (simple max operation)
    2. Reduced vanishing gradient problem
    3. Sparse activation (many zeros in output)
    4. Biological plausibility (similar to neural firing patterns)

Mathematical definition:
    f(x) = max(0, x) = { x  if x > 0
                       { 0  if x ≤ 0
    
    f'(x) = { 1  if x > 0
            { 0  if x ≤ 0
"""

from src.layers import Layer
from src.utils.activations import get_activation
import numpy as np


class ReLULayer(Layer):
    """
    ReLU (Rectified Linear Unit) Activation Layer
    
    Applies the ReLU activation function element-wise: f(x) = max(0, x)
    
    Mathematical properties:
        Forward:  A = max(0, Z)
                  - Outputs Z when Z > 0
                  - Outputs 0 when Z ≤ 0
        
        Backward: dZ = dA * f'(Z)
                  where f'(Z) = 1 if Z > 0, else 0
                  - Gradient passes through when Z > 0
                  - Gradient is blocked (set to 0) when Z ≤ 0
    
    Advantages:
        - Fast computation (no exponentials or divisions)
        - Mitigates vanishing gradient problem
        - Induces sparsity (approximately 50% of activations are zero)
        - Non-saturating for positive values
    
    Disadvantages:
        - "Dying ReLU" problem: neurons can get stuck outputting 0
        - Not zero-centered (can affect optimization)
        - Not differentiable at x = 0 (but works in practice)
    
    This layer has no learnable parameters - it only applies an activation function.
    
    Attributes:
        name (str): Layer identifier
        act_func (callable): ReLU activation function
                             Signature: (Z: np.ndarray) -> np.ndarray
                             Returns max(0, Z)
        act_deriv (callable): Derivative of ReLU activation
                              Signature: (Z: np.ndarray) -> np.ndarray
                              Returns 1 where Z > 0, else 0
        params (dict): Empty - no learnable parameters
        cache (dict): Cached values for backward pass
            - Z (np.ndarray): Linear input (pre-activation)
        grads (dict): Cached gradients
            - dA (np.ndarray): Gradient w.r.t. activation output
    """
    
    def __init__(self, name: str) -> None:
        """
        Initialize the ReLU Activation Layer.
        
        Retrieves the ReLU activation function and its derivative from
        the activation utilities.
        
        Parameters:
            name (str): Layer identifier
            
        Returns:
            None
        """
        super().__init__(name)
        
        # Get ReLU activation function and its derivative
        # get_activation returns (function, derivative)
        self.act_func = get_activation("relu")[0]
        self.act_deriv = get_activation("relu")[1]
    
    def initialize_parameters(self, input_dim: int | tuple, seed: int = 0) -> int | tuple:
        """
        Initialize parameters (no-op for activation layers).
        
        The ReLU activation layer has no learnable parameters to initialize.
        The input dimension passes through unchanged since ReLU activation
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
                        ReLU activation preserves shape
        """
        return input_dim
    
    def forward_pass(self, Z: np.ndarray) -> np.ndarray:
        """
        Perform forward pass: apply ReLU activation.
        
        For ReLU activation: A = max(0, Z)
        
        The function zeros out all negative values while keeping positive
        values unchanged. This introduces non-linearity and sparsity.
        
        Steps:
            1. Cache Z for the backward pass (needed to compute derivative)
            2. Apply ReLU: set negative values to 0, keep positive values
            3. Return activated output
        
        Parameters:
            Z (np.ndarray): Linear input from previous layer (pre-activation values)
                           Shape can be:
                           - (output_dim, batch_size) for dense layers
                           - (batch_size, channels, height, width) for conv layers
                           
        Returns:
            np.ndarray: Activated output A = max(0, Z)
                       Same shape as input Z
                       All negative values replaced with 0
                       
        Example:
            >>> Z = np.array([[1.5, -2.0], [3.0, -0.5]])
            >>> relu = ReLULayer("relu1")
            >>> A = relu.forward_pass(Z)
            >>> print(A)
            [[1.5, 0.0], [3.0, 0.0]]  # Negative values zeroed
        """
        # Cache input for backward pass (needed to determine where Z > 0)
        self.cache["Z"] = Z
        
        # Apply ReLU activation: A = max(0, Z)
        A = self.act_func(Z)
        
        return A
    
    def backward_pass(self, dA: np.ndarray) -> np.ndarray:
        """
        Perform backward pass: compute gradient through ReLU activation.
        
        For ReLU activation:
            f(x) = max(0, x)
            f'(x) = { 1  if x > 0
                    { 0  if x ≤ 0
        
        Therefore: dZ = dA * f'(Z)
        
        The gradient passes through unchanged where the input was positive,
        and is completely blocked (set to 0) where the input was negative or zero.
        This is the key mechanism that:
            - Allows learning for active neurons
            - Prevents learning for inactive neurons
            - Can lead to "dying ReLU" if too many neurons become inactive
        
        Parameters:
            dA (np.ndarray): Gradient of loss with respect to activation output
                            Shape matches the output from forward_pass
                            
        Returns:
            np.ndarray: Gradient of loss with respect to linear input (dZ)
                       dZ = dA * (1 if Z > 0 else 0)
                       Same shape as dA
                       Gradient is zero wherever Z ≤ 0
                       
        Example:
            >>> # Forward pass had Z = [[1.5, -2.0], [3.0, -0.5]]
            >>> # So derivative is [[1, 0], [1, 0]]
            >>> dA = np.array([[0.5, 0.3], [0.2, 0.8]])
            >>> dZ = relu.backward_pass(dA)
            >>> print(dZ)
            [[0.5, 0.0], [0.2, 0.0]]  # Gradient blocked where Z was negative
        """
        # Cache dA (optional, for debugging/analysis)
        self.grads["dA"] = dA
        
        # Compute gradient: dZ = dA * derivative(Z)
        # derivative(Z) = 1 where Z > 0, else 0
        dZ = dA * self.act_deriv(self.cache["Z"])
        
        return dZ