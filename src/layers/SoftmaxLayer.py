"""
SoftmaxLayer - Documented Version

This layer implements the softmax activation function, used primarily as the
output layer for multi-class classification problems.

Softmax converts a vector of arbitrary real values into a probability distribution:
    - All outputs sum to 1
    - Each output is between 0 and 1
    - Can be interpreted as class probabilities

Mathematical definition:
    For input vector z = [z₁, z₂, ..., zₖ]:
    
    softmax(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)

The layer also includes integrated cross-entropy loss computation in the
backward pass, which simplifies to: dZ = (Y_hat - Y_true) / m

This is the standard approach for multi-class classification output layers.
"""

from src.layers import Layer
from src.utils.activations import get_activation
import numpy as np


class SoftmaxLayer(Layer):
    """
    Softmax Activation Layer (with integrated cross-entropy)
    
    Applies the softmax activation function to convert logits into a probability
    distribution over classes. Typically used as the final layer in multi-class
    classification networks.
    
    Mathematical properties:
        Forward:  A = softmax(Z)
                  where softmax(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)
                  
                  Properties of output:
                  - Each element in (0, 1)
                  - Sum of all elements = 1
                  - Differentiable everywhere
        
        Backward: dZ = (Y_hat - Y_true) / m
                  This elegant form combines:
                  - Softmax derivative
                  - Cross-entropy loss derivative
                  Resulting in a simple difference between predictions and targets
    
    Key characteristics:
        - Converts any real-valued vector into probability distribution
        - Preserves relative ordering of inputs (larger input → larger output)
        - Sensitive to outliers (exponential amplification)
        - Temperature parameter can control "sharpness" (not implemented here)
    
    Advantages:
        - Natural probability interpretation
        - Differentiable (enables gradient descent)
        - Combined with cross-entropy, gives stable gradients
        - Theoretically grounded (maximum entropy)
    
    Disadvantages:
        - Computationally expensive (multiple exponentials)
        - Can suffer from numerical instability (addressed by implementation)
        - Always assigns non-zero probability to all classes
    
    Common uses:
        - Multi-class classification output layer
        - Attention mechanisms in transformers
        - Policy networks in reinforcement learning
    
    IMPORTANT: This implementation combines softmax activation with cross-entropy
    loss gradient computation in backward_pass. The backward_pass expects true
    labels (Y_true) rather than upstream gradients (dA), which is non-standard
    but optimal for classification.
    
    Attributes:
        name (str): Layer identifier
        act_func (callable): Softmax activation function
                             Signature: (Z: np.ndarray) -> np.ndarray
                             Returns probability distribution
        params (dict): Empty - no learnable parameters
        cache (dict): Cached values for backward pass
            - Z (np.ndarray): Linear input (pre-activation logits)
        grads (dict): Empty - gradients not stored separately
    """
    
    def __init__(self, name: str) -> None:
        """
        Initialize the Softmax Activation Layer.
        
        Retrieves the softmax activation function from the activation utilities.
        Note: Derivative is not retrieved here because backward_pass uses the
        combined softmax + cross-entropy derivative formula.
        
        Parameters:
            name (str): Layer identifier
            
        Returns:
            None
        """
        super().__init__(name)
        
        # Get softmax activation function
        # Note: get_activation returns (function, derivative), but we only use function
        # Derivative is not used because backward_pass combines softmax + cross-entropy
        self.act_func = get_activation("softmax")[0]
    
    def initialize_parameters(self, input_dim: int | tuple, seed: int = 0) -> int | tuple:
        """
        Initialize parameters (no-op for activation layers).
        
        The softmax activation layer has no learnable parameters to initialize.
        The input dimension passes through unchanged since softmax activation
        doesn't alter the shape or size of the data.
        
        For multi-class classification:
            input_dim = number of classes
        
        Parameters:
            input_dim (int | tuple): Input dimensions
                                     Typically int (number of classes)
                                     For example, 10 for MNIST (10 digits)
            seed (int, optional): Random seed (not used, included for API consistency)
                                 Default: 0
            
        Returns:
            int | tuple: Output dimensions (same as input_dim)
                        Softmax preserves shape
                        
        Example:
            >>> softmax = SoftmaxLayer("output")
            >>> output_dim = softmax.initialize_parameters(input_dim=10)
            >>> print(output_dim)  # 10 (same as input)
        """
        return input_dim
    
    def forward_pass(self, Z: np.ndarray) -> np.ndarray:
        """
        Perform forward pass: apply softmax activation.
        
        For softmax activation: A = softmax(Z)
        where softmax(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)
        
        The function converts logits (unnormalized scores) into a probability
        distribution. Each column (sample) is normalized independently.
        
        Numerical stability:
            Implementations typically subtract max(z) before computing exponentials
            to prevent overflow: softmax(z) = softmax(z - max(z))
            This is mathematically equivalent but numerically stable.
        
        Steps:
            1. Cache Z for the backward pass
            2. Apply softmax transformation
            3. Return probability distribution
        
        Parameters:
            Z (np.ndarray): Logits (unnormalized scores) from previous layer
                           Shape: (num_classes, batch_size)
                           Each column represents one sample's class scores
                           
        Returns:
            np.ndarray: Probability distribution A = softmax(Z)
                       Shape: (num_classes, batch_size)
                       Each column sums to 1
                       All values in range (0, 1)
                       
        Example:
            >>> Z = np.array([[2.0, 1.0, 0.1],    # 3 classes
            ...               [1.0, 0.0, 2.0],     # 2 samples
            ...               [0.5, 3.0, 0.5]])
            >>> softmax = SoftmaxLayer("output")
            >>> A = softmax.forward_pass(Z)
            >>> print(A)
            # Each column sums to 1.0:
            [[0.659, 0.090, 0.112],
             [0.242, 0.033, 0.731],
             [0.099, 0.877, 0.157]]
            >>> print(A.sum(axis=0))  # [1.0, 1.0, 1.0]
        """
        # Cache input for backward pass
        self.cache["Z"] = Z
        
        # Apply softmax activation: A = softmax(Z)
        # Each column becomes a probability distribution
        A = self.act_func(Z)
        
        return A
    
    def backward_pass(self, Y_true: np.ndarray) -> np.ndarray:
        """
        Perform backward pass: compute gradient of softmax + cross-entropy loss.
        
        IMPORTANT: This method has a non-standard signature!
        Unlike other activation layers, it takes Y_true (true labels) instead of
        dA (upstream gradient). This is because softmax is typically the output
        layer combined with cross-entropy loss.
        
        Mathematical derivation:
            Loss: L = -Σᵢ yᵢ log(ŷᵢ)  (cross-entropy)
            
            Where:
                yᵢ = true probability (one-hot encoded)
                ŷᵢ = predicted probability (softmax output)
            
            Derivative: ∂L/∂zⱼ = ŷⱼ - yⱼ
            
            This remarkably simple result is why softmax + cross-entropy
            is the standard choice for multi-class classification.
        
        The gradient is averaged over the batch by dividing by m.
        
        Properties of this gradient:
            - Positive when prediction is too high (overconfident wrong class)
            - Negative when prediction is too low (underconfident correct class)
            - Zero when prediction matches target exactly
            - Magnitude proportional to prediction error
        
        Parameters:
            Y_true (np.ndarray): True labels (one-hot encoded)
                                Shape: (num_classes, batch_size)
                                Each column has one 1 and rest 0s
                                Example: [[0], [1], [0]] for class 1
                                
        Returns:
            np.ndarray: Gradient of loss with respect to logits (dZ)
                       Shape: (num_classes, batch_size)
                       dZ = (Y_hat - Y_true) / m
                       Averaged over batch for stable optimization
                       
        Example:
            >>> # True labels: sample 1 is class 0, sample 2 is class 2
            >>> Y_true = np.array([[1, 0],
            ...                    [0, 0],
            ...                    [0, 1]])
            >>> # Predictions from forward pass
            >>> # Y_hat = softmax(Z) computed internally
            >>> dZ = softmax.backward_pass(Y_true)
            >>> print(dZ.shape)  # (3, 2) - same as Y_true
            >>> # If predictions were perfect, dZ would be zero
            >>> # Non-zero dZ indicates prediction error
        """
        # Retrieve cached logits
        Z = self.cache["Z"]
        
        # Compute predictions (probability distribution)
        Y_hat = self.act_func(Z)
        
        # Get batch size
        m = Y_true.shape[1]
        
        # Compute gradient: dZ = (predictions - targets) / batch_size
        # This combines:
        #   - Softmax Jacobian matrix multiplication
        #   - Cross-entropy loss derivative
        # Resulting in this elegant, simple formula
        dZ = (Y_hat - Y_true) / m
        
        return dZ