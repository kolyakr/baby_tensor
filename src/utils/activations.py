"""
Activation Functions - Documented Version

This module provides activation functions and their derivatives for neural networks.
Activation functions introduce non-linearity, enabling neural networks to learn
complex patterns beyond linear relationships.

Available activations:
    - ReLU: Most common for hidden layers (fast, mitigates vanishing gradients)
    - Sigmoid: Binary classification, gates in RNNs (suffers from vanishing gradients)
    - Softmax: Multi-class classification output (probability distribution)
    - Linear: Regression output, pass-through (identity function)

Each activation (except softmax) comes with its derivative for backpropagation.
Softmax derivative is integrated with cross-entropy loss for numerical stability.
"""

import numpy as np
from typing import Callable, Literal, Union

# Type aliases for clarity and type safety
ActivationType = Literal["relu", "sigmoid", "softmax", "linear"]
ActFunc = Callable[[np.ndarray], np.ndarray]


# ============================================================================
# ReLU ACTIVATION
# ============================================================================

def relu(z: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU) activation function.
    
    The most popular activation function for hidden layers in deep networks.
    
    Mathematical definition:
        f(x) = max(0, x) = { x  if x > 0
                           { 0  if x ≤ 0
    
    Properties:
        - Non-linear (enables learning complex patterns)
        - Sparse activation (~50% of neurons are zero)
        - No vanishing gradient problem for positive values
        - Computationally efficient (simple comparison and clipping)
        - Unbounded for positive values (can grow without limit)
    
    Advantages:
        - Fast computation (no exponentials)
        - Reduces vanishing gradient problem
        - Induces sparsity (biological plausibility)
        - Works well in practice for most deep networks
    
    Disadvantages:
        - "Dying ReLU": neurons can permanently output 0 if weights push inputs negative
        - Not zero-centered (all outputs ≥ 0)
        - Not differentiable at x = 0 (but works in practice)
    
    Args:
        z (np.ndarray): Pre-activation input (linear combination from previous layer)
                       Shape: Any shape, typically (output_dim, batch_size) or
                              (batch_size, channels, height, width) for CNNs
    
    Returns:
        np.ndarray: Activated output, same shape as z
                   All negative values replaced with 0, positive values unchanged
    
    Example:
        >>> z = np.array([[-1.5, 2.0], [0.5, -0.3]])
        >>> relu(z)
        array([[0. , 2. ],
               [0.5, 0. ]])
    """
    return z.clip(0.0,)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of the ReLU function.
    
    Used during backpropagation to compute gradients.
    
    Mathematical definition:
        f'(x) = { 1  if x > 0
                { 0  if x ≤ 0
    
    Note: Technically undefined at x = 0, but we approximate as 0 for
    computational purposes. In practice, this works well.
    
    The derivative creates a binary mask: gradients flow through where
    inputs were positive, blocked where inputs were negative or zero.
    
    Args:
        z (np.ndarray): Pre-activation input (same as used in forward pass)
                       Shape: Same shape as forward pass input
    
    Returns:
        np.ndarray: Derivative values (0 or 1)
                   Same shape as z
                   1 where z > 0, 0 elsewhere
    
    Example:
        >>> z = np.array([[-1.5, 2.0], [0.5, -0.3]])
        >>> relu_derivative(z)
        array([[0., 1.],
               [1., 0.]])
    """
    return (z > 0).astype(float)


# ============================================================================
# SIGMOID ACTIVATION
# ============================================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid (Logistic) activation function.
    
    Maps any real-valued input to range (0, 1), making it suitable for
    probability interpretation.
    
    Mathematical definition:
        f(x) = 1 / (1 + e^(-x)) = σ(x)
    
    Properties:
        - Output range: (0, 1)
        - S-shaped curve
        - Smooth and differentiable everywhere
        - Monotonically increasing
        - Symmetric around 0.5: σ(-x) = 1 - σ(x)
    
    Advantages:
        - Clear probability interpretation
        - Smooth gradients (always differentiable)
        - Historically important and well-understood
        - Good for binary classification output
    
    Disadvantages:
        - Vanishing gradient problem for |x| >> 0
          (gradient approaches 0 for extreme values)
        - Not zero-centered (outputs always positive)
        - Computationally expensive (exponential operation)
        - Saturates for large |x|, causing slow learning
    
    Common uses:
        - Binary classification output layer
        - Gates in LSTM/GRU networks
        - Less common in hidden layers (ReLU preferred)
    
    Args:
        z (np.ndarray): Pre-activation input
                       Shape: Any shape
    
    Returns:
        np.ndarray: Activated output in range (0, 1)
                   Same shape as z
                   Interpreted as probabilities
    
    Example:
        >>> z = np.array([[-2.0, 0.0], [2.0, 5.0]])
        >>> sigmoid(z)
        array([[0.1192, 0.5000],
               [0.8808, 0.9933]])
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of the sigmoid function.
    
    Used during backpropagation to compute gradients.
    
    Mathematical definition:
        f'(x) = σ(x) * (1 - σ(x))
    
    This elegant form allows computing the derivative using the
    forward pass output, which is computationally efficient.
    
    Properties:
        - Maximum at x = 0: f'(0) = 0.25
        - Approaches 0 as |x| → ∞ (vanishing gradient)
        - Symmetric around x = 0
    
    The vanishing gradient problem occurs because:
        - When |z| is large, σ(z) ≈ 0 or ≈ 1
        - Then σ(z) * (1 - σ(z)) ≈ 0
        - This makes gradients very small, slowing learning
    
    Args:
        z (np.ndarray): Pre-activation input (same as forward pass)
                       Shape: Same shape as forward pass input
    
    Returns:
        np.ndarray: Derivative values in range (0, 0.25]
                   Same shape as z
                   Maximum value 0.25 at z = 0
                   Approaches 0 for extreme values
    
    Example:
        >>> z = np.array([[-2.0, 0.0], [2.0, 5.0]])
        >>> sigmoid_derivative(z)
        array([[0.1050, 0.2500],
               [0.1050, 0.0066]])
        >>> # Note: very small gradient at z=5.0
    """
    return sigmoid(z) * (1 - sigmoid(z))


# ============================================================================
# SOFTMAX ACTIVATION
# ============================================================================

def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax activation function.
    
    Converts a vector of arbitrary real values into a probability distribution.
    Used as the output layer for multi-class classification.
    
    Mathematical definition:
        For input vector z = [z₁, z₂, ..., zₖ]:
        softmax(z)ᵢ = e^(zᵢ) / Σⱼ e^(zⱼ)
    
    Properties:
        - Output values in range (0, 1)
        - Sum of all outputs = 1 (valid probability distribution)
        - Preserves relative ordering of inputs
        - Differentiable everywhere
        - "Soft" version of argmax (hence the name)
    
    Numerical stability:
        The implementation uses the "max subtraction trick":
            softmax(z) = softmax(z - max(z))
        This prevents overflow from large exponentials without changing
        the result mathematically.
    
    Advantages:
        - Natural probability interpretation
        - Theoretically grounded (maximum entropy)
        - Combined with cross-entropy, gives simple gradients
        - Standard for multi-class classification
    
    Disadvantages:
        - Computationally expensive (multiple exponentials)
        - Always assigns non-zero probability to all classes
        - Can suffer from numerical instability without careful implementation
    
    Common uses:
        - Multi-class classification output layer
        - Attention mechanisms in transformers
        - Policy networks in reinforcement learning
    
    Implementation note:
        Assumes input shape (num_classes, batch_size) based on your architecture.
        Normalization is performed along axis=0 (across classes for each sample).
    
    Args:
        z (np.ndarray): Pre-activation logits (unnormalized scores)
                       Shape: (num_classes, batch_size)
                       Each column represents one sample's class scores
    
    Returns:
        np.ndarray: Probability distribution
                   Shape: (num_classes, batch_size)
                   Each column sums to 1
                   All values in range (0, 1)
    
    Example:
        >>> z = np.array([[2.0, 1.0],
        ...               [1.0, 0.0],
        ...               [0.5, 3.0]])
        >>> softmax(z)
        array([[0.659, 0.090],
               [0.242, 0.033],
               [0.099, 0.877]])
        >>> # Verify: each column sums to 1.0
        >>> softmax(z).sum(axis=0)
        array([1., 1.])
    """
    # Numerical stability: subtract max to prevent overflow
    # axis=0 because shape is (Classes, Samples)
    z_max = np.max(z, axis=0, keepdims=True)
    z_stable = z - z_max
    
    # Compute exponentials
    exp_z = np.exp(z_stable)
    
    # Normalize to get probability distribution
    # Important: keepdims=True ensures proper broadcasting
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


# ============================================================================
# LINEAR ACTIVATION
# ============================================================================

def linear(z: np.ndarray) -> np.ndarray:
    """
    Linear (Identity) activation function.
    
    Returns input unchanged. Used for regression output layers where
    predictions can be any real value.
    
    Mathematical definition:
        f(x) = x
    
    Properties:
        - Output range: (-∞, +∞)
        - Linear (no non-linearity introduced)
        - Derivative is constant (1 everywhere)
    
    Use cases:
        - Regression output layer (predicting continuous values)
        - Pass-through layer (no activation desired)
        - Debugging/testing
    
    Note: Using only linear activations throughout a network would
    reduce it to a simple linear model, as composition of linear
    functions is still linear. Non-linear activations (ReLU, sigmoid)
    are needed in hidden layers to learn complex patterns.
    
    Args:
        z (np.ndarray): Pre-activation input
                       Shape: Any shape
    
    Returns:
        np.ndarray: Output identical to input
                   Same shape as z
    
    Example:
        >>> z = np.array([[-1.5, 2.0], [0.5, -0.3]])
        >>> linear(z)
        array([[-1.5,  2. ],
               [ 0.5, -0.3]])
    """
    return z


def linear_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of the Linear (Identity) function.
    
    The derivative of f(x) = x is f'(x) = 1 everywhere.
    
    Mathematical definition:
        f'(x) = 1
    
    This means gradients pass through unchanged during backpropagation.
    No gradient scaling or blocking occurs.
    
    Args:
        z (np.ndarray): Pre-activation input (same as forward pass)
                       Shape: Same shape as forward pass input
    
    Returns:
        np.ndarray: Array of ones
                   Same shape as z
                   All values are 1.0
    
    Example:
        >>> z = np.array([[-1.5, 2.0], [0.5, -0.3]])
        >>> linear_derivative(z)
        array([[1., 1.],
               [1., 1.]])
    """
    return np.ones_like(z)


# ============================================================================
# ACTIVATION FACTORY
# ============================================================================

def get_activation(name: ActivationType) -> Union[tuple[ActFunc, ActFunc], ActFunc]:
    """
    Factory function to retrieve activation function and its derivative.
    
    Provides a clean interface for dynamically selecting activation functions
    based on string identifiers. Used throughout the neural network code to
    configure layers.
    
    Return behavior:
        - For hidden layer activations (relu, sigmoid, linear):
          Returns tuple (activation_func, derivative_func)
        - For output layer activation (softmax):
          Returns only activation_func (no derivative)
          
    Softmax derivative is not returned because it's integrated with
    cross-entropy loss for numerical stability and simplicity:
        d(softmax + cross_entropy)/dz = predictions - targets
    
    Args:
        name (ActivationType): Activation identifier
                              Valid values: "relu", "sigmoid", "softmax", "linear"
    
    Returns:
        Union[tuple[ActFunc, ActFunc], ActFunc]:
            - For "relu", "sigmoid", "linear": (function, derivative)
            - For "softmax": function only
    
    Raises:
        ValueError: If name is not one of the valid activation types
    
    Example:
        >>> # Get ReLU activation and derivative
        >>> act_func, act_deriv = get_activation("relu")
        >>> z = np.array([[-1, 2], [3, -0.5]])
        >>> a = act_func(z)
        >>> grad = act_deriv(z)
        >>> 
        >>> # Get softmax (no derivative)
        >>> softmax_func, _ = get_activation("softmax")
        >>> # or simply:
        >>> softmax_func = get_activation("softmax")[0]
        >>> 
        >>> # Invalid activation
        >>> get_activation("tanh")  # Raises ValueError
    
    Usage in layers:
        >>> class ReLULayer(Layer):
        ...     def __init__(self, name):
        ...         super().__init__(name)
        ...         self.act_func = get_activation("relu")[0]
        ...         self.act_deriv = get_activation("relu")[1]
        ...     
        ...     def forward_pass(self, Z):
        ...         return self.act_func(Z)
        ...     
        ...     def backward_pass(self, dA):
        ...         return dA * self.act_deriv(self.cache["Z"])
    """
    if name == "relu":
        return (relu, relu_derivative)
    if name == "sigmoid":
        return (sigmoid, sigmoid_derivative)
    if name == "softmax":
        return (softmax, None)
    if name == "linear":
        return (linear, linear_derivative)
    raise ValueError(f"Unknown activation: {name}")