"""
Loss Functions - Documented Version

This module provides loss (cost) functions and their derivatives for training
neural networks. Loss functions quantify the difference between predictions
and true values, providing the optimization objective.

Available loss functions:
    - MSE (Mean Squared Error): Regression tasks
    - BCE (Binary Cross-Entropy): Binary classification
    - CCE (Categorical Cross-Entropy): Multi-class classification

Each loss function comes with its derivative for computing gradients during
backpropagation. The gradient of the loss is the starting point for the
backward pass through the network.

Key concepts:
    - Loss measures prediction error (lower is better)
    - Derivative indicates direction to adjust parameters
    - Choice of loss function depends on task type
    - Proper loss function is critical for training success
"""

import numpy as np
from typing import Literal, Callable

# Type aliases for clarity and type safety
LossType = Literal["mse", "bce", "cce"]
LossFunc = Callable[[np.ndarray, np.ndarray], float]
Derivative = Callable[[np.ndarray, np.ndarray], np.ndarray]


# ============================================================================
# MEAN SQUARED ERROR (MSE) - FOR REGRESSION
# ============================================================================

def mse(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    """
    Mean Squared Error (MSE) loss function.
    
    The standard loss function for regression tasks. Penalizes large errors
    more heavily than small errors due to the squaring operation.
    
    Mathematical formula:
        L = (1/N) * Σ(ŷ - y)²
    
    where:
        - N: number of samples
        - ŷ: predicted values
        - y: true values
    
    Properties:
        - Always non-negative (≥ 0)
        - Equals 0 only when predictions are perfect
        - Sensitive to outliers (due to squaring)
        - Differentiable everywhere
        - Convex (single global minimum)
    
    Advantages:
        - Simple and intuitive
        - Convex optimization landscape
        - Heavily penalizes large errors
        - Well-studied and understood
    
    Disadvantages:
        - Sensitive to outliers (large errors dominate)
        - Units are squared (not interpretable)
        - Not robust to extreme values
    
    Use cases:
        - Regression problems (predicting continuous values)
        - Output layer with linear activation
        - Examples: house price prediction, temperature forecasting
    
    Args:
        y_hat (np.ndarray): Predicted values from the network
                           Shape: (output_dim, batch_size)
                           Can be any real values
        y_true (np.ndarray): True target values
                            Shape: Same as y_hat
                            Ground truth continuous values
    
    Returns:
        float: Mean squared error (scalar)
              Range: [0, ∞)
              Lower is better, 0 is perfect
    
    Example:
        >>> y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> y_hat = np.array([[1.1, 2.2], [2.9, 3.8]])
        >>> mse(y_hat, y_true)
        0.015  # Small error
        >>> 
        >>> y_hat_bad = np.array([[5.0, 8.0], [1.0, 0.0]])
        >>> mse(y_hat_bad, y_true)
        17.5  # Large error
    """
    return np.mean(np.square(y_hat - y_true))


def mse_derivative(y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Derivative of Mean Squared Error with respect to predictions.
    
    Computes ∂L/∂ŷ, which indicates how to adjust predictions to reduce loss.
    
    Mathematical formula:
        ∂L/∂ŷ = (2/m) * (ŷ - y)
    
    where:
        - m: batch size
        - ŷ: predicted values
        - y: true values
    
    The factor of 2 comes from the chain rule (d/dx of x² is 2x).
    Division by m averages the gradient over the batch.
    
    Properties:
        - Positive when prediction is too high (ŷ > y)
        - Negative when prediction is too low (ŷ < y)
        - Zero when prediction is correct (ŷ = y)
        - Magnitude proportional to error size
    
    Args:
        y_hat (np.ndarray): Predicted values
                           Shape: (output_dim, batch_size)
        y_true (np.ndarray): True target values
                            Shape: Same as y_hat
    
    Returns:
        np.ndarray: Gradient of loss with respect to predictions
                   Shape: Same as y_hat
                   Used to start backpropagation
    
    Example:
        >>> y_true = np.array([[1.0, 2.0]])
        >>> y_hat = np.array([[1.5, 1.5]])  # First too high, second too low
        >>> mse_derivative(y_hat, y_true)
        array([[ 0.5, -0.25]])  # Positive: reduce, Negative: increase
    """
    m = y_hat.shape[1]  # Batch size (number of samples)
    return 2 * (y_hat - y_true) / m


# ============================================================================
# BINARY CROSS-ENTROPY (BCE) - FOR BINARY CLASSIFICATION
# ============================================================================

def binary_cross_entropy(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    """
    Binary Cross-Entropy (BCE) loss function.
    
    The standard loss function for binary classification (two classes).
    Measures the difference between two probability distributions.
    
    Mathematical formula:
        L = -(1/N) * Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
    
    where:
        - N: number of samples
        - y: true label (0 or 1)
        - ŷ: predicted probability (0 to 1)
    
    Interpretation:
        - When y=1: loss = -log(ŷ)
          Penalizes low probabilities for positive class
        - When y=0: loss = -log(1-ŷ)
          Penalizes high probabilities for negative class
    
    Properties:
        - Expects probabilities (ŷ ∈ [0, 1]) not raw logits
        - Always non-negative
        - Approaches 0 as predictions improve
        - Approaches ∞ as predictions worsen (ŷ→0 when y=1, or ŷ→1 when y=0)
    
    Advantages:
        - Probabilistic interpretation
        - Works well with sigmoid output
        - Heavily penalizes confident wrong predictions
        - Theoretically grounded (maximum likelihood)
    
    Disadvantages:
        - Numerically unstable without clipping (log(0) = -∞)
        - Not robust to label noise
        - Sensitive to class imbalance
    
    Use cases:
        - Binary classification (two classes)
        - Output layer with sigmoid activation
        - Examples: spam detection, disease diagnosis, sentiment analysis
    
    Implementation note:
        Uses epsilon clipping to prevent log(0) and numerical instability.
        Clips predictions to [epsilon, 1-epsilon].
    
    Args:
        y_hat (np.ndarray): Predicted probabilities (after sigmoid)
                           Shape: (1, batch_size) typically
                           Values should be in range [0, 1]
        y_true (np.ndarray): True binary labels
                            Shape: Same as y_hat
                            Values: 0 or 1
    
    Returns:
        float: Binary cross-entropy loss (scalar)
              Range: [0, ∞)
              Lower is better, 0 is perfect
    
    Example:
        >>> y_true = np.array([[1, 0, 1, 0]])
        >>> y_hat = np.array([[0.9, 0.1, 0.8, 0.2]])  # Good predictions
        >>> binary_cross_entropy(y_hat, y_true)
        0.174  # Low loss
        >>> 
        >>> y_hat_bad = np.array([[0.1, 0.9, 0.2, 0.8]])  # Wrong predictions
        >>> binary_cross_entropy(y_hat_bad, y_true)
        2.120  # High loss
    """
    epsilon = 1e-15  # Small constant for numerical stability
    
    # Clip predictions to prevent log(0) and log(1)
    # This is essential for numerical stability
    p = np.clip(y_hat, epsilon, 1 - epsilon)
    
    # Compute BCE
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def binary_cross_entropy_derivative(y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Derivative of Binary Cross-Entropy with respect to predictions.
    
    Computes ∂L/∂ŷ for backpropagation.
    
    Mathematical formula:
        ∂L/∂ŷ = (1/m) * [-(y/ŷ) + (1-y)/(1-ŷ)]
    
    where:
        - m: batch size
        - y: true label (0 or 1)
        - ŷ: predicted probability
    
    Simplifies to:
        ∂L/∂ŷ = (1/m) * (ŷ - y) / [ŷ(1-ŷ)]
    
    Note: When combined with sigmoid activation, the derivative simplifies
    further to just (ŷ - y) / m, which is very elegant.
    
    Args:
        y_hat (np.ndarray): Predicted probabilities
                           Shape: (1, batch_size) typically
        y_true (np.ndarray): True binary labels
                            Shape: Same as y_hat
    
    Returns:
        np.ndarray: Gradient of loss with respect to predictions
                   Shape: Same as y_hat
                   Used to start backpropagation
    
    Example:
        >>> y_true = np.array([[1, 0]])
        >>> y_hat = np.array([[0.9, 0.8]])  # First correct, second wrong
        >>> binary_cross_entropy_derivative(y_hat, y_true)
        # Larger gradient for wrong prediction
    """
    m = y_hat.shape[1]  # Batch size
    # Note: Could add epsilon clipping here for stability
    return (-(y_true / y_hat) + ((1 - y_true) / (1 - y_hat))) / m


# ============================================================================
# CATEGORICAL CROSS-ENTROPY (CCE) - FOR MULTI-CLASS CLASSIFICATION
# ============================================================================

def categorical_cross_entropy(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    """
    Categorical Cross-Entropy (CCE) loss function.
    
    The standard loss function for multi-class classification (>2 classes).
    Measures the difference between predicted and true probability distributions.
    
    Mathematical formula:
        L = -(1/N) * Σᵢ Σⱼ yᵢⱼ · log(ŷᵢⱼ)
    
    where:
        - N: number of samples (batch size)
        - i: sample index
        - j: class index
        - yᵢⱼ: true probability (1 for correct class, 0 otherwise in one-hot)
        - ŷᵢⱼ: predicted probability
    
    For one-hot encoded labels, simplifies to:
        L = -(1/N) * Σᵢ log(ŷᵢ,cᵢ)
    where cᵢ is the correct class for sample i.
    
    Properties:
        - Expects probabilities (ŷ ∈ [0, 1]) from softmax
        - Each sample's probabilities should sum to 1
        - Always non-negative
        - Approaches 0 as predictions improve
        - Approaches ∞ as predictions worsen
    
    Advantages:
        - Natural for multi-class problems
        - Works perfectly with softmax output
        - Probabilistic interpretation
        - Theoretically grounded (maximum likelihood)
        - Combined with softmax, gives simple gradients
    
    Disadvantages:
        - Numerically unstable without clipping
        - Sensitive to class imbalance
        - Not robust to label noise
    
    Use cases:
        - Multi-class classification (>2 classes)
        - Output layer with softmax activation
        - Examples: image classification, language modeling, object detection
    
    Special integration:
        When used with softmax activation, the combined derivative
        simplifies to (ŷ - y), which is implemented in the SoftmaxLayer
        for numerical stability and efficiency.
    
    Implementation note:
        Uses epsilon clipping to prevent log(0) and numerical instability.
        Assumes input shape (num_classes, batch_size).
    
    Args:
        y_hat (np.ndarray): Predicted probabilities (after softmax)
                           Shape: (num_classes, batch_size)
                           Each column should sum to 1
        y_true (np.ndarray): True labels (one-hot encoded)
                            Shape: Same as y_hat
                            Each column has one 1, rest 0s
    
    Returns:
        float: Categorical cross-entropy loss (scalar)
              Range: [0, ∞)
              Lower is better
              Minimum: -log(1/K) where K is number of classes (random guessing)
    
    Example:
        >>> # 3 classes, 2 samples
        >>> y_true = np.array([[1, 0],   # Sample 1: class 0
        ...                    [0, 0],   # Sample 2: class 2
        ...                    [0, 1]])
        >>> y_hat = np.array([[0.8, 0.1],   # Good prediction for sample 1
        ...                   [0.1, 0.2],
        ...                   [0.1, 0.7]])  # Good prediction for sample 2
        >>> categorical_cross_entropy(y_hat, y_true)
        0.178  # Low loss (good predictions)
        >>> 
        >>> y_hat_bad = np.array([[0.1, 0.7],   # Wrong predictions
        ...                       [0.2, 0.2],
        ...                       [0.7, 0.1]])
        >>> categorical_cross_entropy(y_hat_bad, y_true)
        1.609  # High loss (bad predictions)
    """
    epsilon = 1e-12  # Slightly smaller epsilon for better precision
    
    # Clip predictions to prevent log(0)
    p = np.clip(y_hat, epsilon, 1.0 - epsilon)
    
    # Assuming shape: (Classes, Batch)
    m = y_true.shape[1]  # Batch size
    
    # Compute CCE
    # Only non-zero for the correct class (due to one-hot encoding)
    return -np.sum(y_true * np.log(p)) / m


def categorical_cross_entropy_derivative(y_hat: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Derivative of Categorical Cross-Entropy with respect to predictions.
    
    Computes ∂L/∂ŷ for backpropagation.
    
    Mathematical formula:
        ∂L/∂ŷⱼ = -(1/m) * (y/ŷ)
    
    where:
        - m: batch size
        - y: true one-hot label
        - ŷ: predicted probability
    
    IMPORTANT NOTE:
        This derivative is typically NOT used directly in practice!
        When combined with softmax activation, the derivative simplifies to:
            ∂(softmax + CCE)/∂z = (ŷ - y) / m
        
        This simplified form is implemented in the SoftmaxLayer.backward_pass()
        for numerical stability and efficiency. The softmax layer expects
        y_true directly, not this gradient.
    
    This function is provided for completeness and for cases where CCE
    is used without softmax (rare).
    
    Args:
        y_hat (np.ndarray): Predicted probabilities
                           Shape: (num_classes, batch_size)
        y_true (np.ndarray): True one-hot labels
                            Shape: Same as y_hat
    
    Returns:
        np.ndarray: Gradient of loss with respect to predictions
                   Shape: Same as y_hat
                   Usually not used directly (see note above)
    
    Example:
        >>> y_true = np.array([[1, 0], [0, 1]])
        >>> y_hat = np.array([[0.8, 0.3], [0.2, 0.7]])
        >>> categorical_cross_entropy_derivative(y_hat, y_true)
        # Gradient shows how to adjust predictions
    """
    epsilon = 1e-12
    y_hat = np.clip(y_hat, epsilon, 1.0 - epsilon)
    m = y_true.shape[1]  # Batch size
    
    # Standard derivative (if NOT using the Softmax-CCE shortcut)
    return -(y_true / y_hat) / m


# ============================================================================
# LOSS FACTORY
# ============================================================================

def get_loss(name: LossType) -> tuple[LossFunc, Derivative]:
    """
    Factory function to retrieve loss function and its derivative.
    
    Provides a clean interface for dynamically selecting loss functions
    based on string identifiers. Used throughout the neural network code
    to configure the training objective.
    
    Available losses:
        - "mse": Mean Squared Error (regression)
        - "bce": Binary Cross-Entropy (binary classification)
        - "cce": Categorical Cross-Entropy (multi-class classification)
    
    Selection guidelines:
        - Use MSE for regression (predicting continuous values)
        - Use BCE for binary classification (2 classes, sigmoid output)
        - Use CCE for multi-class classification (>2 classes, softmax output)
    
    Args:
        name (LossType): Loss function identifier
                        Valid values: "mse", "bce", "cce"
    
    Returns:
        tuple[LossFunc, Derivative]: (loss_function, derivative_function)
            - loss_function: Computes scalar loss value
            - derivative_function: Computes gradient for backpropagation
    
    Raises:
        ValueError: If name is not one of the valid loss types
    
    Example:
        >>> # Get MSE for regression
        >>> loss_fn, loss_deriv = get_loss("mse")
        >>> loss = loss_fn(predictions, targets)
        >>> gradient = loss_deriv(predictions, targets)
        >>> 
        >>> # Get CCE for classification
        >>> loss_fn, loss_deriv = get_loss("cce")
        >>> loss = loss_fn(softmax_output, one_hot_labels)
        >>> 
        >>> # Invalid loss type
        >>> get_loss("mae")  # Raises ValueError
    
    Usage in NeuralNetwork:
        >>> class NeuralNetwork:
        ...     def __init__(self, ..., loss_type="cce"):
        ...         self.loss_type = loss_type
        ...     
        ...     def train(self, X, y):
        ...         loss_fn, _ = get_loss(self.loss_type)
        ...         
        ...         for epoch in range(epochs):
        ...             y_pred = self.forward_pass(X)
        ...             loss = loss_fn(y_pred, y)
        ...             # ... backpropagation ...
    
    Pairing with activations:
        - MSE → Linear activation (regression)
        - BCE → Sigmoid activation (binary classification)
        - CCE → Softmax activation (multi-class classification)
    """
    if name == "mse":
        return (mse, mse_derivative)
    if name == "bce":
        return (binary_cross_entropy, binary_cross_entropy_derivative)
    if name == "cce":
        return (categorical_cross_entropy, categorical_cross_entropy_derivative)
    raise ValueError(f"Unknown loss: {name}")