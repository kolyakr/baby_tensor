"""
BatchNormalizationLayer - Documented Version

This layer implements batch normalization, a technique that normalizes the inputs
of each layer to have mean 0 and variance 1, which helps stabilize and accelerate
training of deep neural networks.
"""

from src.layers import Layer
import numpy as np

class BatchNormalizationLayer(Layer):
    """
    Batch Normalization Layer
    
    Normalizes activations across the batch dimension to have zero mean and unit variance.
    Supports both fully connected (2D) and convolutional (4D) inputs.
    
    Attributes:
        name (str): Name identifier for the layer
        mean (np.ndarray): Running mean computed during forward pass
        variance (np.ndarray): Running variance computed during forward pass
        eps (float): Small constant for numerical stability (default: 1e-10)
        params (dict): Learnable parameters
            - gamma (np.ndarray): Scale parameter
            - betta (np.ndarray): Shift parameter (note: spelled 'betta', not 'beta')
        cache (dict): Cached values for backward pass
            - A_prev (np.ndarray): Input activations from previous layer
        grads (dict): Gradients computed during backward pass
            - dgamma (np.ndarray): Gradient of gamma
            - dbetta (np.ndarray): Gradient of betta
    """
    
    def __init__(self, name: str) -> None:
        """
        Initialize the Batch Normalization layer.
        
        Parameters:
            name (str): Name identifier for the layer
            
        Returns:
            None
        """
        super().__init__(name)
        
        self.mean: np.ndarray = []
        self.variance: np.ndarray = []
        self.eps: float = 1e-10
    
    def initialize_parameters(self, input_dim: tuple | int, seed: int = 0) -> tuple | int:
        """
        Initialize learnable parameters (gamma and betta) based on input dimensions.
        
        For fully connected layers (2D input):
            - gamma shape: (D, 1) where D is the number of features
            - betta shape: (D, 1)
            
        For convolutional layers (4D input):
            - gamma shape: (1, D, 1, 1) where D is the number of channels
            - betta shape: (1, D, 1, 1)
        
        Parameters:
            input_dim (tuple | int): Input dimensions
                - If tuple: (channels, height, width) for convolutional input
                - If int: number of features for fully connected input
            seed (int): Random seed (not currently used, default: 0)
            
        Returns:
            tuple | int: Same as input_dim, passed through unchanged
        """
        
        is_input_conv = isinstance(input_dim, tuple)
        
        if is_input_conv:
            D = input_dim[0]  # Number of channels
        else:
            D = input_dim  # Number of features
            
        # Initialize gamma (scale) to ones
        self.params["gamma"] = np.ones((D, 1) if not is_input_conv else (1, D, 1, 1))
        
        # Initialize betta (shift) to zeros
        self.params["betta"] = np.zeros((D, 1) if not is_input_conv else (1, D, 1, 1))
        
        return input_dim
    
    def forward_pass(self, A_prev: np.ndarray) -> np.ndarray:
        """
        Perform forward pass of batch normalization.
        
        Applies the transformation:
            1. Compute batch mean (mu) and variance (sigma2)
            2. Normalize: Z_hat = (A_prev - mu) / sqrt(sigma2 + eps)
            3. Scale and shift: A = gamma * Z_hat + betta
        
        Parameters:
            A_prev (np.ndarray): Input activations
                - Shape (D, m) for fully connected: D features, m samples
                - Shape (m, C, H, W) for convolutional: m samples, C channels, H height, W width
                
        Returns:
            np.ndarray: Normalized and scaled activations, same shape as A_prev
        """
        self.cache["A_prev"] = A_prev
        
        # Determine axes to sum over and number of elements M
        if len(A_prev.shape) == 2:
            sum_axes = 1  # Sum over samples
            M = A_prev.shape[1]  # Number of samples
        else:
            sum_axes = (0, 2, 3)  # Sum over batch, height, width
            M = A_prev.shape[0] * A_prev.shape[2] * A_prev.shape[3]
            
        # Compute batch statistics
        mu = np.sum(A_prev, axis=sum_axes, keepdims=True) / M  # Mean
        Z_centered = A_prev - mu  # Center the data
        Z_squared = np.power(Z_centered, 2)  # Squared differences
        sigma2 = np.sum(Z_squared, axis=sum_axes, keepdims=True) / M  # Variance
        sigma = np.sqrt(sigma2 + self.eps)  # Standard deviation
        
        # Normalize
        Z_hat = Z_centered / sigma
        
        # Scale and shift
        A = self.params["gamma"] * Z_hat + self.params["betta"]
        
        # Store statistics for potential use (e.g., running averages in inference)
        self.mean = mu
        self.variance = sigma2
        
        return A
    
    def backward_pass(self, dA: np.ndarray) -> np.ndarray:
        """
        Perform backward pass to compute gradients.
        
        Computes gradients for:
            - dgamma: gradient with respect to gamma parameter
            - dbetta: gradient with respect to betta parameter
            - dA_prev: gradient with respect to input (to pass to previous layer)
        
        The backward pass implements the chain rule through the normalization operation,
        accounting for the fact that the mean and variance depend on all batch elements.
        
        Parameters:
            dA (np.ndarray): Gradient of loss with respect to layer output
                - Same shape as the output from forward_pass
                
        Returns:
            np.ndarray: Gradient of loss with respect to layer input (dA_prev)
                - Same shape as A_prev from forward_pass
        """
        A_prev = self.cache["A_prev"]
        mu = self.mean
        sigma2 = self.variance
        
        # Determine axes and number of elements
        if len(A_prev.shape) == 2:
            sum_axes = 1
            M = A_prev.shape[1]
        else:
            sum_axes = (0, 2, 3)
            M = A_prev.shape[0] * A_prev.shape[2] * A_prev.shape[3]

        # Gradient of betta (simple sum since betta is added directly)
        self.grads["dbetta"] = np.sum(dA, axis=sum_axes, keepdims=True)
        
        # Recompute normalized values
        sigma = np.sqrt(sigma2 + self.eps)
        Z_hat = (A_prev - mu) / sigma
        
        # Gradient of gamma (Z_hat is multiplied by gamma)
        self.grads["dgamma"] = np.sum(dA * Z_hat, axis=sum_axes, keepdims=True)

        # Gradient flowing back through the normalization
        dZ_hat = dA * self.params["gamma"]

        # Gradient with respect to variance
        dsigma2 = np.sum(
            dZ_hat * (A_prev - mu), 
            axis=sum_axes, 
            keepdims=True
        ) * (-0.5) * np.power(sigma2 + self.eps, -1.5)

        # Gradient with respect to mean
        dmu = (
            np.sum(dZ_hat * (-1/sigma), axis=sum_axes, keepdims=True) + 
            dsigma2 * np.sum(-2 * (A_prev - mu), axis=sum_axes, keepdims=True) / M
        )

        # Gradient with respect to input (accounts for dependencies through mean and variance)
        dA_prev = (dZ_hat / sigma) + (dsigma2 * 2 * (A_prev - mu) / M) + (dmu / M)

        return dA_prev