"""
Adam Optimizer - Documented Version

Adam (Adaptive Moment Estimation) is one of the most popular optimization algorithms
for training neural networks. It combines ideas from RMSprop and momentum optimization.

Key features:
    1. Adaptive learning rates for each parameter
    2. Momentum-based gradient accumulation (first moment)
    3. Squared gradient accumulation (second moment)
    4. Bias correction for moment estimates
    5. Generally works well with default hyperparameters

Adam computes adaptive learning rates by maintaining exponential moving averages
of both the gradient (first moment) and the squared gradient (second moment).

Reference: "Adam: A Method for Stochastic Optimization" (Kingma & Ba, 2014)
"""

from src.layers import Layer
from src.optimizers import Optimizer
import numpy as np


class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) Optimizer
    
    Implements the Adam optimization algorithm, which computes adaptive learning
    rates for each parameter using estimates of first and second moments of gradients.
    
    Algorithm:
        1. Compute first moment (momentum):
           m1_t = β₁ * m1_{t-1} + (1 - β₁) * g_t
           
        2. Compute second moment (squared gradient average):
           m2_t = β₂ * m2_{t-1} + (1 - β₂) * g_t²
           
        3. Bias correction (important for early iterations):
           m1_corrected = m1_t / (1 - β₁^t)
           m2_corrected = m2_t / (1 - β₂^t)
           
        4. Parameter update:
           θ_t = θ_{t-1} - α * m1_corrected / (√(m2_corrected) + ε)
    
    where:
        - β₁ (b): exponential decay rate for first moment (typically 0.9)
        - β₂ (g): exponential decay rate for second moment (typically 0.999)
        - α: learning rate (step size)
        - ε: small constant for numerical stability (1e-10)
        - g_t: gradient at time t
        - t: iteration number
    
    Advantages:
        - Adaptive learning rates per parameter
        - Works well with sparse gradients
        - Computationally efficient
        - Requires little memory overhead
        - Robust to choice of hyperparameters
        - Good default optimizer for many problems
    
    Disadvantages:
        - Can converge to suboptimal solutions in some cases
        - May require learning rate decay for best results
        - More memory than SGD (stores two moment estimates per parameter)
    
    Attributes:
        b (float): Beta1 - exponential decay rate for first moment (momentum)
                   Default: 0.9
        g (float): Beta2 - exponential decay rate for second moment
                   Default: 0.999
        eps (float): Small constant for numerical stability
                     Default: 1e-10
        iter (int): Current iteration number (starts at 1)
                    Used for bias correction
        m1 (dict): First moment estimates (momentum)
                   Keys: "{layer_name}_{param_name}"
                   Values: np.ndarray with same shape as parameter
        m2 (dict): Second moment estimates (squared gradients)
                   Keys: "{layer_name}_{param_name}"
                   Values: np.ndarray with same shape as parameter
    """
    
    def __init__(self, b: float = 0.9, g: float = 0.999) -> None:
        """
        Initialize the Adam optimizer.
        
        Parameters:
            b (float, optional): Beta1 - first moment decay rate (momentum coefficient)
                                Range: [0, 1), typically 0.9
                                Higher values give more weight to past gradients
                                Default: 0.9
            g (float, optional): Beta2 - second moment decay rate
                                Range: [0, 1), typically 0.999
                                Higher values give more weight to past squared gradients
                                Default: 0.999
                                
        Returns:
            None
            
        Note:
            The default values (b=0.9, g=0.999) are recommended in the original
            paper and work well for most problems. Typically no tuning needed.
        """
        self.b = b  # Beta1 for first moment
        self.g = g  # Beta2 for second moment
        self.eps = 1e-10  # Small constant for numerical stability
        self.iter = 1  # Iteration counter (starts at 1 for bias correction)
        
        # Moment estimates (initialized lazily during first update)
        self.m1 = {}  # First moment (momentum)
        self.m2 = {}  # Second moment (squared gradients)
    
    def update_layers(
        self,
        layers: list[Layer],
        learning_rate: float
    ) -> None:
        """
        Update parameters of all layers using Adam optimization.
        
        For each parameter in each layer:
            1. Initialize or retrieve moment estimates (m1, m2)
            2. Update first moment with current gradient (momentum)
            3. Update second moment with squared gradient
            4. Apply bias correction to both moments
            5. Compute adaptive learning rate
            6. Update parameter
            7. Store updated moments for next iteration
        
        The update rule adaptively scales the learning rate for each parameter
        based on the history of gradients, allowing different parameters to
        learn at different effective rates.
        
        Parameters:
            layers (list[Layer]): List of network layers to update
                                 Only layers with parameters (params dict) are updated
            learning_rate (float): Global learning rate (α)
                                  Common values: 0.001, 0.0001
                                  Can be decayed over training
                                  
        Returns:
            None (updates are applied in-place to layer.params)
            
        Algorithm details:
            For parameter θ with gradient g:
            
            m1 = β₁ * m1_prev + (1 - β₁) * g         # Momentum update
            m2 = β₂ * m2_prev + (1 - β₂) * g²        # RMSprop-like update
            
            m1_hat = m1 / (1 - β₁^t)                 # Bias correction
            m2_hat = m2 / (1 - β₂^t)                 # Bias correction
            
            θ = θ - α * m1_hat / (√m2_hat + ε)      # Parameter update
        
        Example:
            >>> from src.layers import DenseLayer, ReLULayer
            >>> layers = [
            ...     DenseLayer(128, "relu", "dense1"),
            ...     ReLULayer("relu1"),
            ...     DenseLayer(10, "softmax", "dense2")
            ... ]
            >>> optimizer = Adam(b=0.9, g=0.999)
            >>> 
            >>> # After computing gradients via backpropagation:
            >>> optimizer.update_layers(layers, learning_rate=0.001)
            >>> # Parameters in dense1 and dense2 are now updated
        """
        # Update all parameters in all layers
        for layer in layers:
            for key in layer.params.keys():
                # Create unique identifier for this parameter
                m_key = f"{layer.name}_{key}"
                
                # Get gradient for this parameter
                gradient = layer.grads[f"d{key}"]
                
                # Initialize or retrieve moment estimates
                # If first time seeing this parameter, initialize to zeros
                old_m1 = self.m1.get(m_key, np.zeros_like(layer.params[key]))
                old_m2 = self.m2.get(m_key, np.zeros_like(layer.params[key]))
                
                # Update first moment (exponential moving average of gradients)
                # This is similar to momentum: smooths out gradient updates
                m1 = self.b * old_m1 + (1 - self.b) * gradient
                
                # Update second moment (exponential moving average of squared gradients)
                # This adapts learning rate based on gradient magnitude
                m2 = self.g * old_m2 + (1 - self.g) * np.power(gradient, 2)
                
                # Bias correction for first moment
                # Early iterations have biased estimates toward zero
                # Correction: divide by (1 - β^t) which approaches 1 as t grows
                corrected_m1 = m1 / (1 - np.power(self.b, self.iter))
                
                # Bias correction for second moment
                corrected_m2 = m2 / (1 - np.power(self.g, self.iter))
                
                # Parameter update with adaptive learning rate
                # Learning rate is scaled by:
                #   - m1 (momentum direction)
                #   - 1/√m2 (inverse of gradient magnitude)
                # This gives larger steps for consistent gradients,
                # smaller steps for noisy/varying gradients
                layer.params[key] -= learning_rate * (
                    corrected_m1 / (np.sqrt(corrected_m2 + 1) + self.eps)
                )
                
                # Store updated moments for next iteration
                self.m1[m_key] = m1
                self.m2[m_key] = m2
        
        # Increment iteration counter (used for bias correction)
        self.iter += 1