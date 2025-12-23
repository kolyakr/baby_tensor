"""
GradientDescent Optimizer - Documented Version

This implements the simplest and most fundamental optimization algorithm:
Gradient Descent (often called Stochastic Gradient Descent or SGD when used
with mini-batches).

The algorithm is elegantly simple:
    θ_new = θ_old - α * ∇L(θ)

where:
    - θ: parameters (weights, biases)
    - α: learning rate (step size)
    - ∇L(θ): gradient of loss with respect to parameters

Despite its simplicity, gradient descent is:
    1. The foundation of all neural network optimization
    2. Still competitive with more sophisticated methods
    3. Often used with momentum or other enhancements
    4. Provides theoretical guarantees for convex problems
"""

from src.optimizers import Optimizer
from src.layers import Layer


class GradientDescent(Optimizer):
    """
    Gradient Descent (SGD) Optimizer
    
    Implements vanilla gradient descent (also called Stochastic Gradient Descent
    when used with mini-batches). Updates parameters by moving in the direction
    opposite to the gradient, scaled by the learning rate.
    
    Algorithm:
        For each parameter θ with gradient g:
            θ = θ - α * g
    
    where:
        - θ: parameter (weight or bias)
        - α: learning rate (controls step size)
        - g: gradient ∂L/∂θ (direction of steepest ascent)
    
    The negative sign ensures we move downhill (minimize loss).
    
    Characteristics:
        - Memoryless: only uses current gradient
        - Simple and fast per iteration
        - No hyperparameters besides learning rate
        - Deterministic given the same data order
    
    Advantages:
        - Extremely simple to implement and understand
        - Low memory overhead (no auxiliary variables)
        - Fast per-iteration computation
        - Well-studied theoretical properties
        - Can escape shallow local minima (with proper learning rate)
        - Works well with learning rate schedules
    
    Disadvantages:
        - Sensitive to learning rate choice (critical hyperparameter)
        - Same learning rate for all parameters (not adaptive)
        - Can oscillate in narrow valleys
        - Slower convergence than adaptive methods (Adam, RMSprop)
        - May get stuck in saddle points
        - Requires careful learning rate tuning/scheduling
    
    Common enhancements (not implemented here):
        - Momentum: accumulates velocity from past gradients
        - Nesterov momentum: looks ahead before computing gradient
        - Learning rate scheduling: decay learning rate over time
        - Weight decay: L2 regularization applied during update
    
    Attributes:
        None - this is a stateless optimizer
              (enhanced versions like Momentum would store velocity)
    
    Note: This implementation updates parameters in-place, modifying the
          original layer.params dictionaries directly.
    """
    
    def update_layers(
        self,
        layers: list[Layer],
        learning_rate: float
    ) -> None:
        """
        Update parameters of all layers using gradient descent.
        
        Applies the update rule: θ = θ - α * ∇L(θ)
        
        For each layer with parameters:
            - Iterates over all parameters (W, b, etc.)
            - Retrieves the corresponding gradient
            - Updates parameter by moving opposite to gradient
            - Step size controlled by learning_rate
        
        The update is performed in-place, directly modifying layer.params.
        
        Parameters:
            layers (list[Layer]): List of network layers to update
                                 Only layers with parameters (non-empty params dict) are updated
                                 Activation layers, pooling layers, etc. are skipped
            learning_rate (float): Learning rate α (step size)
                                  Controls how far to move in gradient direction
                                  Common values: 0.1, 0.01, 0.001
                                  Too large: divergence/oscillation
                                  Too small: slow convergence
                                  Often decayed during training
                                  
        Returns:
            None (updates are applied in-place to layer.params)
            
        Algorithm:
            For each layer:
                For each parameter key in layer.params:
                    gradient = layer.grads["d{key}"]
                    layer.params[key] = layer.params[key] - learning_rate * gradient
        
        Example:
            >>> from src.layers import DenseLayer, ReLULayer
            >>> layers = [
            ...     DenseLayer(128, "relu", "dense1"),  # Has W and b
            ...     ReLULayer("relu1"),                 # No parameters (skipped)
            ...     DenseLayer(10, "softmax", "dense2") # Has W and b
            ... ]
            >>> optimizer = GradientDescent()
            >>> 
            >>> # After computing gradients via backpropagation:
            >>> # layers[0].grads["dW"] and layers[0].grads["db"] are populated
            >>> # layers[2].grads["dW"] and layers[2].grads["db"] are populated
            >>> 
            >>> optimizer.update_layers(layers, learning_rate=0.01)
            >>> 
            >>> # Parameters updated:
            >>> # layers[0].params["W"] -= 0.01 * layers[0].grads["dW"]
            >>> # layers[0].params["b"] -= 0.01 * layers[0].grads["db"]
            >>> # layers[2].params["W"] -= 0.01 * layers[2].grads["dW"]
            >>> # layers[2].params["b"] -= 0.01 * layers[2].grads["db"]
        
        Learning rate guidelines:
            - Start with 0.01 or 0.001
            - Use learning rate finder to find good range
            - Apply learning rate decay (e.g., reduce by 10x periodically)
            - Monitor loss: if oscillating, reduce learning rate
            - If converging slowly, can try increasing learning rate
            
        Common patterns:
            - Step decay: multiply by 0.1 every N epochs
            - Exponential decay: multiply by 0.95 every epoch
            - Cosine annealing: follow cosine curve
            - Warm restarts: periodic learning rate resets
        """
        # Update all parameters in all layers
        for layer in layers:
            # Only process layers with parameters
            # (activation layers, pooling layers, etc. have empty params dicts)
            for key in layer.params.keys():
                # Get the gradient for this parameter
                # Naming convention: gradient of "W" is stored as "dW"
                gradient_key = f"d{key}"
                gradient = layer.grads[gradient_key]
                
                # Apply gradient descent update rule
                # Move parameter in direction opposite to gradient
                # θ_new = θ_old - α * ∇L(θ)
                layer.params[key] -= learning_rate * gradient