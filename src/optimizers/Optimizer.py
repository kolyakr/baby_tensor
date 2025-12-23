"""
Optimizer - Documented Version

This is the abstract base class for all optimization algorithms used to train
neural networks. It defines the interface that all concrete optimizer implementations
must follow, ensuring consistency across different optimization strategies.

Optimizers are responsible for updating network parameters (weights and biases)
based on computed gradients to minimize the loss function.

Common optimizers include:
    - GradientDescent (SGD): Basic gradient descent
    - Momentum: SGD with velocity accumulation
    - Adam: Adaptive learning rates with momentum
    - RMSprop: Adaptive learning rates
    - AdaGrad: Adaptive learning rates that decrease over time

This follows the Strategy design pattern - the optimizer is a strategy for
updating parameters that can be swapped without changing the neural network code.
"""

from abc import ABC, abstractmethod
from src.layers import Layer


class Optimizer(ABC):
    """
    Abstract Base Class for Neural Network Optimizers
    
    This class defines the standard interface that all optimizer implementations
    must follow. It ensures that different optimization algorithms can be used
    interchangeably within the training pipeline.
    
    The core responsibility of an optimizer is to update network parameters
    (weights, biases, etc.) based on computed gradients to minimize the loss
    function. Different optimizers use different strategies:
        - Vanilla: directly use gradients
        - Momentum: accumulate velocity
        - Adaptive: adjust learning rates per parameter
        - Second-order: use curvature information
    
    All concrete optimizer implementations (GradientDescent, Momentum, Adam, etc.)
    must inherit from this class and implement the abstract update_layers method.
    
    Design Pattern:
        This follows the Strategy pattern - the optimizer is a pluggable strategy
        that can be selected at runtime without changing the model architecture
        or training loop code.
    
    Typical usage in training loop:
        1. Forward pass: compute predictions
        2. Compute loss: compare predictions to targets
        3. Backward pass: compute gradients via backpropagation
        4. Update: call optimizer.update_layers() to update parameters
        5. Repeat
    
    Attributes:
        None at base class level
        Concrete implementations may have:
            - Hyperparameters (learning rate, momentum coefficient, etc.)
            - State variables (velocity, moving averages, etc.)
            - Iteration counters
            - Per-parameter statistics
    
    Note: The learning_rate is passed as a parameter to update_layers rather than
          stored in __init__. This allows easy implementation of learning rate
          schedules without modifying the optimizer object.
    """
    
    def __init__(self) -> None:
        """
        Initialize the base optimizer.
        
        This is a no-op in the base class. Concrete implementations may initialize:
            - Hyperparameters (momentum coefficient, decay rates, etc.)
            - State dictionaries (velocity, moving averages, etc.)
            - Counters (iteration number, etc.)
        
        The learning rate is NOT stored here - it's passed to update_layers
        to enable flexible learning rate scheduling.
        
        Returns:
            None
            
        Example implementations:
            >>> class GradientDescent(Optimizer):
            ...     def __init__(self):
            ...         pass  # Stateless, no initialization needed
            
            >>> class Momentum(Optimizer):
            ...     def __init__(self, beta=0.9):
            ...         self.beta = beta
            ...         self.velocity = {}  # Stores velocity per parameter
            
            >>> class Adam(Optimizer):
            ...     def __init__(self, beta1=0.9, beta2=0.999):
            ...         self.beta1 = beta1
            ...         self.beta2 = beta2
            ...         self.m = {}  # First moment
            ...         self.v = {}  # Second moment
            ...         self.t = 0   # Iteration counter
        """
        pass
    
    @abstractmethod
    def update_layers(
        self,
        layers: list[Layer],
        learning_rate: float
    ) -> None:
        """
        Update parameters of all layers based on computed gradients.
        
        This method must be implemented by all concrete optimizer classes.
        It defines the optimization strategy - how to use gradients to update
        parameters.
        
        The method should:
            1. Iterate through all layers
            2. For each layer with parameters (layer.params is non-empty):
               a. Retrieve gradients from layer.grads
               b. Apply the optimization algorithm
               c. Update parameters in layer.params
            3. Update any internal optimizer state if needed
        
        Common update strategies:
            - Vanilla GD:  θ = θ - α * g
            - Momentum:    θ = θ - α * v, where v = β*v + (1-β)*g
            - Adam:        θ = θ - α * m / (√v + ε), with bias correction
            - RMSprop:     θ = θ - α * g / √v
        
        where:
            θ = parameters
            α = learning_rate
            g = gradients
            v = velocity or second moment
            m = first moment (momentum)
        
        Parameters:
            layers (list[Layer]): List of network layers
                                 Should include all layers in the model
                                 Only layers with parameters will be updated
                                 Layers without parameters (activations, pooling, etc.)
                                 are naturally skipped (empty params dict)
                                 
            learning_rate (float): Learning rate (step size) α
                                  Controls the magnitude of parameter updates
                                  Common ranges: [1e-5, 1e-1]
                                  Typical values: 0.001, 0.01, 0.1
                                  
                                  Can be:
                                  - Constant throughout training
                                  - Scheduled (decayed over time)
                                  - Adaptive per iteration
                                  
        Returns:
            None
            Updates are applied in-place to layer.params dictionaries
            
        Raises:
            NotImplementedError: If called on base class (abstract method)
            
        Example implementation (Gradient Descent):
            >>> def update_layers(self, layers, learning_rate):
            ...     for layer in layers:
            ...         for key in layer.params.keys():
            ...             gradient = layer.grads[f"d{key}"]
            ...             layer.params[key] -= learning_rate * gradient
        
        Example usage:
            >>> from src.layers import DenseLayer, ReLULayer
            >>> from src.optimizers import Adam
            >>> 
            >>> # Build model
            >>> layers = [
            ...     DenseLayer(128, "relu", "layer1"),
            ...     ReLULayer("relu1"),
            ...     DenseLayer(10, "softmax", "output")
            ... ]
            >>> 
            >>> # Create optimizer
            >>> optimizer = Adam(beta1=0.9, beta2=0.999)
            >>> 
            >>> # Training loop
            >>> for epoch in range(100):
            ...     # Forward pass
            ...     activations = forward_propagate(layers, X)
            ...     
            ...     # Compute loss
            ...     loss = compute_loss(activations[-1], y_true)
            ...     
            ...     # Backward pass (populates layer.grads)
            ...     backward_propagate(layers, y_true)
            ...     
            ...     # Update parameters
            ...     optimizer.update_layers(layers, learning_rate=0.001)
        
        Implementation guidelines:
            - Only update parameters that exist (check layer.params.keys())
            - Ensure gradient keys match parameter keys (dW for W, db for b)
            - Handle edge cases (empty gradients, NaN values, etc.)
            - For stateful optimizers (Momentum, Adam):
              * Initialize state lazily (on first parameter encounter)
              * Use unique keys to store per-parameter state
              * Common pattern: f"{layer.name}_{param_name}"
            - Apply updates in-place for memory efficiency
            - Consider numerical stability (clip gradients if needed)
        
        Learning rate scheduling:
            Since learning_rate is a parameter, easy to implement schedules:
            
            >>> # Step decay
            >>> lr = 0.1
            >>> for epoch in range(100):
            ...     if epoch % 30 == 0 and epoch > 0:
            ...         lr *= 0.1
            ...     optimizer.update_layers(layers, learning_rate=lr)
            
            >>> # Exponential decay
            >>> lr_0 = 0.1
            >>> for epoch in range(100):
            ...     lr = lr_0 * (0.95 ** epoch)
            ...     optimizer.update_layers(layers, learning_rate=lr)
            
            >>> # Cosine annealing
            >>> import math
            >>> lr_max, lr_min = 0.1, 0.001
            >>> for epoch in range(100):
            ...     lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / 100))
            ...     optimizer.update_layers(layers, learning_rate=lr)
        """
        pass