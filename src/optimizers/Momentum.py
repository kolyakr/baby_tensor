"""
Momentum Optimizer - Documented Version

Momentum optimization enhances gradient descent by accumulating a velocity vector
in directions of persistent gradient. This helps accelerate convergence and
dampen oscillations.

Key idea: Like a ball rolling downhill, momentum builds up speed in consistent
directions and reduces oscillation in directions where gradients change sign.

Algorithm:
    v_t = β * v_{t-1} + (1 - β) * g_t
    θ_t = θ_{t-1} - α * v_t

where:
    - v: velocity (accumulated momentum)
    - β: momentum coefficient (typically 0.9)
    - g: gradient
    - α: learning rate

Benefits over vanilla SGD:
    1. Faster convergence in relevant directions
    2. Dampens oscillations in irrelevant directions
    3. Can escape shallow local minima
    4. Smooths out noisy gradients

Reference: Originally from classical mechanics; adapted for neural networks
           by Rumelhart, Hinton & Williams (1986)
"""

import numpy as np
from src.optimizers import Optimizer
from src.layers import Layer


class Momentum(Optimizer):
    """
    Momentum Optimizer (SGD with Momentum)
    
    Implements gradient descent with momentum, which accumulates an exponentially
    weighted moving average of past gradients to determine the update direction.
    
    Algorithm:
        For each parameter θ with gradient g:
            v_t = β * v_{t-1} + (1 - β) * g_t
            θ_t = θ_{t-1} - α * v_t
    
    where:
        - v: velocity vector (momentum accumulation)
        - β: momentum coefficient [0, 1) - typically 0.9
        - g: current gradient
        - α: learning rate
        - t: iteration number
    
    Physical intuition:
        Think of a ball rolling down a hill:
        - Gradient is like gravity (force)
        - Velocity accumulates over time
        - Ball speeds up in consistent downhill directions
        - Ball slows down when gradient opposes velocity
        - Can roll over small bumps (escape shallow local minima)
    
    Mathematical properties:
        - Effective learning rate in consistent direction: α / (1 - β)
          With β=0.9: 10x amplification in consistent directions
        - Smooths gradient over ~1/(1-β) steps
          With β=0.9: averages over ~10 steps
        - Reduces variance of updates (more stable)
    
    Advantages:
        - Faster convergence than vanilla SGD
        - Reduces oscillation in high-curvature directions
        - Can escape shallow local minima
        - Smooths noisy gradients
        - Simple with just one hyperparameter (β)
        - Works well with learning rate scheduling
    
    Disadvantages:
        - Adds memory overhead (stores velocity per parameter)
        - Can overshoot minima if momentum too high
        - May not adapt well to changing gradient landscapes
        - Still sensitive to learning rate (like SGD)
        - Not adaptive per-parameter (unlike Adam/RMSprop)
    
    Typical hyperparameters:
        - β = 0.9: standard choice, works well for most problems
        - β = 0.99: for very smooth optimization landscapes
        - Learning rate often needs adjustment vs vanilla SGD
          (can often use 10x larger learning rate)
    
    Variants (not implemented here):
        - Nesterov momentum: looks ahead before computing gradient
        - Adaptive momentum: adjusts β based on progress
    
    Attributes:
        b (float): Momentum coefficient β
                   Controls how much past gradients influence current update
                   Range: [0, 1), typically 0.9
        m (dict): Velocity vectors (momentum accumulation)
                  Keys: "{layer_name}_{param_name}"
                  Values: np.ndarray with same shape as parameter
                  Initialized to zeros on first encounter
    """
    
    def __init__(self, b: float = 0.9) -> None:
        """
        Initialize the Momentum optimizer.
        
        Parameters:
            b (float, optional): Momentum coefficient β
                                Range: [0, 1), typically 0.9
                                Higher values = more momentum (smoother, but slower to adapt)
                                Lower values = less momentum (more responsive, but noisier)
                                
                                Common values:
                                - 0.9: standard choice (good for most problems)
                                - 0.95: more smoothing
                                - 0.99: very smooth (for clean gradients)
                                - 0.5: less smoothing (for noisy gradients)
                                
                                Default: 0.9
                                
        Returns:
            None
            
        Example:
            >>> # Standard momentum
            >>> opt1 = Momentum(b=0.9)
            >>> 
            >>> # Heavy momentum for smooth landscapes
            >>> opt2 = Momentum(b=0.99)
            >>> 
            >>> # Light momentum for noisy gradients
            >>> opt3 = Momentum(b=0.5)
        """
        self.b = b  # Momentum coefficient (beta)
        self.m = {}  # Velocity vectors (initialized lazily)
    
    def update_layers(
        self,
        layers: list[Layer],
        learning_rate: float
    ) -> None:
        """
        Update parameters of all layers using momentum optimization.
        
        For each parameter:
            1. Initialize or retrieve velocity vector
            2. Update velocity as exponential moving average of gradients
            3. Update parameter by moving in velocity direction
            4. Store updated velocity for next iteration
        
        The velocity accumulates in directions of consistent gradient and
        cancels out in directions where gradient oscillates.
        
        Parameters:
            layers (list[Layer]): List of network layers to update
                                 Only layers with parameters are updated
            learning_rate (float): Learning rate α (step size)
                                  Controls magnitude of parameter updates
                                  Common values: 0.01, 0.001
                                  Can often be 10x larger than vanilla SGD
                                  
        Returns:
            None (updates are applied in-place to layer.params)
            
        Algorithm details:
            For parameter θ with gradient g and velocity v:
            
            v_new = β * v_old + (1 - β) * g
            θ_new = θ_old - α * v_new
            
            The (1 - β) factor ensures v remains normalized even as β→1.
        
        Velocity interpretation:
            - v accumulates gradients over time
            - Large consistent gradients → large velocity
            - Oscillating gradients → small velocity (cancel out)
            - Velocity persists even when gradient changes
            - Acts as a "low-pass filter" for gradients
        
        Example:
            >>> from src.layers import DenseLayer, ReLULayer
            >>> layers = [
            ...     DenseLayer(128, "relu", "dense1"),
            ...     ReLULayer("relu1"),
            ...     DenseLayer(10, "softmax", "dense2")
            ... ]
            >>> optimizer = Momentum(b=0.9)
            >>> 
            >>> # Training loop
            >>> for epoch in range(10):
            ...     # ... forward pass, compute loss, backward pass ...
            ...     optimizer.update_layers(layers, learning_rate=0.01)
            >>> 
            >>> # Velocity accumulates over iterations, speeding up convergence
        
        Comparison with vanilla SGD:
            Vanilla SGD:     θ_new = θ_old - α * g
            With Momentum:   θ_new = θ_old - α * v
                            where v = β*v_old + (1-β)*g
            
            Momentum smooths updates, making optimization more stable and
            often converging faster, especially in ravine-like landscapes.
        
        Learning rate adjustment:
            When switching from SGD to Momentum, you may need to:
            - Reduce learning rate slightly (momentum amplifies consistent gradients)
            - OR increase learning rate (momentum can handle larger steps)
            - Experiment to find optimal value for your problem
            
        Monitoring:
            - If loss oscillates: reduce learning rate or momentum coefficient
            - If convergence too slow: increase learning rate
            - If overshooting: reduce momentum coefficient
        """
        # Update all parameters in all layers
        for layer in layers:
            # Only process layers with parameters
            for key in layer.params.keys():
                # Create unique identifier for this parameter's velocity
                momentum_key = f"{layer.name}_{key}"
                
                # Get current gradient
                gradient = layer.grads[f"d{key}"]
                
                # Initialize or retrieve velocity vector
                # First time seeing this parameter: initialize velocity to zeros
                old_momentum = self.m.get(momentum_key, np.zeros_like(layer.params[key]))
                
                # Update velocity (exponential moving average of gradients)
                # v_new = β * v_old + (1 - β) * g
                # 
                # This accumulates gradient information over time:
                # - β * v_old: keeps momentum from previous iterations
                # - (1 - β) * g: adds current gradient contribution
                momentum = self.b * old_momentum + (1 - self.b) * gradient
                
                # Update parameter using velocity
                # θ_new = θ_old - α * v
                # 
                # Instead of moving directly in gradient direction,
                # we move in the accumulated momentum direction
                layer.params[key] -= learning_rate * momentum
                
                # Store updated velocity for next iteration
                self.m[momentum_key] = momentum