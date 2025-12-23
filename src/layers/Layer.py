"""
Layer - Documented Version

This is the abstract base class for all neural network layers.
It defines the interface that all concrete layer implementations must follow,
ensuring consistency across different layer types (Dense, Convolutional, etc.).
"""

import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    """
    Abstract Base Class for Neural Network Layers
    
    This class defines the standard interface that all layer types must implement.
    It provides the basic structure for parameters, gradients, and caching needed
    during forward and backward propagation.
    
    All concrete layer implementations (DenseLayer, ConvolutionalLayer, etc.)
    must inherit from this class and implement the three abstract methods:
        1. initialize_parameters
        2. forward_pass
        3. backward_pass
    
    Attributes:
        name (str): Unique identifier for the layer
                    Useful for debugging and model visualization
        
        params (dict): Dictionary storing learnable parameters
                       Common keys include:
                       - "W": weight matrix/tensor
                       - "b": bias vector/tensor
                       - "gamma", "beta": batch normalization parameters
                       Empty dict for layers without learnable parameters
        
        grads (dict): Dictionary storing gradients of parameters
                      Keys match those in params (e.g., "dW", "db")
                      Computed during backward_pass
                      Empty dict for layers without learnable parameters
        
        cache (dict): Dictionary storing values needed for backward pass
                      Common keys include:
                      - "A_prev": input activations
                      - "Z": linear outputs (before activation)
                      - Other intermediate values needed for gradient computation
                      Populated during forward_pass, used during backward_pass
    
    Design Pattern:
        This follows the Template Method pattern - the base class defines
        the structure (attributes and method signatures) while concrete
        subclasses provide the specific implementations.
    """
    
    def __init__(self, name: str) -> None:
        """
        Initialize the base layer.
        
        Sets up the core data structures that all layers need:
        - params: for storing learnable parameters
        - grads: for storing computed gradients
        - cache: for storing intermediate values during forward pass
        
        Parameters:
            name (str): Identifier for this layer
                       Should be descriptive (e.g., "conv1", "fc2", "bn3")
                       
        Returns:
            None
        """
        self.name = name
        self.params = {}
        self.grads = {}
        self.cache = {}
    
    @abstractmethod
    def initialize_parameters(self, input_dim: int | tuple, seed: int = 0) -> int | tuple:
        """
        Initialize layer parameters based on input dimensions.
        
        This method must be implemented by all concrete layer classes.
        It should:
            1. Set up learnable parameters (W, b, etc.) in self.params
            2. Initialize gradients arrays in self.grads (optional, can be done in backward_pass)
            3. Return the output dimension(s) for the next layer
        
        The initialization strategy typically depends on:
            - Input dimensions
            - Output dimensions (specified in layer constructor)
            - Type of activation function used after this layer
              (e.g., He initialization for ReLU, Xavier for tanh/sigmoid)
        
        Parameters:
            input_dim (int | tuple): Dimensions of the input to this layer
                                     - int: for 1D data (e.g., Dense layers)
                                       Example: 784 for flattened 28×28 images
                                     - tuple: for multi-dimensional data (e.g., Conv layers)
                                       Example: (3, 224, 224) for RGB images
                                       
            seed (int, optional): Random seed for reproducible initialization
                                 Default: 0
                                 Ensures same initialization across runs
                                 
        Returns:
            int | tuple: Output dimension(s) of this layer
                        Used as input_dim for the next layer during model construction
                        - int: for layers outputting 1D data
                        - tuple: for layers outputting multi-dimensional data
                        
        Example:
            >>> layer = DenseLayer(output_dim=128, next_act_layer="relu")
            >>> output_dim = layer.initialize_parameters(input_dim=784, seed=42)
            >>> print(output_dim)  # 128
        """
        pass
    
    @abstractmethod
    def forward_pass(self, A_prev: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the layer.
        
        This method must be implemented by all concrete layer classes.
        It should:
            1. Cache necessary values in self.cache for backward pass
               (at minimum, A_prev is typically cached)
            2. Compute the layer's output using the input and parameters
            3. Return the output activations
        
        The forward pass implements the layer's mathematical operation:
            - Dense layer: Z = W @ A_prev + b
            - Conv layer: Z = convolution(A_prev, W) + b
            - Activation layer: A = activation(A_prev)
            - Flatten layer: A = reshape(A_prev)
        
        Parameters:
            A_prev (np.ndarray): Input activations from the previous layer
                                Shape depends on layer type:
                                - Dense: (input_dim, batch_size)
                                - Conv: (batch_size, channels, height, width)
                                - etc.
                                
        Returns:
            np.ndarray: Output activations of this layer
                       Shape depends on layer type and configuration
                       This becomes A_prev for the next layer
                       
        Example:
            >>> A_prev = np.random.randn(784, 32)  # 32 samples, 784 features
            >>> dense_layer = DenseLayer(output_dim=128, next_act_layer="relu")
            >>> Z = dense_layer.forward_pass(A_prev)
            >>> print(Z.shape)  # (128, 32)
        """
        pass
    
    @abstractmethod
    def backward_pass(self, dA: np.ndarray) -> np.ndarray:
        """
        Perform backward propagation through the layer.
        
        This method must be implemented by all concrete layer classes.
        It should:
            1. Compute gradients of parameters (dW, db, etc.) and store in self.grads
            2. Compute gradient with respect to input (dA_prev)
            3. Return dA_prev to be used by the previous layer
        
        The backward pass uses the chain rule to compute gradients:
            - Uses cached values from forward_pass
            - Uses gradient flowing back from next layer (dA)
            - Computes local gradients and propagates them backwards
        
        For layers with parameters:
            self.grads["dW"] = gradient of loss w.r.t. weights
            self.grads["db"] = gradient of loss w.r.t. biases
        
        For all layers:
            dA_prev = gradient of loss w.r.t. input (to pass to previous layer)
        
        Parameters:
            dA (np.ndarray): Gradient of loss with respect to this layer's output
                            Shape matches the output of forward_pass
                            Flows backwards from the next layer
                            
        Returns:
            np.ndarray: Gradient of loss with respect to this layer's input (dA_prev)
                       Shape matches A_prev from forward_pass
                       This becomes dA for the previous layer
                       
        Example:
            >>> dA = np.random.randn(128, 32)  # Gradient from next layer
            >>> dA_prev = dense_layer.backward_pass(dA)
            >>> print(dA_prev.shape)  # (784, 32)
            >>> print(dense_layer.grads["dW"].shape)  # (128, 784)
            >>> print(dense_layer.grads["db"].shape)  # (128, 1)
        """
        pass