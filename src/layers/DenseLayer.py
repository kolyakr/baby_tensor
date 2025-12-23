"""
DenseLayer - Documented Version

This layer implements a fully connected (dense) neural network layer.
Also known as a linear layer or affine transformation, it performs the operation:
    Z = W @ A_prev + b

where W is the weight matrix, b is the bias vector, and A_prev is the input.
"""

from src.layers.Layer import Layer
from src.utils.initialization import get_initialization
from src.utils.activations import ActivationType, get_activation
import numpy as np


class DenseLayer(Layer):
    """
    Dense (Fully Connected) Layer
    
    Implements a fully connected layer where every input neuron is connected
    to every output neuron. This is the fundamental building block of 
    traditional feedforward neural networks.
    
    The layer performs the linear transformation:
        Z = W @ A + b
    
    where:
        - W: weight matrix (output_dim × input_dim)
        - A: input activations (input_dim × batch_size)
        - b: bias vector (output_dim × 1)
        - Z: output (output_dim × batch_size)
    
    Attributes:
        name (str): Layer identifier
        output_dim (int): Number of output neurons/features
        next_act_layer (ActivationType): Type of activation function in next layer
                                         (used to select weight initialization strategy)
        params (dict): Layer parameters
            - W (np.ndarray): Weight matrix, shape (output_dim, input_dim)
            - b (np.ndarray): Bias vector, shape (output_dim, 1)
        cache (dict): Cached values for backward pass
            - A_prev (np.ndarray): Input activations from previous layer
        grads (dict): Gradients computed during backward pass
            - dW (np.ndarray): Gradient of W, shape (output_dim, input_dim)
            - db (np.ndarray): Gradient of b, shape (output_dim, 1)
            - dZ (np.ndarray): Gradient of Z (cached), shape (output_dim, batch_size)
    """
    
    def __init__(
        self, 
        output_dim: int, 
        next_act_layer: ActivationType, 
        name: str = "Dense"
    ) -> None:
        """
        Initialize the Dense Layer.
        
        Parameters:
            output_dim (int): Number of output neurons/features
            next_act_layer (ActivationType): Activation function type for next layer
                                             ("relu" uses He initialization,
                                              others use Xavier initialization)
            name (str, optional): Layer identifier. Default: "Dense"
            
        Returns:
            None
        """
        super().__init__(name)
        self.output_dim = output_dim
        self.next_act_layer = next_act_layer
    
    def initialize_parameters(self, input_dim: int, seed: int = 0) -> int:
        """
        Initialize weights and biases for the dense layer.
        
        Weights are initialized using:
            - He initialization if next activation is ReLU (better for ReLU networks)
            - Xavier initialization for other activations (tanh, sigmoid, etc.)
        
        Biases are initialized to zero.
        Gradient arrays are also pre-allocated and initialized to zero.
        
        Parameters:
            input_dim (int): Number of input features/neurons
            seed (int, optional): Random seed for reproducibility. Default: 0
            
        Returns:
            int: Output dimension (to be used as input_dim for next layer)
        """
        np.random.seed(seed)
        
        # Select initialization strategy based on next activation
        init_name = "he" if self.next_act_layer == "relu" else "xavier"
        initialization = get_initialization(init_name)
        
        W_shape = (self.output_dim, input_dim)
        
        # Initialize weights using selected strategy
        self.params["W"] = initialization(W_shape)
        
        # Initialize biases to zero
        self.params["b"] = np.zeros((self.output_dim, 1))
        
        # Pre-allocate gradient arrays
        self.grads["dW"] = np.zeros(W_shape)
        self.grads["db"] = np.zeros((self.output_dim, 1))
        
        # Return output dimension for next layer's initialization
        return self.output_dim
    
    def forward_pass(self, A_prev: np.ndarray) -> np.ndarray:
        """
        Perform forward pass of the dense layer.
        
        Computes the linear transformation:
            Z = W @ A_prev + b
        
        The bias term is broadcasted across all samples in the batch.
        
        Note: The implementation uses matrix multiplication to broadcast bias:
            b @ ones(1, m) = b repeated m times horizontally
        This is equivalent to: Z = W @ A_prev + b (with broadcasting)
        
        Parameters:
            A_prev (np.ndarray): Input activations from previous layer
                Shape: (input_dim, batch_size)
                where input_dim is the number of input features
                and batch_size (m) is the number of samples
                
        Returns:
            np.ndarray: Linear output (before activation)
                Shape: (output_dim, batch_size)
        """
        # Cache input for backward pass
        self.cache["A_prev"] = A_prev
        
        # Compute linear transformation: Z = W @ A_prev + b
        # The second term broadcasts bias across all batch samples
        Z = np.matmul(self.params["W"], A_prev) + \
            np.matmul(self.params["b"], np.ones((1, A_prev.shape[1])))
        
        return Z
    
    def backward_pass(self, dZ: np.ndarray) -> np.ndarray:
        """
        Perform backward pass to compute gradients.
        
        Given the gradient of the loss with respect to the layer output (dZ),
        computes three gradients using the chain rule:
        
        1. dW: gradient with respect to weights
           dW = (1/m) * dZ @ A_prev^T
           
        2. db: gradient with respect to biases
           db = (1/m) * sum(dZ, axis=1)
           
        3. dA_prev: gradient with respect to input (to pass to previous layer)
           dA_prev = W^T @ dZ
        
        All gradients are averaged over the batch (divided by m).
        
        Parameters:
            dZ (np.ndarray): Gradient of loss with respect to layer output
                Shape: (output_dim, batch_size)
                
        Returns:
            np.ndarray: Gradient of loss with respect to layer input (dA_prev)
                Shape: (input_dim, batch_size)
        """
        # Get batch size
        m = dZ.shape[1]
        
        # Cache dZ (may be useful for analysis/debugging)
        self.grads["dZ"] = dZ
        
        # Gradient with respect to weights: dL/dW = (1/m) * dZ @ A_prev^T
        self.grads["dW"] = np.matmul(dZ, self.cache["A_prev"].transpose()) / m
        
        # Gradient with respect to biases: dL/db = (1/m) * sum over batch
        self.grads["db"] = np.sum(dZ, axis=1, keepdims=True) / m
        
        # Gradient with respect to input: dL/dA_prev = W^T @ dZ
        dA_prev = np.matmul(self.params["W"].transpose(), dZ)
        
        return dA_prev