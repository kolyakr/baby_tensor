"""
NeuralNetwork - Documented Version

This is the main class that orchestrates the entire deep learning pipeline.
It combines layers, loss functions, and optimizers into a trainable model.

The NeuralNetwork class handles:
    1. Model construction (building the computational graph)
    2. Forward propagation (computing predictions)
    3. Backward propagation (computing gradients)
    4. Training loop (optimization over multiple epochs)
    5. Learning rate scheduling (adaptive learning rate reduction)

This class follows the Facade design pattern - it provides a simple interface
to the complex subsystems of layers, losses, and optimizers.
"""

import numpy as np
from src.layers import Layer
from src.utils.losses import LossType, get_loss
from src.optimizers import OptimizerType, get_optimizer


class NeuralNetwork:
    """
    Neural Network Model
    
    High-level class that orchestrates the complete deep learning workflow.
    Combines layers, loss function, and optimizer into a trainable model.
    
    The class provides methods for:
        - Building the network (initializing parameters)
        - Forward propagation (making predictions)
        - Backward propagation (computing gradients)
        - Training (optimization loop with mini-batches)
    
    Architecture:
        Input → Layer1 → Layer2 → ... → LayerN → Output
                  ↓        ↓              ↓
               [W1, b1] [W2, b2]      [WN, bN]
    
    Training flow:
        1. Forward pass: X → predictions (ŷ)
        2. Compute loss: L(ŷ, y)
        3. Backward pass: compute gradients ∂L/∂θ
        4. Update: θ = θ - α * ∂L/∂θ
        5. Repeat for multiple epochs
    
    Features:
        - Supports any layer architecture (dense, conv, pooling, etc.)
        - Flexible loss functions (MSE, cross-entropy, etc.)
        - Multiple optimizer choices (SGD, Momentum, Adam, etc.)
        - Mini-batch training with shuffling
        - Automatic learning rate decay based on loss plateau
        - Progress monitoring (loss and accuracy per batch)
    
    Attributes:
        layers (list[Layer]): Ordered list of layers in the network
                             Input → ... → Output
        loss_type (LossType): Type of loss function to use
                             Examples: "mse", "cce" (categorical cross-entropy)
        optimizer (Optimizer): Optimizer instance for parameter updates
                              Created from optimizer_type
        seed (int): Random seed for reproducible parameter initialization
    """
    
    def __init__(
        self,
        layers: list[Layer],
        loss_type: LossType,
        optimizer_type: OptimizerType,
        seed: int
    ) -> None:
        """
        Initialize the Neural Network.
        
        Sets up the model architecture but does NOT initialize parameters yet.
        Parameters are initialized later in the build() method once input
        dimensions are known.
        
        Parameters:
            layers (list[Layer]): Sequential list of layers
                                 Should include all layers from input to output
                                 Example: [Dense(128), ReLU(), Dense(10), Softmax()]
            loss_type (LossType): Loss function identifier
                                 Common values:
                                 - "mse": Mean Squared Error (regression)
                                 - "cce": Categorical Cross-Entropy (classification)
                                 - "bce": Binary Cross-Entropy
            optimizer_type (OptimizerType): Optimizer identifier
                                           Common values:
                                           - "sgd": Vanilla gradient descent
                                           - "momentum": SGD with momentum
                                           - "adam": Adam optimizer
            seed (int): Random seed for parameter initialization
                       Ensures reproducibility across runs
                       
        Returns:
            None
            
        Example:
            >>> from src.layers import DenseLayer, ReLULayer, SoftmaxLayer
            >>> layers = [
            ...     DenseLayer(128, "relu", "dense1"),
            ...     ReLULayer("relu1"),
            ...     DenseLayer(64, "relu", "dense2"),
            ...     ReLULayer("relu2"),
            ...     DenseLayer(10, "softmax", "output"),
            ...     SoftmaxLayer("softmax")
            ... ]
            >>> model = NeuralNetwork(
            ...     layers=layers,
            ...     loss_type="cce",
            ...     optimizer_type="adam",
            ...     seed=42
            ... )
        """
        self.layers = layers
        self.loss_type = loss_type
        self.optimizer = get_optimizer(optimizer_type)
        self.seed = seed
    
    def build(self, input_dim: int | tuple) -> None:
        """
        Build the network by initializing all layer parameters.
        
        Chains parameter initialization through all layers. Each layer's
        output dimension becomes the next layer's input dimension.
        
        This method must be called before training or prediction.
        
        Parameters:
            input_dim (int | tuple): Dimensions of the input data
                                    - int: for 1D data (e.g., flattened images)
                                      Example: 784 for MNIST (28×28 flattened)
                                    - tuple: for multi-dimensional data
                                      Example: (3, 224, 224) for RGB images
                                      
        Returns:
            None (parameters are initialized in-place in layer.params)
            
        Algorithm:
            current_dim = input_dim
            for each layer:
                current_dim = layer.initialize_parameters(current_dim, seed)
            
            This ensures dimensional compatibility between consecutive layers.
            
        Example:
            >>> # For MNIST: 28×28 grayscale images flattened to 784
            >>> model.build(input_dim=784)
            >>> 
            >>> # For CIFAR-10: 32×32 RGB images
            >>> model.build(input_dim=(3, 32, 32))
            >>> 
            >>> # Now model is ready for training/prediction
        """
        temp_input_dim = input_dim
        
        # Chain initialization through all layers
        for layer in self.layers:
            temp_input_dim = layer.initialize_parameters(temp_input_dim, self.seed)
    
    def forward_pass(self, X: np.ndarray) -> np.ndarray:
        """
        Perform forward propagation through the network.
        
        Passes input through all layers sequentially, with each layer's
        output becoming the next layer's input.
        
        Each layer caches values needed for backward pass.
        
        Parameters:
            X (np.ndarray): Input data
                           Shape depends on data type and batch axis:
                           - Dense networks: (features, batch_size)
                           - CNNs: (batch_size, channels, height, width)
                           
        Returns:
            np.ndarray: Network output (predictions)
                       Shape depends on task:
                       - Classification: (num_classes, batch_size)
                       - Regression: (output_dim, batch_size)
                       
        Algorithm:
            A = X
            for each layer:
                A = layer.forward_pass(A)
            return A
            
        Example:
            >>> # Classification on MNIST
            >>> X_batch = np.random.randn(784, 32)  # 32 samples
            >>> predictions = model.forward_pass(X_batch)
            >>> print(predictions.shape)  # (10, 32) for 10 classes
            >>> 
            >>> # Get predicted classes
            >>> predicted_classes = np.argmax(predictions, axis=0)
        """
        A = X
        
        # Pass through each layer sequentially
        for layer in self.layers:
            A = layer.forward_pass(A)
        
        return A
    
    def backward_pass(self, y_true: np.ndarray, y_hat: np.ndarray) -> None:
        """
        Perform backward propagation to compute gradients.
        
        Computes gradients of the loss with respect to all parameters
        by applying the chain rule backwards through the network.
        
        Special handling for categorical cross-entropy (CCE):
            When using softmax + CCE, the combined derivative simplifies
            to (y_hat - y_true), so we pass y_true directly to the last layer.
        
        Parameters:
            y_true (np.ndarray): True labels
                                Shape: (num_classes, batch_size) for classification
                                      (output_dim, batch_size) for regression
            y_hat (np.ndarray): Predicted values (from forward_pass)
                               Same shape as y_true
                               
        Returns:
            None (gradients are stored in layer.grads dictionaries)
            
        Algorithm:
            if loss_type == "cce":
                # Softmax + CCE combined derivative
                dA = y_true  # Will be used as (y_hat - y_true) in softmax layer
            else:
                # Standard loss derivative
                dA = ∂L/∂y_hat
            
            for each layer (in reverse order):
                dA = layer.backward_pass(dA)
            
        Note on CCE:
            The SoftmaxLayer.backward_pass() expects y_true (not dA) and
            internally computes dZ = (y_hat - y_true) / m. This is why we
            pass y_true directly when loss_type == "cce".
            
        Example:
            >>> y_true = np.array([[0, 1], [1, 0], [0, 0]])  # One-hot encoded
            >>> y_hat = model.forward_pass(X_batch)
            >>> model.backward_pass(y_true, y_hat)
            >>> 
            >>> # Now gradients are available in layer.grads
            >>> # Example: layers[0].grads["dW"], layers[0].grads["db"]
        """
        _, loss_derivative = get_loss(self.loss_type)
        
        # Special case: Categorical Cross-Entropy with Softmax
        # The combined derivative is simply (y_hat - y_true)
        # So we pass y_true directly to the softmax layer
        if self.loss_type == "cce":
            dA = y_true
        else:
            # Standard loss derivative
            dA = loss_derivative(y_hat, y_true)
        
        # Propagate gradients backwards through all layers
        for layer in self.layers[::-1]:  # Reverse order
            dA = layer.backward_pass(dA)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        decrease_after: int = 10,
        gamma: float = 0.9,
        epochs: int = 100,
        batch_size: int = 34,
        learning_rate: float = 0.01
    ) -> None:
        """
        Train the neural network using mini-batch gradient descent.
        
        Implements the complete training loop with:
            - Mini-batch processing with shuffling
            - Learning rate decay on loss plateau
            - Progress monitoring (loss and accuracy)
        
        Training algorithm:
            for each epoch:
                shuffle training data
                for each mini-batch:
                    1. Forward pass
                    2. Backward pass (compute gradients)
                    3. Update parameters (optimizer step)
                    4. Monitor loss and accuracy
                    5. Reduce learning rate if loss plateaus
        
        Learning rate scheduling:
            Uses a simple plateau-based decay:
            - Tracks best loss seen so far
            - If loss doesn't improve for 'decrease_after' steps
            - Multiply learning rate by gamma (e.g., 0.9)
            - Reset counter and continue training
        
        Parameters:
            X_train (np.ndarray): Training data
                                 Shape options:
                                 - Dense: (features, num_samples)
                                 - CNN: (num_samples, channels, height, width)
            y_train (np.ndarray): Training labels (one-hot encoded)
                                 Shape: (num_classes, num_samples)
            decrease_after (int, optional): Number of steps without improvement
                                           before reducing learning rate
                                           Default: 10
            gamma (float, optional): Learning rate decay factor
                                    new_lr = old_lr * gamma
                                    Default: 0.9 (10% reduction)
            epochs (int, optional): Number of complete passes through dataset
                                   Default: 100
            batch_size (int, optional): Number of samples per mini-batch
                                       Default: 34
                                       Common values: 16, 32, 64, 128, 256
            learning_rate (float, optional): Initial learning rate
                                            Default: 0.01
                                            Will be decayed during training
                                            
        Returns:
            None (model parameters are updated in-place)
            
        Side effects:
            - Prints training progress (loss and accuracy per batch)
            - Prints learning rate reductions
            - Updates all layer parameters via optimizer
            
        Example:
            >>> # Prepare data
            >>> X_train = np.random.randn(784, 60000)  # MNIST
            >>> y_train = np.eye(10)[:, np.random.randint(0, 10, 60000)]  # One-hot
            >>> 
            >>> # Build and train model
            >>> model.build(input_dim=784)
            >>> model.train(
            ...     X_train=X_train,
            ...     y_train=y_train,
            ...     epochs=50,
            ...     batch_size=128,
            ...     learning_rate=0.001,
            ...     decrease_after=5,
            ...     gamma=0.9
            ... )
            >>> 
            >>> # Output:
            >>> # Epoch 0, Batch 0: Loss = 2.3045 | Acc = 12.50%
            >>> # Epoch 0, Batch 1: Loss = 2.2891 | Acc = 15.62%
            >>> # ...
            >>> # --- Learning Rate reduced to 0.000900 ---
            >>> # ...
        
        Training tips:
            - Start with learning_rate=0.01 or 0.001
            - Smaller batch_size: more frequent updates, noisier gradients
            - Larger batch_size: stabler gradients, faster per epoch
            - decrease_after: smaller = more aggressive LR decay
            - gamma: typical values 0.9 (10% reduction) or 0.5 (50% reduction)
            - Monitor accuracy: should increase over epochs
            - Monitor loss: should decrease (with possible plateaus)
        """
        # Determine batch axis based on data shape
        if len(X_train.shape) == 4:
            batch_axis = 0  # CNN format: (batch, channels, height, width)
        else:
            batch_axis = 1  # Dense format: (features, batch)
        
        n = X_train.shape[batch_axis]  # Total number of samples
        idx = range(n)
        loss_fnc, _ = get_loss(self.loss_type)
        
        # Learning rate scheduling variables
        best_loss = np.inf
        loss_degradation_steps = 0
        
        # Training loop
        for i in range(epochs):
            # Shuffle data at start of each epoch
            shuffled_idx = np.random.permutation(idx)
            
            # Mini-batch loop
            for start_idx in range(0, n, batch_size):
                iteration = start_idx // batch_size
                
                # Extract mini-batch indices
                end_index = start_idx + batch_size
                batch_idx = shuffled_idx[start_idx:end_index]
                
                # Extract mini-batch data
                if batch_axis == 1:
                    # Dense format: (features, batch)
                    X_train_batch = X_train[:, batch_idx]
                else:
                    # CNN format: (batch, channels, height, width)
                    X_train_batch = X_train[batch_idx, :, :, :]
                
                y_train_batch = y_train[:, batch_idx]
                
                # Training step: forward → backward → update
                y_hat = self.forward_pass(X_train_batch)
                self.backward_pass(y_train_batch, y_hat)
                self.optimizer.update_layers(self.layers, learning_rate)
                
                # Compute metrics
                loss = loss_fnc(y_hat, y_train_batch)
                acc = np.mean(np.argmax(y_hat, axis=0) == np.argmax(y_train_batch, axis=0))
                print(f"Epoch {i}, Batch {iteration}: Loss = {loss:.4f} | Acc = {acc:.2%}")
                
                # Learning rate decay logic
                if loss < best_loss:
                    best_loss = loss
                    loss_degradation_steps = 0
                else:
                    loss_degradation_steps += 1
                
                # Reduce learning rate if no improvement for 'decrease_after' steps
                if loss_degradation_steps == decrease_after:
                    learning_rate *= gamma
                    loss_degradation_steps = 0
                    print(f"--- Learning Rate reduced to {learning_rate:.6f} ---")