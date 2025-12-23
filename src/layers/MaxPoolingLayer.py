"""
MaxPoolingLayer - Documented Version

This layer implements max pooling, a downsampling operation commonly used in CNNs.
Max pooling reduces spatial dimensions by taking the maximum value within each
pooling window, helping to:
    1. Reduce computational cost and memory usage
    2. Provide translation invariance
    3. Extract dominant features
    4. Control overfitting

The layer slides a pooling window over the input and outputs the maximum value
in each window.
"""

from src.layers import Layer
import numpy as np


class MaxPoolingLayer(Layer):
    """
    Max Pooling Layer
    
    Applies max pooling operation to reduce spatial dimensions of feature maps.
    For each pooling window, outputs the maximum value and remembers its position
    for gradient backpropagation.
    
    Operation:
        - Slides a pooling window (e.g., 2×2) over the input with a given stride
        - For each window position, outputs the maximum value
        - During backward pass, gradients flow only to the maximum value's position
    
    Example with 2×2 pooling, stride 2:
        Input (4×4):           Output (2×2):
        [[1, 3, 2, 4],         [[3, 4],
         [5, 6, 1, 2],    →     [8, 9]]
         [7, 8, 0, 1],
         [2, 3, 9, 5]]
    
    Attributes:
        name (str): Layer identifier
        pool_size (tuple): Size of pooling window (height, width)
                          Common values: (2, 2), (3, 3)
        stride (int): Step size for sliding the pooling window
                     Common value: 2 (non-overlapping windows)
        index_map (dict): Maps output positions to input positions of max values
                         Key format: "b_c_h_w" (batch_channel_height_width)
                         Value: (in_h, in_w) - coordinates of max value in input
                         Used for routing gradients during backward pass
        params (dict): Empty - no learnable parameters
        cache (dict): Cached values for backward pass
            - A_prev (np.ndarray): Input feature maps
        grads (dict): Cached gradients
            - dA (np.ndarray): Gradient w.r.t. output
    """
    
    def __init__(self, name: str, pool_size: tuple = (2, 2), stride: int = 2) -> None:
        """
        Initialize the Max Pooling Layer.
        
        Parameters:
            name (str): Layer identifier
            pool_size (tuple, optional): Pooling window size (height, width)
                                        Default: (2, 2) - standard 2×2 pooling
            stride (int, optional): Step size for sliding window
                                   Default: 2 - non-overlapping windows
                                   If stride < pool_size: overlapping windows
                                   If stride = pool_size: non-overlapping (standard)
                                   If stride > pool_size: gaps between windows
                                   
        Returns:
            None
        """
        super().__init__(name)
        self.pool_size = pool_size
        self.stride = stride
        self.index_map = {}
    
    def initialize_parameters(self, input_dim: tuple, seed: int = 0) -> tuple:
        """
        Calculate output dimensions after max pooling.
        
        No parameters to initialize since max pooling has no learnable weights.
        Computes output spatial dimensions using the formula:
            out_size = floor((in_size - pool_size) / stride) + 1
        
        Parameters:
            input_dim (tuple): Input dimensions (channels, height, width)
                              Format: (C, H, W)
            seed (int, optional): Random seed (not used, included for API consistency)
                                 Default: 0
            
        Returns:
            tuple: Output dimensions (channels, out_height, out_width)
                   Channels remain unchanged, spatial dimensions are reduced
                   
        Example:
            >>> # Input: 64 channels, 28×28 spatial
            >>> # Pool size: 2×2, stride: 2
            >>> layer = MaxPoolingLayer("pool1", pool_size=(2, 2), stride=2)
            >>> output_dim = layer.initialize_parameters((64, 28, 28))
            >>> print(output_dim)  # (64, 14, 14)
        """
        c, in_h, in_w = input_dim
        
        # Calculate output spatial dimensions
        out_h = int(np.floor((in_h - self.pool_size[0]) / self.stride)) + 1
        out_w = int(np.floor((in_w - self.pool_size[1]) / self.stride)) + 1
        
        return (c, out_h, out_w)
    
    def forward_pass(self, A_prev: np.ndarray) -> np.ndarray:
        """
        Perform forward pass: apply max pooling operation.
        
        For each pooling window:
            1. Find the maximum value within the window
            2. Store the position (indices) of the maximum value
            3. Output the maximum value
        
        The index_map is crucial for backward pass - it remembers where each
        maximum value came from so gradients can be routed correctly.
        
        Algorithm:
            For each output position (b, c, out_h, out_w):
                - Define pooling window in input
                - Find max value and its coordinates
                - Store max value in output
                - Store coordinates in index_map
        
        Parameters:
            A_prev (np.ndarray): Input feature maps
                                Shape: (batch_size, channels, in_height, in_width)
                                
        Returns:
            np.ndarray: Pooled output
                       Shape: (batch_size, channels, out_height, out_width)
                       where out_height = floor((in_height - pool_h) / stride) + 1
                             out_width = floor((in_width - pool_w) / stride) + 1
                             
        Example:
            >>> A_prev = np.random.randn(32, 64, 28, 28)  # 32 samples, 64 channels
            >>> pool = MaxPoolingLayer("pool1", pool_size=(2, 2), stride=2)
            >>> A = pool.forward_pass(A_prev)
            >>> print(A.shape)  # (32, 64, 14, 14)
        """
        self.cache["A_prev"] = A_prev
        
        batch_size, channels, in_h, in_w = A_prev.shape
        
        # Calculate output dimensions
        out_h = int(np.floor((in_h - self.pool_size[0]) / self.stride)) + 1
        out_w = int(np.floor((in_w - self.pool_size[1]) / self.stride)) + 1
        
        # Pre-allocate output array
        A = np.zeros((batch_size, channels, out_h, out_w))
        
        # Iterate over all output positions
        for b in range(batch_size):
            for c in range(channels):
                for out_x in range(out_h):
                    for out_y in range(out_w):
                        # Create unique key for this output position
                        map_key = f"{b}_{c}_{out_x}_{out_y}"
                        
                        # Find maximum in current pooling window
                        max_val = -np.inf
                        max_indices = (0, 0)
                        
                        # Scan pooling window
                        for pool_f_x in range(self.pool_size[0]):
                            for pool_f_y in range(self.pool_size[1]):
                                # Calculate input position
                                in_x = out_x * self.stride + pool_f_x
                                in_y = out_y * self.stride + pool_f_y
                                
                                curr_val = A_prev[b, c, in_x, in_y]
                                
                                # Update max if current value is larger
                                if curr_val >= max_val:
                                    max_val = curr_val
                                    max_indices = (in_x, in_y)
                        
                        # Store max value in output
                        A[b, c, out_x, out_y] = max_val
                        
                        # Store position of max value for backward pass
                        self.index_map[map_key] = max_indices
        
        return A
    
    def backward_pass(self, dA: np.ndarray) -> np.ndarray:
        """
        Perform backward pass: route gradients to maximum value positions.
        
        Key insight: In max pooling, only the maximum value in each window
        contributed to the output, so only its position receives the gradient.
        All other positions in the window get zero gradient.
        
        Algorithm:
            For each output position receiving gradient dA[b, c, out_h, out_w]:
                1. Look up which input position had the max value (using index_map)
                2. Route the entire gradient to that position
                3. All other positions in that pooling window remain zero
        
        This is sometimes called "routing" - the gradient is routed through
        the "winning" (maximum) position.
        
        Optimization note:
            We don't need to iterate over the pooling window during backward pass.
            Since dA_prev is initialized to zeros, we only need to update the
            positions that had maximum values. This is more efficient than
            iterating over all window positions.
        
        Parameters:
            dA (np.ndarray): Gradient of loss with respect to pooling output
                            Shape: (batch_size, channels, out_height, out_width)
                            
        Returns:
            np.ndarray: Gradient of loss with respect to input (dA_prev)
                       Shape: (batch_size, channels, in_height, in_width)
                       Sparse - only max positions have non-zero gradients
                       
        Example:
            >>> dA = np.random.randn(32, 64, 14, 14)
            >>> dA_prev = pool.backward_pass(dA)
            >>> print(dA_prev.shape)  # (32, 64, 28, 28)
            >>> # Note: dA_prev is sparse - only ~25% of values are non-zero
            >>> # (one per 2×2 pooling window)
        """
        self.grads["dA"] = dA
        
        # Initialize gradient array (all zeros)
        dA_prev = np.zeros_like(self.cache["A_prev"])
        
        batch_size, channels, out_h, out_w = dA.shape
        
        # Route gradients to max value positions
        for b in range(batch_size):
            for c in range(channels):
                for out_x in range(out_h):
                    for out_y in range(out_w):
                        # Look up which input position had the max value
                        map_key = f"{b}_{c}_{out_x}_{out_y}"
                        max_in_x, max_in_y = self.index_map[map_key]
                        
                        # Get gradient from this output position
                        contribution = dA[b, c, out_x, out_y]
                        
                        # OPTIMIZATION: We don't need to iterate over the pooling window:
                        #
                        # for pool_f_x in range(self.pool_size[0]):
                        #     for pool_f_y in range(self.pool_size[1]):
                        #         ...
                        #
                        # Because dA_prev is already initialized to zeros, and only
                        # the max position should receive the gradient. All other
                        # positions in the window remain zero.
                        
                        # Route gradient to the max position
                        dA_prev[b, c, max_in_x, max_in_y] += contribution
        
        # Clear index map after use (prepare for next forward pass)
        self.index_map = {}
        
        return dA_prev