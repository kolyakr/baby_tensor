# Baby Tensor: A Lightweight Deep Learning Framework

Baby Tensor is a modular, from-scratch implementation of a deep learning framework built using NumPy. It is designed for educational purposes to demonstrate the internal mechanics of forward and backward propagation, parameter optimization, and various neural network architectures.

## Project Development

This project was a rigorous undertaking that spanned nearly **two weeks** of continuous development. The goal was to move beyond high-level libraries and build the foundational "gears" of a neural network manually.

One of the primary challenges was handling the **MNIST dataset**. Implementing the logic for reading the raw IDX-formatted files and ensuring the data shapes remained consistent across convolutional, pooling, and flattening layers required significant debugging. Despite these hurdles, the current model achieves nearly **90% test accuracy**. While this is a strong result for a custom-built engine, there is room for further optimization. Currently, training from scratch is a **lengthy process** due to the computational overhead of Python loops, and I plan to explore further optimizations to improve training efficiency and accuracy.

---

## Project Structure

- **`src/`**: The core engine of the framework.
- **`layers/`**: Implements various layer types including Dense, Convolutional, MaxPooling, Flatten, and BatchNorm.
- **`optimizers/`**: Optimization algorithms including Vanilla Gradient Descent, Momentum, and Adam.
- **`utils/`**: Utility functions for activations (ReLU, Sigmoid, Softmax), loss functions (MSE, Cross-Entropy), and weight initialization (He, Xavier).

- **`data/`**: Data loading utilities, specifically for the MNIST dataset.
- **`diagrams/`**: Visual documentation of the class architecture.

---

## Core Features

- **Modular Design**: All layers inherit from an abstract `Layer` class, ensuring a consistent interface for `forward_pass` and `backward_pass`.
- **Optimized Convolutions**: Computationally heavy operations in the `ConvolutionalLayer` utilize **Numba JIT** compilation to bridge the performance gap between Python and low-level languages.
- **Advanced Architectures**: Supports modern components like `ResidualBlock` (skip connections) to mitigate vanishing gradients in deep networks.
- **Adaptive Training**: The `NeuralNetwork` class features a plateau-based learning rate scheduler that automatically reduces the learning rate by a factor of `gamma` when the loss stops improving.

---

## Testing on MNIST (`cnn.ipynb`)

The framework's capabilities are demonstrated through a Convolutional Neural Network (CNN) trained on the MNIST digit classification dataset.

### 1. Data Preparation

The `MnistDataloader` extracts training and test images from raw binary files.

- **Normalization**: Pixel values are scaled to the `[0, 1]` range by dividing by 255.0 to facilitate convergence.
- **Reshaping**: Images are formatted to `(batch, 1, 28, 28)` for compatibility with 2D convolution filters.
- **One-Hot Encoding**: Integer labels (0-9) are converted into 10-dimensional probability vectors.

### 2. Model Architecture

The testing notebook utilizes the following stack:

1. **Convolutional Layers**: Two stages of convolution for feature extraction.
2. **Activations**: ReLU layers to introduce non-linearity.
3. **Downsampling**: MaxPooling layers to reduce spatial dimensions and computational load.
4. **Flattening**: A transition layer to convert 2D feature maps into a 1D vector.
5. **Output**: A Dense layer followed by Softmax to produce final class probabilities.

### 3. Training and Evaluation

- **Loss & Optimizer**: The model uses Categorical Cross-Entropy loss and the **Adam optimizer** for adaptive per-parameter learning rates.
- **Persistence**: Trained models are saved using `pickle`. This is essential because training from scratch takes a **very long time**; saving allows for immediate evaluation or fine-tuning without restarting the entire two-week development effort.

---

## References & Further Reading

The theoretical foundations and mathematical derivations used in this project are based on:

- **Prince, S. J. D. (2023). _Understanding Deep Learning_. MIT Press.** [udlbook.com](https://udlbook.github.io/udlbook/)
