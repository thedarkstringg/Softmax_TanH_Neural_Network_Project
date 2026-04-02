import numpy as np

from softmax import SoftmaxLayer, LossMetrics


class NeuralNetworkModel:
    """1-hidden-layer neural network with tanh activation."""

    def __init__(self, input_dim, hidden_dim, num_classes, seed=0):
        """
        Initialize neural network.

        Args:
            input_dim: Input feature dimension (d)
            hidden_dim: Hidden layer dimension (h)
            num_classes: Number of output classes (k)
            seed: Random seed
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        rng = np.random.default_rng(seed)

        # Xavier initialization
        std1 = np.sqrt(1.0 / (input_dim + hidden_dim))
        self.W1 = std1 * rng.standard_normal((hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim)

        std2 = np.sqrt(1.0 / (hidden_dim + num_classes))
        self.W2 = std2 * rng.standard_normal((num_classes, hidden_dim))
        self.b2 = np.zeros(num_classes)

    def _hidden_activation(self, X):
        """Compute hidden activation with tanh."""
        return np.tanh(X @ self.W1.T + self.b1)

    def forward(self, X):
        """
        Forward pass for 1-hidden-layer NN.

        Args:
            X: Input of shape (n, d)

        Returns:
            H: Hidden activations (n, h)
            P: Probabilities (n, k)
        """
        H = self._hidden_activation(X)
        S = H @ self.W2.T + self.b2
        P = SoftmaxLayer.softmax(S)
        return H, P

    def loss(self, P, Y_onehot, lam):
        """Compute total loss: cross-entropy + L2 regularization."""
        data_loss = LossMetrics.cross_entropy_loss(P, Y_onehot)
        reg_loss = (lam / 2.0) * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return data_loss + reg_loss

    def backward(self, X, H, P, Y_onehot, lam):
        """
        Compute gradients.

        Args:
            X: Input (n, d)
            H: Hidden activations (n, h)
            P: Output probabilities (n, k)
            Y_onehot: One-hot labels (n, k)
            lam: Regularization coefficient

        Returns:
            dW1, db1, dW2, db2: Gradients
        """
        n = X.shape[0]

        # Output layer gradient
        dS = (P - Y_onehot) / n
        dW2 = dS.T @ H
        db2 = dS.T @ np.ones(n)

        # Backprop to hidden layer
        dH = dS @ self.W2
        dZ1 = dH * (1 - H ** 2)  # tanh derivative

        dW1 = dZ1.T @ X
        db1 = dZ1.T @ np.ones(n)

        # Add L2 gradients
        dW2 += lam * self.W2
        dW1 += lam * self.W1

        return dW1, db1, dW2, db2

    def update_weights(self, dW1, db1, dW2, db2, lr):
        """Update weights and biases."""
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def get_weights(self):
        """Return current weights as dictionary."""
        return {
            "W1": self.W1.copy(),
            "b1": self.b1.copy(),
            "W2": self.W2.copy(),
            "b2": self.b2.copy(),
        }

    def set_weights(self, weights_dict):
        """Set weights from dictionary."""
        self.W1 = weights_dict["W1"].copy()
        self.b1 = weights_dict["b1"].copy()
        self.W2 = weights_dict["W2"].copy()
        self.b2 = weights_dict["b2"].copy()


# Backward compatibility functions (deprecated)
def hidden_activation(X, W, b):
    return np.tanh(X @ W.T + b)

def nn_forward(X, W1, b1, W2, b2):
    H = hidden_activation(X, W1, b1)
    P = SoftmaxLayer.softmax(H @ W2.T + b2)
    return H, P

def nn_loss(P, Y_onehot, W1, W2, lam):
    data_loss = LossMetrics.cross_entropy_loss(P, Y_onehot)
    reg_loss = (lam / 2.0) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    return data_loss + reg_loss

def nn_gradients(X, H, P, Y_onehot, W1, W2, lam):
    n = X.shape[0]
    dS = (P - Y_onehot) / n
    dW2 = dS.T @ H
    db2 = dS.T @ np.ones(n)
    dH = dS @ W2
    dZ1 = dH * (1 - H**2)
    dW1 = dZ1.T @ X
    db1 = dZ1.T @ np.ones(n)
    dW2 += lam * W2
    dW1 += lam * W1
    return dW1, db1, dW2, db2

def initialize_nn(d, h, k, seed=0):
    rng = np.random.default_rng(seed)
    std1 = np.sqrt(1.0 / (d + h))
    W1 = std1 * rng.standard_normal((h, d))
    b1 = np.zeros(h)
    std2 = np.sqrt(1.0 / (h + k))
    W2 = std2 * rng.standard_normal((k, h))
    b2 = np.zeros(k)
    return W1, b1, W2, b2

def train_nn(X, Y_onehot, X_val, Y_val,
             d, h, k,
             epochs=100,
             lr=0.05,
             batch_size=64,
             lam=1e-4):
    W1, b1, W2, b2 = initialize_nn(d, h, k)
    n = X.shape[0]
    for epoch in range(epochs):
        idx = np.random.permutation(n)
        X_shuff = X[idx]
        Y_shuff = Y_onehot[idx]
        for i in range(0, n, batch_size):
            X_batch = X_shuff[i:i+batch_size]
            Y_batch = Y_shuff[i:i+batch_size]
            H, P = nn_forward(X_batch, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = nn_gradients(
                X_batch, H, P, Y_batch, W1, W2, lam
            )
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
        H_train, P_train = nn_forward(X, W1, b1, W2, b2)
        loss = nn_loss(P_train, Y_onehot, W1, W2, lam)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    return W1, b1, W2, b2

def nn_predict(X, W1, b1, W2, b2):
    H, P = nn_forward(X, W1, b1, W2, b2)
    return np.argmax(P, axis=1)



