import numpy as np
from softmax import softmax, softmax_forward, cross_entropy_loss, l2_regularization_loss, l2_regularization_grad


def hidden_activation (X, W, b):
    return np.tan(X @ W.T + b)

# -------------------------
# Forward Pass
# -------------------------
def nn_forward(X, W1, b1, W2, b2):
    """
    Forward pass for 1-hidden-layer NN

    Returns:
        H: hidden activations (n, h)
        P: probabilities (n, k)
    """
    H = hidden_activation (X, W1, b1)          # (n, h)
    P = softmax_forward(X,W2,b2)               # (n, k)

    return H, P


# -------------------------
# Loss
# -------------------------
def nn_loss(P, Y_onehot, W1, W2, lam):
    """
    Cross-entropy + L2 regularization
    """
    data_loss = cross_entropy_loss(P, Y_onehot)
    reg_loss = l2_regularization_loss(W1, lam) + l2_regularization_loss(W2, lam)
    return data_loss + reg_loss


# -------------------------
# Backward Pass
# -------------------------
def nn_gradients(X, H, P, Y_onehot, W1, W2, lam):
    """
    Compute gradients for NN
    """
    n = X.shape[0]

    # Output layer gradient
    dS = (P - Y_onehot) / n              # (n, k)

    dW2 = dS.T @ H                      # (k, h)
    db2 = dS.T @ np.ones(n)             # (k,)

    # Backprop into hidden
    dH = dS @ W2                        # (n, h)
    dZ1 = dH * (1 - H**2)               # tanh derivative

    dW1 = dZ1.T @ X                     # (h, d)
    db1 = dZ1.T @ np.ones(n)            # (h,)

    # Add L2 gradients
    dW2 += l2_regularization_grad(W2, lam)
    dW1 += l2_regularization_grad(W1, lam)

    return dW1, db1, dW2, db2


# -------------------------
# Initialization
# -------------------------
def initialize_nn(d, h, k, seed=0):
    rng = np.random.default_rng(seed)

    W1 = 0.01 * rng.standard_normal((h, d))
    b1 = np.zeros(h)

    W2 = 0.01 * rng.standard_normal((k, h))
    b2 = np.zeros(k)

    return W1, b1, W2, b2


# -------------------------
# Training Loop
# -------------------------
def train_nn(X, Y_onehot, X_val, Y_val,
             d, h, k,
             epochs=100,
             lr=0.05,
             batch_size=64,
             lam=1e-4):

    W1, b1, W2, b2 = initialize_nn(d, h, k)

    n = X.shape[0]

    for epoch in range(epochs):

        # Shuffle
        idx = np.random.permutation(n)
        X_shuff = X[idx]
        Y_shuff = Y_onehot[idx]

        for i in range(0, n, batch_size):
            X_batch = X_shuff[i:i+batch_size]
            Y_batch = Y_shuff[i:i+batch_size]

            # Forward
            H, P = nn_forward(X_batch, W1, b1, W2, b2)

            # Backward
            dW1, db1, dW2, db2 = nn_gradients(
                X_batch, H, P, Y_batch, W1, W2, lam
            )

            # Update
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

        # Monitor loss
        H_train, P_train = nn_forward(X, W1, b1, W2, b2)
        loss = nn_loss(P_train, Y_onehot, W1, W2, lam)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return W1, b1, W2, b2


# -------------------------
# Prediction
# -------------------------
def nn_predict(X, W1, b1, W2, b2):
    H, P = nn_forward(X, W1, b1, W2, b2)
    return np.argmax(P, axis=1)



