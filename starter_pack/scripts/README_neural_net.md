# One-Hidden-Layer Neural Network (NumPy)

A from-scratch implementation of a shallow neural network using NumPy. This model is designed for **multi-class classification** and introduces a non-linear hidden layer to capture complex patterns that standard softmax regression cannot.

## 🚀 Overview
The model follows a standard feed-forward architecture:
1. **Hidden Layer:** Uses the **tanh** activation function.
2. **Output Layer:** Uses the **softmax** function to produce a probability distribution.

---

## 🛠 Core API Reference

### 1. Initialization & Prediction
| Function | Description |
| :--- | :--- |
| **initialize_nn(d, h, k)** | Initializes weights with small random values and biases to zero. |
| **nn_forward(X, W1, b1, W2, b2)** | Performs the forward pass to get hidden activations (H) and probabilities (P). |
| **nn_predict(X, W1, b1, W2, b2)** | Converts class probabilities into final predicted labels. |

### 2. Training & Learning
| Function | Description |
| :--- | :--- |
| **nn_gradients(...)** | **The Learning Mechanism:** Computes weight adjustments via backpropagation. |
| **nn_loss(P, Y, W1, W2, lam)** | Measures error using Cross-Entropy and adds L2 regularization. |
| **train_nn(...)** | The full training loop: Shuffles data, predicts, computes gradients, and updates weights. |

---

## 📐 Implementation Details

### Activation & Loss
* **Hidden Layer:** Uses `tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
* **Total Loss:** `Loss = Cross-Entropy + lambda * (L2_norm_Weights)`
    * *Cross-entropy* measures prediction error.
    * *L2 regularization* prevents the model from overfitting by penalizing large weights.

### Backpropagation Logic
Gradients are calculated in two distinct stages:

**Stage 1: Output Layer**
* Error (dS) = (Predictions - Labels) / n
* dW2 = dS_transpose * Hidden_Activations
* db2 = sum(dS)

**Stage 2: Hidden Layer**
* dH = dS * W2
* dZ1 = dH * (1 - Hidden_Activations^2)
* dW1 = dZ1_transpose * X
* db1 = sum(dZ1)

---

## 🎯 When to Use This Model

### ✅ Recommended for:
* **Non-linear Data:** Use when patterns cannot be separated by a straight line (e.g., XOR patterns).
* **Moderate Complexity:** More powerful than linear regression, but simpler and faster than deep networks.
* **Small to Medium Datasets:** Helps avoid the overfitting issues often found in deeper models.

### ❌ Avoid if:
* **Linear Separability:** If a simple line can separate your data, use Softmax Regression instead.
* **Complex Data:** For high-res images or complex audio, consider Deep Neural Networks (CNNs/RNNs).
* **Interpretability:** If you need to explain exactly "why" a feature triggered a result, linear models are easier to interpret.

---

## 💻 Quick Start
```python
# 1. Initialize the weights
W1, b1, W2, b2 = initialize_nn(input_dim=784, hidden_dim=64, num_classes=10)

# 2. Train the model
W1, b1, W2, b2 = train_nn(X_train, Y_onehot, X_val, Y_val, d, h, k)

# 3. Make predictions
predictions = nn_predict(X_test, W1, b1, W2, b2)
