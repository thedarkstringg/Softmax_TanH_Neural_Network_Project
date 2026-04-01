One-Hidden-Layer Neural Network (NumPy)
What is this?

This module is a from-scratch implementation of a one-hidden-layer neural network using NumPy. It is designed for multi-class classification problems and extends softmax regression by introducing a non-linear hidden layer.

The model learns:

A hidden representation using tanh activation
A probability distribution over classes using softmax

This allows it to capture non-linear relationships in data that linear models cannot.
Quick Overview of Functions
Core Functions We’ll Actually Use
nn_forward(X, W1, b1, W2, b2) — Make Predictions
H, P = nn_forward(X, W1, b1, W2, b2)

What it does:
Takes input data and produces class probability predictions using a neural network.

Inputs:

X: Input data (n examples × d features)
W1, b1: Hidden layer weights and biases (h × d), (h,)
W2, b2: Output layer weights and biases (k × h), (k,)

Output:

H: Hidden layer activations
P: Probabilities for each class (each row sums to 1)

Example:
Input an image → hidden layer learns features → output gives probabilities like:
[0.01, 0.03, 0.92, ...]

nn_gradients(X, H, P, Y_onehot, W1, W2, lam) — Learn from Mistakes
dW1, db1, dW2, db2 = nn_gradients(X, H, P, Y_onehot, W1, W2, lam)

What it does:
Computes how to adjust all weights and biases to improve predictions (backpropagation).

Inputs:

X: Input data
H: Hidden activations
P: Predictions
Y_onehot: True labels
W1, W2: Current weights
lam: Regularization strength

Output:

Gradients for all parameters (dW1, db1, dW2, db2)

Why it matters:
This is the learning mechanism — it tells the model how to update itself.

nn_loss(P, Y_onehot, W1, W2, lam) — Measure Errors
loss = nn_loss(P, Y_onehot, W1, W2, lam)

What it does:
Measures how wrong the predictions are.

Output:

A single number (lower is better)

Includes:

Cross-entropy loss (prediction error)
L2 regularization (prevents overfitting)

Use case:
Track training progress — loss should decrease over time.

train_nn(...) — Train the Model
W1, b1, W2, b2 = train_nn(X, Y_onehot, X_val, Y_val, d, h, k)

What it does:
Runs the full training loop:

Shuffle data
Make predictions (forward pass)
Compute gradients (backward pass)
Update parameters

Output:

Trained weights and biases

Why it matters:
This is where the model actually learns from data.

nn_predict(X, W1, b1, W2, b2) — Final Predictions
y_pred = nn_predict(X, W1, b1, W2, b2)

What it does:
Converts probabilities into final predicted labels.

Output:

Predicted class index for each example

Example:
[2, 0, 1, 9, ...]

initialize_nn(d, h, k) — Start the Model
W1, b1, W2, b2 = initialize_nn(d, h, k)

What it does:
Creates initial weights and biases before training starts.

Why it matters:
Good initialization helps stable learning.
Implementation Details
1. Hidden Layer
Uses tanh activation

Formula:

tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})
2. Loss Function

Total loss:

Loss = CrossEntropy + λ (||W1||² + ||W2||²)
Cross-entropy measures prediction error
L2 regularization prevents overfitting
3. Backpropagation

Gradients are computed in two steps:

Output layer:

dS = (P - Y) / n
dW2 = dS^T H
db2 = sum(dS)

Hidden layer:

dH  = dS W2
dZ1 = dH * (1 - H²)
dW1 = dZ1^T X
db1 = sum(dZ1)
4. Optimization
Uses mini-batch gradient descent
Parameters updated as:
W := W - lr * dW
b := b - lr * db
5. Initialization

Weights initialized with small random values:

W ~ N(0, 0.01)
Biases initialized to zero
When to Use Neural Network with One Hidden Layer

Use this model when:

✅ Data is non-linearly separable
Example: XOR-type patterns
Softmax regression will fail here
✅ You need moderate model complexity
More powerful than linear models
Simpler and faster than deep networks
✅ Dataset is small to medium
Avoids overfitting compared to deeper models
❌ Avoid when:
Data is linearly separable → use softmax regression instead
Dataset is very large/complex → consider deep neural networks
Interpretability is critical → linear models are easier