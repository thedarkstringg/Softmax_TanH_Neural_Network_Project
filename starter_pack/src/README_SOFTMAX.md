# Softmax Regression Module


## What is this?

`softmax.py` is a complete implementation of **Softmax Regression** — a machine learning algorithm that learns to classify data into multiple categories. Think of it like teaching a model to recognize handwritten digits (0-9) or different types of objects.

It's built entirely with NumPy, so we can see exactly how everything works under the hood.

---

## Quick Overview of Functions

### **Core Functions We'll Actually Use**

#### `softmax_forward(X, W, b)` — Make Predictions
```python
P = softmax_forward(X, W, b)
```
- **What it does**: Takes raw data and produces probability predictions
- **Inputs**:
  - `X`: Our data (n examples × d features)
  - `W`: Learned weights (k classes × d features)
  - `b`: Learned biases (k classes)
- **Output**: Probabilities that sum to 1 for each example
- **Example**: Input a 28×28 pixel image → Get probabilities [0.02, 0.95, 0.01, ...] for each digit

#### `softmax_gradients(X, P, Y_onehot)` — Learn from Mistakes
```python
dW, db = softmax_gradients(X, P, Y_onehot)
```
- **What it does**: Calculates how much to adjust weights to improve predictions
- **Inputs**: Our data, predictions, and true labels
- **Output**: Gradients (direction and magnitude to update W and b)
- **Why it matters**: This is how the model learns

#### `cross_entropy_loss(P, Y_onehot)` — Measure Errors
```python
loss = cross_entropy_loss(P, Y_onehot)
```
- **What it does**: Measures how wrong the predictions are
- **Output**: A single number (lower is better)
- **Use case**: Track training progress — loss should decrease as the model learns

#### `accuracy(P, Y)` — Measure Success
```python
acc = accuracy(P, Y)
```
- **What it does**: Checks what percentage of predictions are correct
- **Output**: A number from 0 to 1 (0.85 = 85% correct)
- **Use case**: Evaluate model performance on test data

---

### **Helper Functions**

#### `softmax(S)` — Probability Converter
Turns raw scores into valid probabilities (numbers between 0 and 1 that sum to 1).
- **Technical detail**: Uses max-subtraction trick to prevent numerical overflow

#### `l2_regularization_loss(W, lam)` & `l2_regularization_grad(W, lam)`
- **What they do**: Prevent overfitting by penalizing large weights
- **When to use**: When our model memorizes training data instead of learning patterns
- **Parameter `lam`**: Control strength of penalty (typical: 0.0001 to 0.01)

---

## Input/Output Shapes (Matrix Conventions)

```
X: (n, d)       — n examples, d features each
W: (k, d)       — k classes, d features each
b: (k,)         — k bias values
S: (n, k)       — Raw scores (logits)
P: (n, k)       — Probabilities (what softmax outputs)
Y_onehot: (n, k) — Labels in one-hot format
```

**What's one-hot?**
- Class 2 out of 5 classes → `[0, 0, 1, 0, 0]`
- Class 0 out of 5 classes → `[1, 0, 0, 0, 0]`

---

## Implementation Details

 **Numerical Stability**: The `softmax()` function uses a max-subtraction trick to prevent overflow when computing exponentials

 **Correct Gradients**: Gradients are derived from calculus (backpropagation) and mathematically verified

 **Efficient**: Matrix operations use NumPy for speed

---

## Sanity Checks (Verify Everything Works)

We included `sanity_checks.py` to test that our softmax implementation is correct. It runs 5 critical checks:

1. **Probabilities sum to 1** — Verifies softmax outputs are valid
2. **No NaN/Inf on extreme inputs** — Tests numerical stability with very large/small numbers
3. **Numerical gradient check** — Compares analytical gradients vs finite differences (catches math errors)
4. **Loss decreases** — Confirms the model actually learns
5. **Overfit tiny subset** — Verifies model can memorize small datasets

**How to run:**
```bash
python ../scripts/sanity_checks.py
```

All 5 checks should pass if the implementation is correct. If any fail, there's a bug to fix!

---

## When to Use Softmax Regression

Good for:
- Multi-class classification (3+ categories)
- Interpretable models (we can see which features matter)
- Fast training on smaller datasets
- Educational purposes (learning ML fundamentals)

Not ideal for:
- Very complex patterns (we use neural networks instead)
- Image recognition at scale (we use CNNs)
- Imbalanced datasets (needs special techniques)
