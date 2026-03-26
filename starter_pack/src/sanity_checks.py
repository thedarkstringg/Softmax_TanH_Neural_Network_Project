"""
Sanity checks for softmax regression implementation.
"""

import numpy as np
from softmax import (
    softmax, cross_entropy_loss, accuracy, softmax_forward,
    softmax_gradients, l2_regularization_loss, l2_regularization_grad
)


def check_probabilities_sum_to_one():
    """
    Check 1: Verify softmax outputs sum to 1 along each row.
    """
    np.random.seed(42)
    S = np.random.randn(5, 4)
    P = softmax(S)

    sums = np.sum(P, axis=1)
    passed = np.allclose(sums, 1.0)

    if passed:
        print("PASS: probabilities sum to 1")
    else:
        print(f"FAIL: probabilities do not sum to 1. Max deviation: {np.max(np.abs(sums - 1.0))}")

    return passed


def check_no_nan_inf_extreme_inputs():
    """
    Check 2: Verify softmax handles extreme inputs without NaN or Inf.
    """
    # Test with very large positive values
    S_large_pos = np.full((3, 4), 1000.0)
    P_large_pos = softmax(S_large_pos)

    # Test with very large negative values
    S_large_neg = np.full((3, 4), -1000.0)
    P_large_neg = softmax(S_large_neg)

    # Mix of extreme values
    S_mixed = np.array([[1000, -1000, 500, -500],
                        [-1000, 1000, -500, 500],
                        [1000, 1000, -1000, -1000]])
    P_mixed = softmax(S_mixed)

    has_nan = np.any(np.isnan(P_large_pos)) or np.any(np.isnan(P_large_neg)) or np.any(np.isnan(P_mixed))
    has_inf = np.any(np.isinf(P_large_pos)) or np.any(np.isinf(P_large_neg)) or np.any(np.isinf(P_mixed))

    passed = not has_nan and not has_inf

    if passed:
        print("PASS: no NaN or Inf on extreme inputs")
    else:
        print("FAIL: NaN or Inf detected on extreme inputs")

    return passed


def check_numerical_gradient():
    """
    Check 3: Verify analytical gradients match numerical gradients.
    """
    np.random.seed(42)
    n, d, k = 4, 3, 2
    X = np.random.randn(n, d)
    W = np.random.randn(k, d)
    b = np.random.randn(k)
    Y = np.array([0, 1, 0, 1])

    # Convert Y to one-hot
    Y_onehot = np.zeros((n, k))
    Y_onehot[np.arange(n), Y] = 1

    # Forward pass
    P = softmax_forward(X, W, b)

    # Analytical gradients
    dW_analytical, db_analytical = softmax_gradients(X, P, Y_onehot)

    # Numerical gradient check for a few entries of W
    epsilon = 1e-5
    rel_errors = []

    for i in range(min(2, k)):
        for j in range(min(2, d)):
            # Loss at W + epsilon
            W_plus = W.copy()
            W_plus[i, j] += epsilon
            P_plus = softmax_forward(X, W_plus, b)
            loss_plus = cross_entropy_loss(P_plus, Y_onehot)

            # Loss at W - epsilon
            W_minus = W.copy()
            W_minus[i, j] -= epsilon
            P_minus = softmax_forward(X, W_minus, b)
            loss_minus = cross_entropy_loss(P_minus, Y_onehot)

            # Numerical gradient
            dW_numerical = (loss_plus - loss_minus) / (2 * epsilon)

            # Analytical gradient
            dW_analytical_entry = dW_analytical[i, j]

            # Relative error
            rel_error = np.abs(dW_numerical - dW_analytical_entry) / (
                np.abs(dW_analytical_entry) + np.abs(dW_numerical) + 1e-12
            )
            rel_errors.append(rel_error)

    max_rel_error = np.max(rel_errors)
    passed = max_rel_error < 1e-4

    if passed:
        print(f"PASS: numerical gradient check relative error = {max_rel_error:.2e}")
    else:
        print(f"FAIL: numerical gradient check relative error = {max_rel_error:.2e} (threshold 1e-4)")

    return passed


def check_loss_decreases():
    """
    Check 4: Verify loss decreases during gradient descent.
    """
    np.random.seed(42)
    n, d, k = 10, 5, 3
    X = np.random.randn(n, d)
    W = np.random.randn(k, d) * 0.1
    b = np.random.randn(k) * 0.1
    Y = np.random.randint(0, k, n)

    # Convert Y to one-hot
    Y_onehot = np.zeros((n, k))
    Y_onehot[np.arange(n), Y] = 1

    # Compute initial loss
    P = softmax_forward(X, W, b)
    initial_loss = cross_entropy_loss(P, Y_onehot)

    # Run 20 gradient steps
    lr = 0.05
    for step in range(20):
        P = softmax_forward(X, W, b)
        dW, db = softmax_gradients(X, P, Y_onehot)
        W -= lr * dW
        b -= lr * db

    # Compute final loss
    P = softmax_forward(X, W, b)
    final_loss = cross_entropy_loss(P, Y_onehot)

    passed = final_loss < initial_loss

    if passed:
        print(f"PASS: loss decreased from {initial_loss:.6f} to {final_loss:.6f}")
    else:
        print(f"FAIL: loss did not decrease (initial: {initial_loss:.6f}, final: {final_loss:.6f})")

    return passed


def check_overfit_tiny_subset():
    """
    Check 5: Verify model can overfit a tiny subset of 10 examples.
    """
    np.random.seed(42)
    n, d, k = 10, 8, 3
    X = np.random.randn(n, d)
    W = np.random.randn(k, d) * 0.01
    b = np.zeros(k)
    Y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    # Convert Y to one-hot
    Y_onehot = np.zeros((n, k))
    Y_onehot[np.arange(n), Y] = 1

    # Train for 200 epochs
    lr = 0.1
    for epoch in range(200):
        P = softmax_forward(X, W, b)
        dW, db = softmax_gradients(X, P, Y_onehot)
        W -= lr * dW
        b -= lr * db

    # Compute final accuracy
    P_final = softmax_forward(X, W, b)
    final_acc = accuracy(P_final, Y)

    passed = final_acc >= 0.99

    if passed:
        print(f"PASS: overfit tiny subset, accuracy = {final_acc:.4f}")
    else:
        print(f"FAIL: could not overfit tiny subset (accuracy: {final_acc:.4f}, threshold: 0.99)")

    return passed


def run_all_checks():
    """
    Run all sanity checks and print summary.
    """
    print("=" * 70)
    print("SOFTMAX REGRESSION SANITY CHECKS")
    print("=" * 70)
    print()

    checks = [
        ("Check 1: Probabilities sum to 1", check_probabilities_sum_to_one),
        ("Check 2: No NaN/Inf on extreme inputs", check_no_nan_inf_extreme_inputs),
        ("Check 3: Numerical gradient check", check_numerical_gradient),
        ("Check 4: Loss decreases", check_loss_decreases),
        ("Check 5: Overfit tiny subset", check_overfit_tiny_subset),
    ]

    results = []
    for check_name, check_func in checks:
        print(f"{check_name}")
        try:
            passed = check_func()
            results.append(passed)
        except Exception as e:
            print(f"FAIL: Exception raised: {e}")
            results.append(False)
        print()

    print("=" * 70)
    all_passed = all(results)
    if all_passed:
        print("✓ ALL CHECKS PASSED")
    else:
        failed_count = len([r for r in results if not r])
        print(f"✗ {failed_count}/{len(results)} CHECKS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    run_all_checks()
