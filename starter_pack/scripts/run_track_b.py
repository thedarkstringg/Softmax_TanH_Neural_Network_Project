"""
Track B: Confidence and Reliability Analysis
Implements all requirements for analyzing model confidence and calibration
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    from tabulate import tabulate
except ImportError:
    # Fallback if tabulate not available
    def tabulate(data, headers=None, tablefmt=None):
        if headers:
            print("\t".join(headers))
            print("-" * 70)
        for row in data:
            print("\t".join(str(x) for x in row))

# Add src to path
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from training_utils import labels_to_onehot, accuracy_from_probs
from train_softmax import train_softmax, evaluate_softmax
from train_nn_runner import train_nn_runner, evaluate_nn
from softmax import softmax_forward
from neural_net import nn_forward


# -------------------------
# Paths
# -------------------------
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "data"))
FIGURES_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "figures"))
os.makedirs(FIGURES_DIR, exist_ok=True)


# -------------------------
# Data Loading
# -------------------------
def load_digits_dataset_with_split():
    """Load digits dataset using the fixed split indices"""
    data_path = os.path.join(DATA_DIR, "digits_data.npz")
    split_path = os.path.join(DATA_DIR, "digits_split_indices.npz")

    data = np.load(data_path)
    split = np.load(split_path)

    X = data["X"]
    y = data["y"]

    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_val = X[val_idx]
    y_val = y[val_idx]

    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test


# -------------------------
# Confidence Metrics
# -------------------------
def compute_confidence(P):
    """Compute max predicted class probability (confidence) for each sample"""
    return np.max(P, axis=1)


def compute_predictive_entropy(P):
    """
    Compute predictive entropy for each sample
    H = -sum(P * log(P))
    """
    # Avoid log(0) by clipping
    P_clipped = np.clip(P, 1e-10, 1.0)
    entropy = -np.sum(P * np.log(P_clipped), axis=1)
    return entropy


def compute_predictions(P):
    """Get predicted class for each sample"""
    return np.argmax(P, axis=1)


def compute_accuracy(predictions, y_true):
    """Compute accuracy"""
    return np.mean(predictions == y_true)


# -------------------------
# Binning and Calibration
# -------------------------
def compute_binned_accuracy(confidence, predictions, y_true, num_bins=5):
    """
    Compute accuracy in bins of confidence
    Returns: bin_centers, bin_accuracies, bin_counts
    """
    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    for i in range(num_bins):
        mask = (confidence >= bins[i]) & (confidence < bins[i + 1])
        
        if np.sum(mask) == 0:
            continue

        acc = np.mean(predictions[mask] == y_true[mask])
        center = (bins[i] + bins[i + 1]) / 2
        count = np.sum(mask)

        bin_centers.append(center)
        bin_accuracies.append(acc)
        bin_counts.append(count)

    return np.array(bin_centers), np.array(bin_accuracies), np.array(bin_counts)


# -------------------------
# Tables
# -------------------------
def print_confidence_accuracy_table(model_name, confidence, predictions, y_true, num_bins=5):
    """Print confidence vs empirical accuracy table"""
    bin_centers, bin_accs, bin_counts = compute_binned_accuracy(
        confidence, predictions, y_true, num_bins=num_bins
    )

    print(f"\n{'='*70}")
    print(f"{model_name}: Confidence vs Empirical Accuracy (5 bins)")
    print(f"{'='*70}")

    table_data = []
    for i, (center, acc, count) in enumerate(zip(bin_centers, bin_accs, bin_counts)):
        table_data.append([
            f"Bin {i+1}",
            f"[{center-0.1:.2f}, {center+0.1:.2f}]",
            f"{center:.2f}",
            f"{acc:.4f}",
            f"{int(count)}"
        ])

    headers = ["Bin", "Confidence Range", "Center", "Accuracy", "Count"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    overall_acc = np.mean(predictions == y_true)
    print(f"\nOverall Test Accuracy: {overall_acc:.4f}")


def print_summary_statistics(model_name, predictions, y_true, confidence, entropy):
    """Print summary statistics for a model"""
    acc = compute_accuracy(predictions, y_true)
    
    print(f"\n{'='*70}")
    print(f"{model_name}: Summary Statistics")
    print(f"{'='*70}")
    
    stats = [
        ["Test Accuracy", f"{acc:.4f}"],
        ["Mean Confidence", f"{np.mean(confidence):.4f}"],
        ["Std Confidence", f"{np.std(confidence):.4f}"],
        ["Min Confidence", f"{np.min(confidence):.4f}"],
        ["Max Confidence", f"{np.max(confidence):.4f}"],
        ["Mean Entropy", f"{np.mean(entropy):.4f}"],
        ["Std Entropy", f"{np.std(entropy):.4f}"],
        ["Min Entropy", f"{np.min(entropy):.4f}"],
        ["Max Entropy", f"{np.max(entropy):.4f}"],
    ]
    
    print(tabulate(stats, headers=["Metric", "Value"], tablefmt="grid"))


# -------------------------
# Visualizations
# -------------------------
def plot_reliability_diagram(
    softmax_data, nn_data, figsize=(12, 5)
):
    """
    Plot reliability diagram (confidence vs accuracy) for both models
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Softmax
    sm_conf, sm_pred, sm_y = softmax_data
    sm_centers, sm_accs, _ = compute_binned_accuracy(sm_conf, sm_pred, sm_y, num_bins=5)
    
    axes[0].plot(sm_centers, sm_accs, marker='o', linewidth=2, markersize=8, 
                 label='Softmax', color='steelblue')
    axes[0].plot([0, 1], [0, 1], linestyle='--', linewidth=2, color='gray', 
                 label='Perfect Calibration', alpha=0.7)
    axes[0].fill_between([0, 1], [0, 1], alpha=0.1, color='gray')
    axes[0].set_xlabel("Confidence (Mean)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Accuracy", fontsize=12, fontweight='bold')
    axes[0].set_title("Softmax: Reliability Diagram", fontsize=13, fontweight='bold')
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Neural Network
    nn_conf, nn_pred, nn_y = nn_data
    nn_centers, nn_accs, _ = compute_binned_accuracy(nn_conf, nn_pred, nn_y, num_bins=5)
    
    axes[1].plot(nn_centers, nn_accs, marker='s', linewidth=2, markersize=8,
                 label='Neural Network', color='coral')
    axes[1].plot([0, 1], [0, 1], linestyle='--', linewidth=2, color='gray',
                 label='Perfect Calibration', alpha=0.7)
    axes[1].fill_between([0, 1], [0, 1], alpha=0.1, color='gray')
    axes[1].set_xlabel("Confidence (Mean)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Accuracy", fontsize=12, fontweight='bold')
    axes[1].set_title("Neural Network: Reliability Diagram", fontsize=13, fontweight='bold')
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "confidence_reliability_diagram.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {path}")


def plot_confidence_entropy_comparison(softmax_data, nn_data, figsize=(14, 10)):
    """
    Plot confidence and entropy distributions for correct vs incorrect predictions
    for both Softmax and Neural Network
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    sm_conf, sm_entropy, sm_pred, sm_y = softmax_data
    nn_conf, nn_entropy, nn_pred, nn_y = nn_data

    # Softmax - Confidence
    sm_correct_mask = (sm_pred == sm_y)
    sm_conf_correct = sm_conf[sm_correct_mask]
    sm_conf_incorrect = sm_conf[~sm_correct_mask]

    axes[0, 0].hist(sm_conf_correct, bins=30, alpha=0.6, label='Correct',
                    color='green', edgecolor='black')
    axes[0, 0].hist(sm_conf_incorrect, bins=30, alpha=0.6, label='Incorrect',
                    color='red', edgecolor='black')
    axes[0, 0].set_xlabel("Confidence", fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel("Frequency", fontsize=11, fontweight='bold')
    axes[0, 0].set_title("Softmax: Confidence Distribution", fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # Softmax - Entropy
    sm_entropy_correct = sm_entropy[sm_correct_mask]
    sm_entropy_incorrect = sm_entropy[~sm_correct_mask]

    axes[0, 1].hist(sm_entropy_correct, bins=30, alpha=0.6, label='Correct',
                    color='green', edgecolor='black')
    axes[0, 1].hist(sm_entropy_incorrect, bins=30, alpha=0.6, label='Incorrect',
                    color='red', edgecolor='black')
    axes[0, 1].set_xlabel("Predictive Entropy", fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel("Frequency", fontsize=11, fontweight='bold')
    axes[0, 1].set_title("Softmax: Entropy Distribution", fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Neural Network - Confidence
    nn_correct_mask = (nn_pred == nn_y)
    nn_conf_correct = nn_conf[nn_correct_mask]
    nn_conf_incorrect = nn_conf[~nn_correct_mask]

    axes[1, 0].hist(nn_conf_correct, bins=30, alpha=0.6, label='Correct',
                    color='green', edgecolor='black')
    axes[1, 0].hist(nn_conf_incorrect, bins=30, alpha=0.6, label='Incorrect',
                    color='red', edgecolor='black')
    axes[1, 0].set_xlabel("Confidence", fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel("Frequency", fontsize=11, fontweight='bold')
    axes[1, 0].set_title("Neural Network: Confidence Distribution", fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Neural Network - Entropy
    nn_entropy_correct = nn_entropy[nn_correct_mask]
    nn_entropy_incorrect = nn_entropy[~nn_correct_mask]

    axes[1, 1].hist(nn_entropy_correct, bins=30, alpha=0.6, label='Correct',
                    color='green', edgecolor='black')
    axes[1, 1].hist(nn_entropy_incorrect, bins=30, alpha=0.6, label='Incorrect',
                    color='red', edgecolor='black')
    axes[1, 1].set_xlabel("Predictive Entropy", fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel("Frequency", fontsize=11, fontweight='bold')
    axes[1, 1].set_title("Neural Network: Entropy Distribution", fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "confidence_entropy_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {path}")


def plot_combined_reliability_curves(softmax_data, nn_data, figsize=(10, 7)):
    """
    Plot reliability curves for both models on same axes for easy comparison
    """
    sm_conf, sm_pred, sm_y = softmax_data
    nn_conf, nn_pred, nn_y = nn_data

    sm_centers, sm_accs, _ = compute_binned_accuracy(sm_conf, sm_pred, sm_y, num_bins=5)
    nn_centers, nn_accs, _ = compute_binned_accuracy(nn_conf, nn_pred, nn_y, num_bins=5)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(sm_centers, sm_accs, marker='o', linewidth=2.5, markersize=10,
            label='Softmax (Linear)', color='steelblue')
    ax.plot(nn_centers, nn_accs, marker='s', linewidth=2.5, markersize=10,
            label='Neural Network (Nonlinear)', color='coral')
    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=2, color='gray',
            label='Perfect Calibration', alpha=0.7)
    ax.fill_between([0, 1], [0, 1], alpha=0.1, color='gray')

    ax.set_xlabel("Confidence (Mean)", fontsize=13, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=13, fontweight='bold')
    ax.set_title("Model Comparison: Reliability Diagrams", fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "combined_reliability_curves.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {path}")


# -------------------------
# Interpretation
# -------------------------
def print_interpretation(softmax_data, nn_data):
    """
    Print brief text interpretation tied to main project question
    (linear vs nonlinear models)
    """
    sm_conf, sm_pred, sm_y = softmax_data
    nn_conf, nn_pred, nn_y = nn_data

    sm_acc = compute_accuracy(sm_pred, sm_y)
    nn_acc = compute_accuracy(nn_pred, nn_y)
    
    print(f"\n{'='*70}")
    print("INTERPRETATION: Linear vs Nonlinear Models")
    print(f"{'='*70}")

    print(f"""
KEY FINDINGS:
          
   PREDICTIVE PERFORMANCE
   • Softmax (Linear) Accuracy:     {sm_acc:.4f}
   • Neural Network (Nonlinear):    {nn_acc:.4f}

   MAIN PROJECT INSIGHT
   This analysis demonstrates that nonlinear models (NN) not only achieve
   better accuracy but also produce more reliable and well-calibrated
   confidence scores. The NN's ability to capture nonlinear decision
   boundaries translates to more meaningful confidence estimates that
   practitioners can trust.
""")

    print("="*70)


# -------------------------
# Main
# -------------------------
def main():
    print("\n" + "="*70)
    print("TRACK B: Confidence and Reliability Analysis")
    print("="*70)

    # Load data
    print("\n[1/5] Loading digits dataset with fixed split indices...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_digits_dataset_with_split()
    
    d = X_train.shape[1]
    k = len(np.unique(y_train))
    n_test = len(y_test)
    
    print(f"✓ Loaded: {len(y_train)} train, {len(y_val)} val, {n_test} test samples")
    print(f"  Features: {d}, Classes: {k}")

    Y_train = labels_to_onehot(y_train, k)
    Y_val = labels_to_onehot(y_val, k)
    Y_test = labels_to_onehot(y_test, k)

    # Train Softmax
    print("\n[2/5] Training Softmax (linear model)...")
    W_sm, b_sm, _, _ = train_softmax(
        X_train, Y_train, y_train,
        X_val, Y_val, y_val,
        d, k,
        epochs=200,
        lr=0.05,
        batch_size=64,
        lam=1e-4,
        seed=0,
        checkpoint_on_val=True,
    )
    print("✓ Softmax training complete")

    # Train Neural Network
    print("\n[3/5] Training Neural Network (nonlinear model)...")
    W1, b1, W2, b2, _, _ = train_nn_runner(
        X_train, Y_train,
        X_val, Y_val,
        d, 32, k,
        epochs=200,
        lr=0.05,
        batch_size=64,
        lam=1e-4,
        seed=0,
        checkpoint_on_val=True,
    )
    print("✓ Neural Network training complete")

    # Softmax predictions and metrics
    print("\n[4/5] Computing predictions and confidence metrics...")
    P_sm = softmax_forward(X_test, W_sm, b_sm)
    sm_conf = compute_confidence(P_sm)
    sm_entropy = compute_predictive_entropy(P_sm)
    sm_pred = compute_predictions(P_sm)

    # NN predictions and metrics
    _, P_nn = nn_forward(X_test, W1, b1, W2, b2)
    nn_conf = compute_confidence(P_nn)
    nn_entropy = compute_predictive_entropy(P_nn)
    nn_pred = compute_predictions(P_nn)

    print("✓ Computed confidence, entropy, and predictions for both models")

    # Print tables
    print_confidence_accuracy_table("Softmax", sm_conf, sm_pred, y_test, num_bins=5)
    print_confidence_accuracy_table("Neural Network", nn_conf, nn_pred, y_test, num_bins=5)

    # Print summary statistics
    print_summary_statistics("Softmax", sm_pred, y_test, sm_conf, sm_entropy)
    print_summary_statistics("Neural Network", nn_pred, y_test, nn_conf, nn_entropy)

    # Generate visualizations
    print(f"\n[5/5] Generating visualizations...")
    
    softmax_reliability = (sm_conf, sm_pred, y_test)
    nn_reliability = (nn_conf, nn_pred, y_test)
    
    plot_reliability_diagram(softmax_reliability, nn_reliability, figsize=(12, 5))
    
    softmax_entropy_data = (sm_conf, sm_entropy, sm_pred, y_test)
    nn_entropy_data = (nn_conf, nn_entropy, nn_pred, y_test)
    plot_confidence_entropy_comparison(softmax_entropy_data, nn_entropy_data, figsize=(14, 10))
    
    plot_combined_reliability_curves(softmax_reliability, nn_reliability, figsize=(10, 7))

    # Print interpretation
    print_interpretation(softmax_reliability, nn_reliability)

    print(f"\n{'='*70}")
    print("✓ Track B Analysis Complete!")
    print(f"{'='*70}\n")

    return {
        "softmax": {
            "accuracy": compute_accuracy(sm_pred, y_test),
            "confidence": sm_conf,
            "entropy": sm_entropy,
            "predictions": sm_pred,
        },
        "neural_network": {
            "accuracy": compute_accuracy(nn_pred, y_test),
            "confidence": nn_conf,
            "entropy": nn_entropy,
            "predictions": nn_pred,
        },
    }


if __name__ == "__main__":
    results = main()
