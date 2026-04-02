import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from training_utils import DataUtils
from train_nn_runner import NNTrainer
from experiment_logger import ExperimentLogger
from plot_decision_boundaries import plot_decision_boundary, predict_class

# -------------------------
# Paths
# -------------------------
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "data"))


# -------------------------
# Data loading
# -------------------------
def load_moons_dataset():
    path = os.path.join(DATA_DIR, "moons.npz")
    data = np.load(path)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    return X_train, y_train, X_val, y_val, X_test, y_test


# -------------------------
# Comparison Plot
# -------------------------
def plot_ablation_comparison(X_train, y_train, results, step_size=0.02, padding=0.5):
    """
    Create a 1x3 comparison plot of decision boundaries for h=2, 8, 32.
    
    Args:
        X_train: Training features (n, 2)
        y_train: Training labels (n,)
        results: List of result dicts with hidden_width and params
        step_size: Step size for meshgrid
        padding: Extra space around data
    
    Returns:
        fig: Matplotlib figure with 3 subplots
    """
    # Create meshgrid once (same for all)
    x_min, x_max = X_train[:, 0].min() - padding, X_train[:, 0].max() + padding
    y_min, y_max = X_train[:, 1].min() - padding, X_train[:, 1].max() + padding
    
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step_size),
        np.arange(y_min, y_max, step_size)
    )
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, r in enumerate(results):
        h = r["hidden_width"]
        W1, b1, W2, b2 = r["params"]
        
        # Get predictions for meshgrid
        X_mesh = np.c_[xx.ravel(), yy.ravel()]
        Z = predict_class(X_mesh, (W1, b1, W2, b2), None, model_type='nn')
        Z = Z.reshape(xx.shape)
        
        # Plot on subplot
        ax = axes[idx]
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=40, 
                  edgecolors='k', cmap=plt.cm.Spectral)
        
        ax.set_xlabel('Feature 1', fontsize=10)
        ax.set_ylabel('Feature 2', fontsize=10)
        ax.set_title(f'Moons (h={h})\nAcc: {r["test_acc"]:.3f}', fontsize=11, fontweight='bold')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
    
    plt.tight_layout()
    return fig


# -------------------------
# Main
# -------------------------
def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_moons_dataset()

    k = len(np.unique(y_train))
    d = X_train.shape[1]

    Y_train = DataUtils.labels_to_onehot(y_train, k)
    Y_val = DataUtils.labels_to_onehot(y_val, k)
    Y_test = DataUtils.labels_to_onehot(y_test, k)

    widths = [2, 8, 32]
    results = []

    for h in widths:
        print(f"\n===== Moons Width Ablation: h = {h} =====")

        trainer = NNTrainer(d, h, k, seed=0)
        W1, b1, W2, b2, history, _ = trainer.train(
            X_train, Y_train,
            X_val, Y_val,
            epochs=500,
            lr=0.1,
            batch_size=64,
            lam=1e-4,
            checkpoint_on_val=False,
        )

        train_loss, train_acc = trainer.evaluate(X_train, Y_train, 1e-4)
        val_loss, val_acc = trainer.evaluate(X_val, Y_val, 1e-4)
        test_loss, test_acc = trainer.evaluate(X_test, Y_test, 1e-4)

        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

        results.append({
            "hidden_width": h,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "history": history,
            "params": (W1, b1, W2, b2),
        })

    # Save decision boundary plots
    figures_dir = os.path.abspath(os.path.join(CURRENT_DIR, "..", "figures"))
    os.makedirs(figures_dir, exist_ok=True)

    print(f"\n===== Generating Decision Boundary Plots =====")
    for r in results:
        h = r["hidden_width"]
        W1, b1, W2, b2 = r["params"]

        # Create plot
        model_params = {"nn": (W1, b1, W2, b2)}
        fig = plot_decision_boundary(
            X_train, y_train,
            model_params,
            f"Moons (h={h})",
            "Neural Net",
            step_size=0.02,
            padding=0.5
        )

        # Save plot
        filename = f"decision_boundary_moons_ablation_h_{h}.png"
        filepath = os.path.join(figures_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close(fig)
    
    # Create and save comparison plot
    print(f"\n===== Generating Comparison Plot =====")
    fig_comparison = plot_ablation_comparison(X_train, y_train, results, step_size=0.02, padding=0.5)
    comparison_filename = "moons_ablation_comparison.png"
    comparison_filepath = os.path.join(figures_dir, comparison_filename)
    fig_comparison.savefig(comparison_filepath, dpi=300, bbox_inches='tight')
    print(f"Saved: {comparison_filepath}")
    plt.close(fig_comparison)

    print("\n===== Summary =====")
    for r in results:
        print(
            f"h={r['hidden_width']}: "
            f"Test Loss={r['test_loss']:.4f}, Test Acc={r['test_acc']:.4f}"
        )

    return results


if __name__ == "__main__":
    logger = ExperimentLogger(CURRENT_DIR, "run_moons_ablation")
    logger.run(main)