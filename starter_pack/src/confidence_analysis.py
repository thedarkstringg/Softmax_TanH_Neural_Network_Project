import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from softmax import SoftmaxLayer
from neural_net import NeuralNetworkModel
from training_utils import DataUtils
from train_softmax import SoftmaxTrainer
from train_nn_runner import NNTrainer

from sklearn.model_selection import train_test_split


class ConfidenceAnalyzer:
    """Analyzer for model confidence and prediction calibration."""

    @staticmethod
    def compute_confidence_and_predictions(P):
        """
        Compute confidence and predicted classes from probability matrix.

        Args:
            P: Probability matrix of shape (n, k)

        Returns:
            Tuple of (confidence, predictions)
                - confidence: Max probability for each sample
                - predictions: Predicted class labels
        """
        confidence = np.max(P, axis=1)
        predictions = np.argmax(P, axis=1)
        return confidence, predictions

    @staticmethod
    def compute_binned_accuracy(confidence, predictions, y_true, num_bins=5):
        """
        Compute accuracy binned by confidence levels.

        Args:
            confidence: Array of confidence scores
            predictions: Array of predicted labels
            y_true: Array of true labels
            num_bins: Number of confidence bins

        Returns:
            Tuple of (bin_centers, bin_accuracies)
        """
        bins = np.linspace(0, 1, num_bins + 1)
        bin_centers = []
        bin_accuracies = []

        for i in range(num_bins):
            mask = (confidence >= bins[i]) & (confidence < bins[i+1])

            if np.sum(mask) == 0:
                continue

            acc = np.mean(predictions[mask] == y_true[mask])
            center = (bins[i] + bins[i+1]) / 2

            bin_centers.append(center)
            bin_accuracies.append(acc)

        return np.array(bin_centers), np.array(bin_accuracies)


class DigitsDatasetLoader:
    """Loader for the digits dataset."""

    def __init__(self, data_dir=None):
        """
        Initialize dataset loader.

        Args:
            data_dir: Path to data directory. If None, uses default relative path.
        """
        if data_dir is None:
            current_dir = os.path.dirname(__file__)
            self.data_dir = os.path.abspath(os.path.join(current_dir, "..", "data"))
        else:
            self.data_dir = data_dir

    def load(self):
        """
        Load and split digits dataset.

        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        path = os.path.join(self.data_dir, "digits_data.npz")

        data = np.load(path)
        X = data["X"]
        y = data["y"]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=0
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=0
        )

        return X_train, y_train, X_val, y_val, X_test, y_test


class CalibrationPlotter:
    """Plot reliability diagrams for model calibration."""

    @staticmethod
    def plot_confidence_reliability(bins_sm_x, bins_sm_y, bins_nn_x, bins_nn_y, save_path):
        """
        Plot confidence vs accuracy reliability diagram.

        Args:
            bins_sm_x, bins_sm_y: Softmax bin centers and accuracies
            bins_nn_x, bins_nn_y: NN bin centers and accuracies
            save_path: Path to save figure
        """
        plt.figure(figsize=(8, 6))

        plt.plot(bins_sm_x, bins_sm_y, marker='o', label='Softmax')
        plt.plot(bins_nn_x, bins_nn_y, marker='s', label='Neural Net')

        # Ideal calibration line
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')

        plt.xlabel("Confidence")
        plt.ylabel("Accuracy")
        plt.title("Confidence vs Accuracy (Reliability Diagram)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main analysis workflow."""
    figures_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "figures"
    ))
    os.makedirs(figures_dir, exist_ok=True)

    # Load data
    loader = DigitsDatasetLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = loader.load()

    d = X_train.shape[1]
    k = len(np.unique(y_train))

    Y_train = DataUtils.labels_to_onehot(y_train, k)
    Y_val = DataUtils.labels_to_onehot(y_val, k)

    # Train models
    print("Training Softmax...")
    softmax_trainer = SoftmaxTrainer(d, k, seed=0)
    W_sm, b_sm, _, _ = softmax_trainer.train(
        X_train, Y_train, y_train,
        X_val, Y_val, y_val,
        epochs=100,
        lr=0.05,
        batch_size=64,
        lam=1e-4,
        checkpoint_on_val=False,
    )

    print("Training Neural Network...")
    nn_trainer = NNTrainer(d, 32, k, seed=0)
    W1, b1, W2, b2, _, _ = nn_trainer.train(
        X_train, Y_train,
        X_val, Y_val,
        epochs=100,
        lr=0.05,
        batch_size=64,
        lam=1e-4,
        checkpoint_on_val=False,
    )

    # Softmax predictions
    softmax_model = SoftmaxLayer(d, k, seed=0)
    softmax_model.W = W_sm
    softmax_model.b = b_sm
    P_sm = softmax_model.forward(X_test)
    conf_sm, pred_sm = ConfidenceAnalyzer.compute_confidence_and_predictions(P_sm)

    # NN predictions
    nn_model = NeuralNetworkModel(d, 32, k, seed=0)
    nn_model.W1, nn_model.b1, nn_model.W2, nn_model.b2 = W1, b1, W2, b2
    _, P_nn = nn_model.forward(X_test)
    conf_nn, pred_nn = ConfidenceAnalyzer.compute_confidence_and_predictions(P_nn)

    # Binned accuracy
    bins_sm_x, bins_sm_y = ConfidenceAnalyzer.compute_binned_accuracy(conf_sm, pred_sm, y_test)
    bins_nn_x, bins_nn_y = ConfidenceAnalyzer.compute_binned_accuracy(conf_nn, pred_nn, y_test)

    # Plot
    save_path = os.path.join(figures_dir, "confidence_reliability.png")
    CalibrationPlotter.plot_confidence_reliability(bins_sm_x, bins_sm_y, bins_nn_x, bins_nn_y, save_path)

    print(f"Saved: {save_path}")


# Backward compatibility functions (deprecated)
def compute_confidence_and_predictions(P):
    return ConfidenceAnalyzer.compute_confidence_and_predictions(P)

def compute_binned_accuracy(confidence, predictions, y_true, num_bins=5):
    return ConfidenceAnalyzer.compute_binned_accuracy(confidence, predictions, y_true, num_bins)

def load_digits_dataset():
    loader = DigitsDatasetLoader()
    return loader.load()


if __name__ == "__main__":
    main()