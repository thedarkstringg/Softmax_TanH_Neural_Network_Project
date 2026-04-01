import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from softmax import softmax_forward
from neural_net import nn_forward
from training_utils import labels_to_onehot
from train_softmax import train_softmax
from train_nn_runner import train_nn_runner


def compute_confidence_and_predictions(P):
    confidence = np.max(P, axis=1)
    predictions = np.argmax(P, axis=1)
    return confidence, predictions


def compute_binned_accuracy(confidence, predictions, y_true, num_bins=5):
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


from sklearn.model_selection import train_test_split

def load_digits_dataset():
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(current_dir, "..", "data"))
    path = os.path.join(data_dir, "digits_data.npz")

    data = np.load(path)

    
    X = data["X"]
    y = data["y"]

    # Разбиваем как у всех
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=0
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    figures_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "figures"
    ))
    os.makedirs(figures_dir, exist_ok=True)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_digits_dataset()

    d = X_train.shape[1]
    k = len(np.unique(y_train))

    Y_train = labels_to_onehot(y_train, k)
    Y_val = labels_to_onehot(y_val, k)

    # Train models
    print("Training Softmax...")
    W_sm, b_sm, _, _ = train_softmax(
        X_train, Y_train, y_train,
        X_val, Y_val, y_val,
        d, k,
        epochs=100,
        lr=0.05,
        batch_size=64,
        lam=1e-4,
        seed=0,
        checkpoint_on_val=False,
    )

    print("Training Neural Network...")
    W1, b1, W2, b2, _, _ = train_nn_runner(
        X_train, Y_train,
        X_val, Y_val,
        d, 32, k,
        epochs=100,
        lr=0.05,
        batch_size=64,
        lam=1e-4,
        seed=0,
        checkpoint_on_val=False,
    )

    # Softmax predictions
    P_sm = softmax_forward(X_test, W_sm, b_sm)
    conf_sm, pred_sm = compute_confidence_and_predictions(P_sm)

    # NN predictions
    _, P_nn = nn_forward(X_test, W1, b1, W2, b2)
    conf_nn, pred_nn = compute_confidence_and_predictions(P_nn)

    # Binned accuracy
    bins_sm_x, bins_sm_y = compute_binned_accuracy(conf_sm, pred_sm, y_test)
    bins_nn_x, bins_nn_y = compute_binned_accuracy(conf_nn, pred_nn, y_test)

    # Plot
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

    save_path = os.path.join(figures_dir, "confidence_reliability.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()