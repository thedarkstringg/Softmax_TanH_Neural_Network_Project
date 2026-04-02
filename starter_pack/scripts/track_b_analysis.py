"""
Track B: Confidence and Reliability Analysis
Computes confidence, entropy, and calibration metrics for both models.
"""
import os
import sys
import numpy as np

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
src_dir = os.path.join(parent_dir, "src")
sys.path.insert(0, src_dir)
sys.path.insert(0, parent_dir)

from softmax import SoftmaxLayer
from neural_net import NeuralNetworkModel
from training_utils import DataUtils
from train_softmax import SoftmaxTrainer
from train_nn_runner import NNTrainer
from confidence_analysis import ConfidenceAnalyzer, DigitsDatasetLoader


def compute_entropy(P):
    """Compute Shannon entropy from probability matrix."""
    P_clipped = np.clip(P, 1e-12, 1.0)
    return -np.sum(P_clipped * np.log(P_clipped), axis=1)


def analyze_confidence_correctness(conf, pred, y_true):
    """Analyze confidence for correct vs incorrect predictions."""
    correct = pred == y_true
    incorrect = pred != y_true

    conf_correct = conf[correct]
    conf_incorrect = conf[incorrect]

    return {
        'correct_mean': np.mean(conf_correct) if len(conf_correct) > 0 else 0,
        'correct_std': np.std(conf_correct) if len(conf_correct) > 0 else 0,
        'incorrect_mean': np.mean(conf_incorrect) if len(conf_incorrect) > 0 else 0,
        'incorrect_std': np.std(conf_incorrect) if len(conf_incorrect) > 0 else 0,
        'num_correct': len(conf_correct),
        'num_incorrect': len(conf_incorrect),
    }


def analyze_entropy_correctness(entropy, pred, y_true):
    """Analyze entropy for correct vs incorrect predictions."""
    correct = pred == y_true
    incorrect = pred != y_true

    ent_correct = entropy[correct]
    ent_incorrect = entropy[incorrect]

    return {
        'correct_mean': np.mean(ent_correct) if len(ent_correct) > 0 else 0,
        'correct_std': np.std(ent_correct) if len(ent_correct) > 0 else 0,
        'incorrect_mean': np.mean(ent_incorrect) if len(ent_incorrect) > 0 else 0,
        'incorrect_std': np.std(ent_incorrect) if len(ent_incorrect) > 0 else 0,
    }


def binned_accuracy_table(confidence, predictions, y_true, num_bins=5):
    """Create 5-bin confidence table."""
    bins = np.linspace(0, 1, num_bins + 1)
    results = []

    for i in range(num_bins):
        mask = (confidence >= bins[i]) & (confidence < bins[i+1])
        if np.sum(mask) == 0:
            continue

        n_samples = np.sum(mask)
        acc = np.mean(predictions[mask] == y_true[mask])
        conf_avg = np.mean(confidence[mask])
        results.append({
            'bin': f'{bins[i]:.2f}--{bins[i+1]:.2f}',
            'conf_center': (bins[i] + bins[i+1]) / 2,
            'num_samples': n_samples,
            'accuracy': acc,
            'avg_confidence': conf_avg,
        })

    return results


def main():
    # Load data
    loader = DigitsDatasetLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = loader.load()

    d = X_train.shape[1]
    k = len(np.unique(y_train))

    Y_train = DataUtils.labels_to_onehot(y_train, k)
    Y_val = DataUtils.labels_to_onehot(y_val, k)
    Y_test = DataUtils.labels_to_onehot(y_test, k)

    # Train Softmax
    print("Training Softmax...")
    softmax_trainer = SoftmaxTrainer(d, k, seed=0)
    W_sm, b_sm, _, _ = softmax_trainer.train(
        X_train, Y_train, y_train,
        X_val, Y_val, y_val,
        epochs=100, lr=0.05, batch_size=64, lam=1e-4, checkpoint_on_val=False,
    )

    # Train Neural Network
    print("Training Neural Network...")
    nn_trainer = NNTrainer(d, 32, k, seed=0)
    W1, b1, W2, b2, _, _ = nn_trainer.train(
        X_train, Y_train, X_val, Y_val,
        epochs=100, lr=0.05, batch_size=64, lam=1e-4, checkpoint_on_val=False,
    )

    # Softmax predictions on test set
    softmax_model = SoftmaxLayer(d, k, seed=0)
    softmax_model.W = W_sm
    softmax_model.b = b_sm
    P_sm = softmax_model.forward(X_test)
    conf_sm, pred_sm = ConfidenceAnalyzer.compute_confidence_and_predictions(P_sm)
    entropy_sm = compute_entropy(P_sm)

    # NN predictions on test set
    nn_model = NeuralNetworkModel(d, 32, k, seed=0)
    nn_model.W1, nn_model.b1, nn_model.W2, nn_model.b2 = W1, b1, W2, b2
    _, P_nn = nn_model.forward(X_test)
    conf_nn, pred_nn = ConfidenceAnalyzer.compute_confidence_and_predictions(P_nn)
    entropy_nn = compute_entropy(P_nn)

    # === ANALYSIS ===
    print("\n" + "="*70)
    print("SOFTMAX REGRESSION")
    print("="*70)

    # Overall statistics
    acc_sm = np.mean(pred_sm == y_test)
    print(f"\nTest Accuracy: {acc_sm:.4f}")
    print(f"Mean Confidence: {np.mean(conf_sm):.4f} ± {np.std(conf_sm):.4f}")
    print(f"Mean Entropy: {np.mean(entropy_sm):.4f} ± {np.std(entropy_sm):.4f}")

    # Confidence analysis
    conf_stats_sm = analyze_confidence_correctness(conf_sm, pred_sm, y_test)
    print(f"\nConfidence (Correct): {conf_stats_sm['correct_mean']:.4f} ± {conf_stats_sm['correct_std']:.4f}")
    print(f"Confidence (Incorrect): {conf_stats_sm['incorrect_mean']:.4f} ± {conf_stats_sm['incorrect_std']:.4f}")

    # Entropy analysis
    ent_stats_sm = analyze_entropy_correctness(entropy_sm, pred_sm, y_test)
    print(f"Entropy (Correct): {ent_stats_sm['correct_mean']:.4f} ± {ent_stats_sm['correct_std']:.4f}")
    print(f"Entropy (Incorrect): {ent_stats_sm['incorrect_mean']:.4f} ± {ent_stats_sm['incorrect_std']:.4f}")

    # Binned accuracy
    print(f"\n5-Bin Confidence-Accuracy Table (Softmax):")
    print(f"{'Confidence Bin':<20} {'# Samples':<12} {'Accuracy':<12} {'Avg Conf':<12}")
    print("-" * 56)
    bins_sm = binned_accuracy_table(conf_sm, pred_sm, y_test)
    for row in bins_sm:
        print(f"{row['bin']:<20} {row['num_samples']:<12} {row['accuracy']:.4f}       {row['avg_confidence']:.4f}")

    print("\n" + "="*70)
    print("ONE-HIDDEN-LAYER NEURAL NETWORK")
    print("="*70)

    # Overall statistics
    acc_nn = np.mean(pred_nn == y_test)
    print(f"\nTest Accuracy: {acc_nn:.4f}")
    print(f"Mean Confidence: {np.mean(conf_nn):.4f} ± {np.std(conf_nn):.4f}")
    print(f"Mean Entropy: {np.mean(entropy_nn):.4f} ± {np.std(entropy_nn):.4f}")

    # Confidence analysis
    conf_stats_nn = analyze_confidence_correctness(conf_nn, pred_nn, y_test)
    print(f"\nConfidence (Correct): {conf_stats_nn['correct_mean']:.4f} ± {conf_stats_nn['correct_std']:.4f}")
    print(f"Confidence (Incorrect): {conf_stats_nn['incorrect_mean']:.4f} ± {conf_stats_nn['incorrect_std']:.4f}")

    # Entropy analysis
    ent_stats_nn = analyze_entropy_correctness(entropy_nn, pred_nn, y_test)
    print(f"Entropy (Correct): {ent_stats_nn['correct_mean']:.4f} ± {ent_stats_nn['correct_std']:.4f}")
    print(f"Entropy (Incorrect): {ent_stats_nn['incorrect_mean']:.4f} ± {ent_stats_nn['incorrect_std']:.4f}")

    # Binned accuracy
    print(f"\n5-Bin Confidence-Accuracy Table (Neural Network):")
    print(f"{'Confidence Bin':<20} {'# Samples':<12} {'Accuracy':<12} {'Avg Conf':<12}")
    print("-" * 56)
    bins_nn = binned_accuracy_table(conf_nn, pred_nn, y_test)
    for row in bins_nn:
        print(f"{row['bin']:<20} {row['num_samples']:<12} {row['accuracy']:.4f}       {row['avg_confidence']:.4f}")

    print("\n" + "="*70)
    print("Interpretation complete.")
    print("="*70)


if __name__ == "__main__":
    main()
