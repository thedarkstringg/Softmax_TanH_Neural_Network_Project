import numpy as np

from training_utils import DataUtils, MetricsCalculator
from neural_net import NeuralNetworkModel


class NNTrainer:
    """Trainer for 1-hidden-layer neural network."""

    def __init__(self, input_dim, hidden_dim, num_classes, seed=0):
        """
        Initialize NN trainer.

        Args:
            input_dim: Input feature dimension (d)
            hidden_dim: Hidden layer dimension (h)
            num_classes: Number of classes (k)
            seed: Random seed
        """
        self.model = NeuralNetworkModel(input_dim, hidden_dim, num_classes, seed=seed)
        self.seed = seed

    def evaluate(self, X, Y_onehot, lam):
        """Evaluate model on data."""
        H, P = self.model.forward(X)
        loss = self.model.loss(P, Y_onehot, lam)
        acc = MetricsCalculator.accuracy_from_probs(P, Y_onehot)
        return loss, acc

    def train(
        self,
        X_train, Y_train_onehot,
        X_val, Y_val_onehot,
        epochs=100,
        lr=0.05,
        batch_size=64,
        lam=1e-4,
        checkpoint_on_val=False,
    ):
        """
        Train neural network.

        Returns:
            W1, b1, W2, b2, history, best_epoch
        """
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        best = {
            "val_loss": np.inf,
            "weights": self.model.get_weights(),
            "epoch": 0,
        }

        for epoch in range(epochs):
            for X_batch, Y_batch in DataUtils.make_minibatches(
                X_train, Y_train_onehot, batch_size=batch_size, shuffle=True, seed=self.seed + epoch
            ):
                H_batch, P_batch = self.model.forward(X_batch)
                dW1, db1, dW2, db2 = self.model.backward(
                    X_batch, H_batch, P_batch, Y_batch, lam
                )
                self.model.update_weights(dW1, db1, dW2, db2, lr)

            train_loss, train_acc = self.evaluate(X_train, Y_train_onehot, lam)
            val_loss, val_acc = self.evaluate(X_val, Y_val_onehot, lam)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if checkpoint_on_val and val_loss < best["val_loss"]:
                best["val_loss"] = val_loss
                best["weights"] = self.model.get_weights()
                best["epoch"] = epoch

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}, "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

        if checkpoint_on_val:
            self.model.set_weights(best["weights"])
            return self.model.W1, self.model.b1, self.model.W2, self.model.b2, history, best["epoch"]

        return self.model.W1, self.model.b1, self.model.W2, self.model.b2, history, None