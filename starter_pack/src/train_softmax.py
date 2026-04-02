import numpy as np

from softmax import SoftmaxLayer, LossMetrics
from training_utils import DataUtils


class SoftmaxTrainer:
    """Trainer for softmax regression model."""

    def __init__(self, input_dim, num_classes, seed=0):
        """
        Initialize softmax trainer.

        Args:
            input_dim: Input feature dimension (d)
            num_classes: Number of classes (k)
            seed: Random seed
        """
        self.model = SoftmaxLayer(input_dim, num_classes, seed=seed)
        self.seed = seed

    def evaluate(self, X, Y_onehot, y, lam):
        """Evaluate model on data."""
        P = self.model.forward(X)
        data_loss = LossMetrics.cross_entropy_loss(P, Y_onehot)
        reg_loss = self.model.l2_loss(lam)
        loss = data_loss + reg_loss
        acc = LossMetrics.accuracy(P, y)
        return loss, acc

    def train(
        self,
        X_train, Y_train_onehot, y_train,
        X_val, Y_val_onehot, y_val,
        epochs=100,
        lr=0.05,
        batch_size=64,
        lam=1e-4,
        checkpoint_on_val=False,
    ):
        """
        Train softmax model.

        Returns:
            W, b, history, best_epoch
        """
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        best = {
            "val_loss": np.inf,
            "W": self.model.W.copy(),
            "b": self.model.b.copy(),
            "epoch": 0,
        }

        for epoch in range(epochs):
            for X_batch, Y_batch in DataUtils.make_minibatches(
                X_train, Y_train_onehot, batch_size=batch_size, shuffle=True, seed=self.seed + epoch
            ):
                P_batch = self.model.forward(X_batch)
                dW, db = self.model.backward(X_batch, P_batch, Y_batch)
                dW += self.model.l2_grad(lam)
                self.model.update_weights(dW, db, lr)

            train_loss, train_acc = self.evaluate(X_train, Y_train_onehot, y_train, lam)
            val_loss, val_acc = self.evaluate(X_val, Y_val_onehot, y_val, lam)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if checkpoint_on_val and val_loss < best["val_loss"]:
                best["val_loss"] = val_loss
                best["W"] = self.model.W.copy()
                best["b"] = self.model.b.copy()
                best["epoch"] = epoch

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}, "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )

        if checkpoint_on_val:
            self.model.W = best["W"]
            self.model.b = best["b"]
            return self.model.W, self.model.b, history, best["epoch"]

        return self.model.W, self.model.b, history, None