import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Add src to path
CURRENT_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "src"))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from softmax import softmax_forward, accuracy
from neural_net import nn_forward, initialize_nn
from training_utils import labels_to_onehot
from train_softmax import train_softmax
from train_nn_runner import train_nn_runner, evaluate_nn

def predict_class(X_mesh, W, b, model_type='softmax'):
    """
    Predict class labels for meshgrid points.
    
    Args:
        X_mesh: Feature matrix of shape (n, d)
        W, b: Softmax parameters
        model_type: 'softmax' or 'nn'
    
    Returns:
        y_pred: Predicted class labels of shape (n,)
    """
    if model_type == 'softmax':
        P = softmax_forward(X_mesh, W, b)
    else:  # nn
        # W, b are actually (W1, b1) and (W2, b2) for NN
        _, P = nn_forward(X_mesh, W[0], W[1], W[2], W[3])
    
    return np.argmax(P, axis=1)


def plot_decision_boundary(
    X_train, y_train, 
    model_params, 
    dataset_name, 
    model_name,
    step_size=0.02,
    padding=0.5
):
    """
    Plot decision boundary for a trained model.
    
    Args:
        X_train: Training features (n, 2)
        y_train: Training labels (n,)
        model_params: Model parameters (dict with 'softmax' or 'nn' keys)
        dataset_name: Name of dataset
        model_name: 'Softmax' or 'Neural Net'
        step_size: Step size for meshgrid (0.02 gives fine detail)
        padding: Extra space around data when creating meshgrid
    """
    # Create meshgrid with fixed step size for consistent fine detail
    x_min, x_max = X_train[:, 0].min() - padding, X_train[:, 0].max() + padding
    y_min, y_max = X_train[:, 1].min() - padding, X_train[:, 1].max() + padding
    
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step_size),
        np.arange(y_min, y_max, step_size)
    )
    
    # Reshape meshgrid for prediction
    X_mesh = np.c_[xx.ravel(), yy.ravel()]
    
    # Get predictions
    if 'softmax' in model_params:
        W, b = model_params['softmax']
        Z = predict_class(X_mesh, W, b, model_type='softmax')
    else:  # nn
        W1, b1, W2, b2 = model_params['nn']
        Z = predict_class(X_mesh, (W1, b1, W2, b2), None, model_type='nn')
    
    Z = Z.reshape(xx.shape)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot decision regions with fine contourf
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)
    
    # Plot training data
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=40, 
              edgecolors='k', cmap=plt.cm.Spectral)
    
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_title(f'{dataset_name} - {model_name} Decision Boundary', 
                fontsize=14, fontweight='bold')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    return fig


def load_synthetic_dataset(filename):
    """Load synthetic dataset."""
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(current_dir, "..", "data"))
    path = os.path.join(data_dir, filename)
    
    data = np.load(path)
    return (data["X_train"], data["y_train"],
            data["X_val"], data["y_val"],
            data["X_test"], data["y_test"])


def main():
    """Train models and plot decision boundaries."""
    
    figures_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "figures"
    ))
    os.makedirs(figures_dir, exist_ok=True)
    
    # Training parameters
    epochs_softmax = 200
    epochs_nn = 500
    lr = 0.1
    batch_size = 64
    lam = 1e-4
    seed = 0
    nn_hidden = 32
    
    # Datasets to process
    datasets = [
        ('linear_gaussian.npz', 'Linear Gaussian'),
        ('moons.npz', 'Moons'),
    ]
    
    for filename, dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*50}")
        
        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test = load_synthetic_dataset(filename)
        
        d = X_train.shape[1]
        k = len(np.unique(y_train))
        
        # One-hot encode labels
        Y_train = labels_to_onehot(y_train, k)
        Y_val = labels_to_onehot(y_val, k)
        Y_test = labels_to_onehot(y_test, k)
        
        # Train Softmax
        print(f"\nTraining Softmax on {dataset_name}...")
        W_sm, b_sm, _, _ = train_softmax(
            X_train, Y_train, y_train,
            X_val, Y_val, y_val,
            d, k,
            epochs=epochs_softmax,
            lr=lr,
            batch_size=batch_size,
            lam=lam,
            seed=seed,
            checkpoint_on_val=False,
        )
        print(f"Softmax training complete")
        
        # Train Neural Network
        print(f"Training Neural Network on {dataset_name}...")
        W1, b1, W2, b2, _, _ = train_nn_runner(
            X_train, Y_train,
            X_val, Y_val,
            d, nn_hidden, k,
            epochs=epochs_nn,
            lr=lr,
            batch_size=batch_size,
            lam=lam,
            seed=seed,
            checkpoint_on_val=False,
        )
        print(f"Neural Network training complete")
        
        # Plot decision boundaries
        model_params_softmax = {'softmax': (W_sm, b_sm)}
        model_params_nn = {'nn': (W1, b1, W2, b2)}
        
        # Softmax plot
        print(f"\nPlotting Softmax decision boundary...")
        fig_sm = plot_decision_boundary(
            X_train, y_train,
            model_params_softmax,
            dataset_name,
            'Softmax',
            step_size=0.02,
            padding=0.5
        )
        
        # Save Softmax plot
        filename_sm = f"decision_boundary_{dataset_name.lower().replace(' ', '_')}_softmax.png"
        filepath_sm = os.path.join(figures_dir, filename_sm)
        fig_sm.savefig(filepath_sm, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath_sm}")
        plt.close(fig_sm)
        
        # Neural Network plot
        print(f"Plotting Neural Network decision boundary...")
        fig_nn = plot_decision_boundary(
            X_train, y_train,
            model_params_nn,
            dataset_name,
            'Neural Network',
            step_size=0.02,
            padding=0.5
        )
        
        # Save NN plot
        filename_nn = f"decision_boundary_{dataset_name.lower().replace(' ', '_')}_nn.png"
        filepath_nn = os.path.join(figures_dir, filename_nn)
        fig_nn.savefig(filepath_nn, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath_nn}")
        plt.close(fig_nn)
    
    print(f"\n{'='*50}")
    print("All decision boundary plots saved!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
