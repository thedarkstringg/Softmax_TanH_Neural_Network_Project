import os
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_optimizer_log(log_path):
    """
    Parse optimizer comparison log file and extract metrics for each optimizer.

    Returns:
        dict: {optimizer_name: {metric: list}}
    """
    optimizers = {}
    current_optimizer = None

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Check for optimizer header
            if '===== Optimizer:' in line:
                match = re.search(r'Optimizer: (\w+)', line)
                if match:
                    current_optimizer = match.group(1)
                    optimizers[current_optimizer] = {
                        'epochs': [],
                        'train_loss': [],
                        'train_acc': [],
                        'val_loss': [],
                        'val_acc': []
                    }

            # Parse epoch lines
            elif current_optimizer and line.startswith('['):
                match = re.search(
                    r'\[(\w+)\] Epoch (\d+), .*Train Loss: ([\d.]+), Train Acc: ([\d.]+), Val Loss: ([\d.]+), Val Acc: ([\d.]+)',
                    line
                )
                if match:
                    opt, epoch, train_loss, train_acc, val_loss, val_acc = match.groups()
                    if opt == current_optimizer:
                        optimizers[current_optimizer]['epochs'].append(int(epoch))
                        optimizers[current_optimizer]['train_loss'].append(float(train_loss))
                        optimizers[current_optimizer]['train_acc'].append(float(train_acc))
                        optimizers[current_optimizer]['val_loss'].append(float(val_loss))
                        optimizers[current_optimizer]['val_acc'].append(float(val_acc))

    return optimizers


def plot_optimizer_comparison(optimizers, output_dir):
    """
    Generate comparison plots for different optimizers.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Color scheme for optimizers
    colors = {'sgd': 'blue', 'momentum': 'green', 'adam': 'red'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Training Loss
    ax = axes[0, 0]
    for opt_name, metrics in optimizers.items():
        if metrics['epochs']:
            ax.plot(metrics['epochs'], metrics['train_loss'], 'o-',
                   label=opt_name.upper(), color=colors.get(opt_name, 'black'),
                   linewidth=2, markersize=3, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Loss', fontsize=11)
    ax.set_title('Training Loss vs Epoch', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    ax = axes[0, 1]
    for opt_name, metrics in optimizers.items():
        if metrics['epochs']:
            ax.plot(metrics['epochs'], metrics['val_loss'], 's-',
                   label=opt_name.upper(), color=colors.get(opt_name, 'black'),
                   linewidth=2, markersize=3, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation Loss', fontsize=11)
    ax.set_title('Validation Loss vs Epoch', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Training Accuracy
    ax = axes[1, 0]
    for opt_name, metrics in optimizers.items():
        if metrics['epochs']:
            ax.plot(metrics['epochs'], metrics['train_acc'], 'o-',
                   label=opt_name.upper(), color=colors.get(opt_name, 'black'),
                   linewidth=2, markersize=3, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Accuracy', fontsize=11)
    ax.set_title('Training Accuracy vs Epoch', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 4: Validation Accuracy
    ax = axes[1, 1]
    for opt_name, metrics in optimizers.items():
        if metrics['epochs']:
            ax.plot(metrics['epochs'], metrics['val_acc'], 's-',
                   label=opt_name.upper(), color=colors.get(opt_name, 'black'),
                   linewidth=2, markersize=3, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation Accuracy', fontsize=11)
    ax.set_title('Validation Accuracy vs Epoch', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = os.path.join(output_dir, 'optimizer_comparison_curves.png')
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved optimizer comparison curves: {filepath}")
    plt.close()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "results", "logs")
    figures_dir = os.path.abspath(os.path.join(script_dir, "..", "figures"))

    # Find the most recent optimizer log
    if os.path.exists(logs_dir):
        log_files = sorted([f for f in os.listdir(logs_dir) if 'optimizers' in f])
        if log_files:
            most_recent_log = os.path.join(logs_dir, log_files[-1])
            print(f"Parsing: {os.path.basename(most_recent_log)}")
            optimizers = parse_optimizer_log(most_recent_log)
            plot_optimizer_comparison(optimizers, figures_dir)
        else:
            print("No optimizer log files found")
    else:
        print(f"Logs directory not found at {logs_dir}")
