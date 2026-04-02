import os
import re
import matplotlib.pyplot as plt


def parse_log_file(log_path):
    """
    Parse a log file and extract training metrics.
    
    Returns:
        dict: Dictionary with keys:
            - 'experiments': list of experiment names
            - 'metrics': list of dicts, each containing:
                - 'experiment': name
                - 'epochs': list of epoch numbers
                - 'train_loss': list of train losses
                - 'train_acc': list of train accuracies
                - 'val_loss': list of validation losses
                - 'val_acc': list of validation accuracies
    """
    experiments = {}
    current_experiment = None
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Check if this is an experiment header
            if line.startswith('=====') and line.endswith('====='):
                # Extract experiment name
                current_experiment = line.strip('= ')
                experiments[current_experiment] = {
                    'epochs': [],
                    'train_loss': [],
                    'train_acc': [],
                    'val_loss': [],
                    'val_acc': []
                }
            
            # Check if this is an epoch line
            elif current_experiment and line.startswith('Epoch'):
                match = re.match(
                    r'Epoch (\d+), Train Loss: ([\d.]+), Train Acc: ([\d.]+), Val Loss: ([\d.]+), Val Acc: ([\d.]+)',
                    line
                )
                if match:
                    epoch, train_loss, train_acc, val_loss, val_acc = match.groups()
                    experiments[current_experiment]['epochs'].append(int(epoch))
                    experiments[current_experiment]['train_loss'].append(float(train_loss))
                    experiments[current_experiment]['train_acc'].append(float(train_acc))
                    experiments[current_experiment]['val_loss'].append(float(val_loss))
                    experiments[current_experiment]['val_acc'].append(float(val_acc))
    
    return experiments


def plot_training_curves(experiments, output_dir):
    """
    Generate and save training curves plots for all experiments.
    
    Args:
        experiments (dict): Dictionary of experiments with training metrics
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for exp_name, metrics in experiments.items():
        if not metrics['epochs']:
            continue  # Skip if no data
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = metrics['epochs']
        
        # Plot Loss
        ax1.plot(epochs, metrics['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
        ax1.plot(epochs, metrics['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'{exp_name} - Loss vs Epoch', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot Accuracy
        ax2.plot(epochs, metrics['train_acc'], 'b-o', label='Train Accuracy', linewidth=2, markersize=4)
        ax2.plot(epochs, metrics['val_acc'], 'r-s', label='Val Accuracy', linewidth=2, markersize=4)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title(f'{exp_name} - Accuracy vs Epoch', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        filename = exp_name.replace(':', '').replace('/', '_').replace(' ', '_') + '.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up one level (..) then into results/logs
    logs_dir = os.path.abspath(os.path.join(script_dir, "..", "results", "logs"))

    # Figures is likely in the same parent directory
    figures_dir = os.path.abspath(os.path.join(script_dir, "..", "figures"))
    
    # Find and parse all log files
    all_experiments = {}
    
    if os.path.exists(logs_dir):
        for log_file in sorted(os.listdir(logs_dir)):
            if log_file.endswith('.log'):
                log_path = os.path.join(logs_dir, log_file)
                print(f"Parsing: {log_file}")
                experiments = parse_log_file(log_path)
                
                # Merge experiments (using log filename as prefix for unique naming)
                base_name = log_file.replace('.log', '')
                for exp_name, metrics in experiments.items():
                    full_name = f"{base_name} - {exp_name}"
                    all_experiments[full_name] = metrics
    else:
        print(f"Error: Logs directory not found at {logs_dir}")
        return
    
    if not all_experiments:
        print("No experiments found in log files")
        return
    
    # Generate and save plots
    print(f"\nGenerating plots and saving to: {figures_dir}")
    plot_training_curves(all_experiments, figures_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
