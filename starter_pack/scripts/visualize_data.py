#!/usr/bin/env python3
"""Visualize the fixed datasets for the Math4AI capstone."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_npz(path):
    data = np.load(path)
    print(f"\n--- Loaded {path.name} ---")
    for key in data.keys():
        print(f"Key: {key}, Shape: {data[key].shape}")
    return data

def plot_synthetic(data, title, ax):
    X = np.vstack([data['X_train'], data['X_val'], data['X_test']])
    y = np.concatenate([data['y_train'], data['y_val'], data['y_test']])
    
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    return scatter

def plot_digits(data):
    X = data['X']
    y = data['y']
    
    fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    fig.suptitle("Example Digits (8x8)", fontsize=16)
    
    for i in range(10):
        # Find first occurrence of each digit
        idx = np.where(y == i)[0][0]
        img = X[idx].reshape(8, 8)
        
        ax = axes[i // 5, i % 5]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Label: {i}")
        ax.axis('off')
    
    plt.tight_layout()

def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    
    # 1. Linear Gaussian
    gaussian_path = data_dir / "linear_gaussian.npz"
    if gaussian_path.exists():
        gaussian_data = load_npz(gaussian_path)
        
        # 2. Moons
        moons_path = data_dir / "moons.npz"
        moons_data = load_npz(moons_path)
        
        # Plot Synthetic Datasets
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plot_synthetic(gaussian_data, "Linear Gaussian Task", ax1)
        plot_synthetic(moons_data, "Nonlinear Moons Task", ax2)
        plt.tight_layout()
        print("\nClose the synthetic data plot window to see the digits...")
        plt.show()

    # 3. Digits
    digits_path = data_dir / "digits_data.npz"
    if digits_path.exists():
        digits_data = load_npz(digits_path)
        plot_digits(digits_data)
        plt.show()

if __name__ == "__main__":
    main()
