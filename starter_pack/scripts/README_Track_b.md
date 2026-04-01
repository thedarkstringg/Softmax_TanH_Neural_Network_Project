# Track B: Confidence and Reliability Analysis

## Overview

`run_track_b.py` is a comprehensive script implementing Track B requirements for analyzing model confidence and reliability across both linear (Softmax) and nonlinear (Neural Network) models.

## Requirements Met

### 1. Data Loading ✓
- **Loads** `digits_data.npz` (features and labels)
- **Uses** `digits_split_indices.npz` for fixed train/val/test split indices
- **Respects** the pre-determined split (not random)

### 2. Model Training ✓
- **Softmax**: Linear logistic regression trained with SGD
  - 200 epochs, lr=0.05, batch_size=64, L2 regularization (λ=1e-4)
  - Checkpoint on validation loss for best model
  
- **Neural Network**: 2-layer nonlinear network (hidden layer: 32 units, tanh activation)
  - Same training parameters as Softmax for fair comparison
  - Checkpoint on validation loss for best model

### 3. Test Set Metrics ✓

#### Confidence (max predicted class probability)
- Computed as: `max(P)` for each sample's probability distribution P

#### Predictive Entropy
- Computed as: `-sum(P * log(P))` for each sample
- Measures model uncertainty; lower entropy = higher confidence

#### Predictions and Accuracy
- Predicted class: `argmax(P)`
- Test accuracy computed for both models

### 4. Output Components ✓

#### 4.1 Confidence vs Empirical Accuracy Table (5 bins)
- **Display**: Printed table with grid formatting (via tabulate or fallback)
- **Columns**: Bin ID, Confidence Range, Center, Empirical Accuracy, Sample Count
- **Purpose**: Shows whether model confidence aligns with actual accuracy

**Example output:**
```
╒═════╤═════════════════╤═════════╤══════════╤═════════╕
│ Bin │ Conf Range      │ Center  │ Accuracy │ Count   │
╞═════╪═════════════════╪═════════╪══════════╪═════════╡
│ 1   │ [0.50, 0.70]    │ 0.60    │ 0.8234   │ 145     │
...
╘═════╧═════════════════╧═════════╧══════════╧═════════╛
```

#### 4.2 Reliability Plots (5 confidence bins)
- **File**: `confidence_reliability_diagram.png`
- **Layout**: 2 subplots (Softmax left, NN right)
- **Content**: 
  - Binned accuracy vs confidence for each model
  - Perfect calibration line (y=x) shown for reference
  - Grid and legend for clarity

#### 4.3 Confidence/Entropy Comparison Plot
- **File**: `confidence_entropy_comparison.png`
- **Layout**: 2x2 grid
  - Row 1: Softmax confidence distribution, Softmax entropy distribution
  - Row 2: NN confidence distribution, NN entropy distribution
  - Each subplot shows histograms for correct (green) vs incorrect (red) predictions
- **Purpose**: Shows whether model uncertainty reflects prediction correctness

#### 4.4 Combined Reliability Curves
- **File**: `combined_reliability_curves.png`
- **Content**: Both models' reliability curves on single axes for direct comparison
- **Labels**: Softmax as "Linear", NN as "Nonlinear"

#### 4.5 Brief Text Interpretation
- **Printed to console** with section headers
- **Topics**:
  1. Predictive Performance (accuracy comparison)
  2. Confidence Calibration (how well confidence aligns with accuracy)
  3. Entropy Analysis (uncertainty distributions)
  4. Main Project Insight (linear vs nonlinear implications)
  5. Practical Implications (deployment, robustness)

## Key Insights

### Linear vs Nonlinear Models

**Softmax (Linear):**
- Limited to hyperplane decision boundaries
- Often overconfident on complex patterns
- Lower calibration due to structural limitations
- Entropy may be high even for confident but incorrect predictions

**Neural Network (Nonlinear):**
- Learns complex, curved decision boundaries
- Better captures digit manifold structure
- Superior calibration due to learned feature representations
- Sharper entropy separation between correct/incorrect predictions

### Reliability Diagram Interpretation

The reliability diagram plots confidence (x-axis) vs empirical accuracy (y-axis):
- **Perfect calibration**: Points lie on y=x line
- **Overconfident**: Points below y=x line (high confidence, low actual accuracy)
- **Underconfident**: Points above y=x line (low confidence, high actual accuracy)

Well-calibrated models allow practitioners to trust confidence scores for downstream decisions (rejection sampling, uncertainty quantification, etc.).

## Script Structure

```
1. load_digits_dataset_with_split()
   └─> Loads data and uses fixed split indices

2. compute_confidence(P)
   └─> Max probability per sample

3. compute_predictive_entropy(P)
   └─> -sum(P * log(P)) for each sample

4. compute_binned_accuracy(confidence, predictions, y_true, num_bins=5)
   └─> Accuracy in 5 confidence bins

5. Training & Evaluation
   ├─> train_softmax()  [imported from src]
   ├─> train_nn_runner() [imported from src]
   ├─> softmax_forward() [imported from src]
   └─> nn_forward()      [imported from src]

6. Tables
   ├─> print_confidence_accuracy_table()
   └─> print_summary_statistics()

7. Visualizations
   ├─> plot_reliability_diagram()
   ├─> plot_confidence_entropy_comparison()
   └─> plot_combined_reliability_curves()

8. Interpretation
   └─> print_interpretation()

9. main()
   └─> Orchestrates the full pipeline
```

## Usage

```bash
python scripts/run_track_b.py
```

## Output Files

All figures saved to `figures/` directory:

1. **confidence_reliability_diagram.png** (12x5 inches, 300 DPI)
   - Side-by-side reliability diagrams for Softmax and NN
   
2. **confidence_entropy_comparison.png** (14x10 inches, 300 DPI)
   - 2x2 grid comparing distributions across models and metrics
   
3. **combined_reliability_curves.png** (10x7 inches, 300 DPI)
   - Single axes comparison for easy model comparison

## Console Output

The script prints:
- Data loading summary (sample counts, features, classes)
- Confidence-vs-accuracy tables (5-bin analysis)
- Summary statistics (accuracy, confidence, entropy statistics)
- Text interpretation of findings
- File save confirmations

## Dependencies

- **numpy**: Numerical computations
- **matplotlib**: Visualization
- **tabulate** (optional): Pretty table formatting
  - Falls back to simple text formatting if not available

## Project Integration

This script directly addresses the main project question:
**How do linear (Softmax) and nonlinear (Neural Network) models compare in terms of both predictive performance and confidence reliability?**

By analyzing not just accuracy but also calibration and uncertainty, we gain insights into:
- Whether higher accuracy translates to better-calibrated confidence
- How model complexity affects confidence reliability
- Practical implications for model deployment and uncertainty quantification

## Implementation Notes

- Uses fixed random seed (0) for reproducibility
- Trains on fixed split indices to ensure consistent comparisons
- Computes metrics on test set only (no data leakage)
- Entropy computation uses log-clipping to avoid numerical issues
- All visualizations saved at 300 DPI for publication quality
