# plot_training_curves.py

A script for **automatically plotting training curves** from experiment log files. It reads all `.log` files from `results/logs/`, extracts per-epoch metrics, and saves Loss and Accuracy plots for every experiment found.

---

## What the Script Does

1. Scans the `results/logs/` directory for all `.log` files.
2. Parses each file, detecting experiment blocks and extracting metrics per epoch.
3. For each experiment, generates a figure with **two side-by-side plots**:
   - Train Loss vs. Validation Loss
   - Train Accuracy vs. Validation Accuracy
4. Saves all figures to `figures/`.

---

## Dependencies

| Module | Source |
|--------|--------|
| `os`, `re` | Python standard library |
| `matplotlib` | pip |

No ML libraries required — the script works entirely from pre-saved log files.

---

## Project Structure

```
project/
├── scripts/
│   ├── results/
│   │   └── logs/
│   │       ├── experiment1.log
│   │       └── experiment2.log
│   └── plot_training_curves.py
└── figures/                   # created automatically
```

---

## How to Run

```bash
python scripts/plot_training_curves.py
```

One PNG file is saved to `figures/` for each experiment found across all log files.

---

## Log File Format

The script expects `.log` files structured as follows:

```
===== Experiment Name =====
Epoch 1, Train Loss: 1.2345, Train Acc: 0.4500, Val Loss: 1.3000, Val Acc: 0.4200
Epoch 2, Train Loss: 1.1000, Train Acc: 0.5100, Val Loss: 1.1800, Val Acc: 0.4900
...
===== Another Experiment =====
Epoch 1, Train Loss: 0.9800, Train Acc: 0.6200, Val Loss: 1.0200, Val Acc: 0.5900
...
```

- **Experiment header** — a line matching `===== Name =====`
- **Metric line** — parsed with the following regular expression:

```
Epoch (\d+), Train Loss: ([\d.]+), Train Acc: ([\d.]+), Val Loss: ([\d.]+), Val Acc: ([\d.]+)
```

---

## Key Functions

### `parse_log_file(log_path)`
Reads a log file and returns a dictionary of experiments:

```python
{
    "Experiment Name": {
        "epochs":     [1, 2, 3, ...],
        "train_loss": [1.23, 1.10, ...],
        "train_acc":  [0.45, 0.51, ...],
        "val_loss":   [1.30, 1.18, ...],
        "val_acc":    [0.42, 0.49, ...]
    },
    ...
}
```

### `plot_training_curves(experiments, output_dir)`
For each experiment, creates a `matplotlib` figure with two subplots:
- **Left** — loss curves (blue = train, red = validation)
- **Right** — accuracy curves (blue = train, red = validation)

Output filename is derived from the experiment name (spaces → `_`, special characters removed).

---

## Output File Naming

Filenames follow the pattern:

```
{log_basename} - {experiment_name}.png
```

For example, a log file `softmax_digits.log` containing an experiment called `Run lr=0.05` produces:

```
figures/softmax_digits_-_Run_lr=0.05.png
```

---

## Output

Each saved plot includes:
- Markers `o` for training metrics and `s` for validation metrics
- A grid for readability
- Axis labels and a title with the experiment name

This is useful for **detecting overfitting**: if the validation curve diverges upward from the training curve, the model is overfitting.
