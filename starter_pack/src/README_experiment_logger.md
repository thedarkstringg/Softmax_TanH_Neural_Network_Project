# Experiment Logger (`src/experiment_logger.py`)

This module provides utilities for logging script output to both the console and a file. It is used across the `scripts/` directory to ensure that experiment results and terminal outputs are preserved for later analysis.

## Features

- **`Tee` Class**: A custom stream object that duplicates any written data to multiple output streams (e.g., `sys.stdout` and a log file).
- **`make_log_path`**: Generates a standardized log file path in `results/logs/` based on the script name and a timestamp.
- **`run_with_logging`**: A wrapper function that executes a `main` function while redirecting all standard output and error streams to a timestamped log file.

## Usage

```python
from experiment_logger import run_with_logging

def main():
    print("This will be logged to both console and file.")

if __name__ == "__main__":
    run_with_logging(main, script_dir=".", run_name="my_experiment")
```
