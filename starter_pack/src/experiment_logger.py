import os
import sys
from datetime import datetime


class Tee:
    """Output redirection: write to multiple streams (stdout + file)."""

    def __init__(self, *streams):
        """
        Initialize with multiple output streams.

        Args:
            *streams: Variable number of file objects to write to
        """
        self.streams = streams

    def write(self, data):
        """Write data to all streams."""
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        """Flush all streams."""
        for stream in self.streams:
            stream.flush()


class ExperimentLogger:
    """Experiment logging utility with file and console output."""

    def __init__(self, script_dir, run_name):
        """
        Initialize logger.

        Args:
            script_dir: Directory of the script
            run_name: Name of the run (used in log filename)
        """
        self.script_dir = script_dir
        self.run_name = run_name
        self.log_path = self._make_log_path()
        self.original_stdout = None
        self.original_stderr = None

    def _make_log_path(self):
        """Create log path with timestamp."""
        logs_dir = os.path.join(self.script_dir, "results", "logs")
        os.makedirs(logs_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.run_name}_{timestamp}.log"
        return os.path.join(logs_dir, filename)

    def start(self):
        """Start logging: redirect stdout and stderr to file and console."""
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        log_file = open(self.log_path, "w", encoding="utf-8")
        tee = Tee(self.original_stdout, log_file)
        sys.stdout = tee
        sys.stderr = tee

        print(f"Log file: {self.log_path}")

    def stop(self):
        """Stop logging: restore original stdout and stderr."""
        if self.original_stdout is not None:
            sys.stdout = self.original_stdout
        if self.original_stderr is not None:
            sys.stderr = self.original_stderr
        print(f"Saved log to: {self.log_path}")

    def run(self, main_func):
        """
        Run a function with logging enabled.

        Args:
            main_func: Function to execute with logging

        Returns:
            Result from main_func
        """
        self.start()
        try:
            result = main_func()
            print(f"\nFinished successfully.")
            return result
        finally:
            self.stop()