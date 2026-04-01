import os
import sys
from datetime import datetime


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def make_log_path(script_dir, run_name):
    logs_dir = os.path.join(script_dir, "results", "logs")
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{run_name}_{timestamp}.log"
    return os.path.join(logs_dir, filename)


def run_with_logging(main_func, script_dir, run_name):
    log_path = make_log_path(script_dir, run_name)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(log_path, "w", encoding="utf-8") as log_file:
        tee = Tee(original_stdout, log_file)
        sys.stdout = tee
        sys.stderr = tee

        try:
            print(f"Log file: {log_path}")
            result = main_func()
            print(f"\nFinished successfully.")
            print(f"Saved log to: {log_path}")
            return result
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr