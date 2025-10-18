import os
import mlflow

MLRUNS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "mlruns"))
# Convert Windows path to file:// URI format
MLRUNS_URI = f"file:///{MLRUNS.replace(os.sep, '/')}"
os.environ.setdefault("MLFLOW_TRACKING_URI", MLRUNS_URI)

def start_run(run_name: str, tags: dict | None = None, nested: bool = False):
    return mlflow.start_run(run_name=run_name, tags=tags, nested=nested)
