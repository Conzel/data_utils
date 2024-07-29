from datetime import datetime
from pathlib import Path
import yaml


def save_args(d: dict, path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    if "results_folder" in d:
        d["results_folder"] = d["results_folder"].as_posix()
    d["timestamp"] = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    with open(path / "config.yaml", "w") as f:
        yaml.dump(d, f)
