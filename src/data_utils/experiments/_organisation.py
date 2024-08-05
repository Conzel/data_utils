from datetime import datetime
from pathlib import Path
import yaml


def save_args(d: dict, path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    for k, v in d.items():
        if isinstance(v, Path):
            d[k] = v.as_posix()
    d["timestamp"] = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    with open(path / "config.yaml", "w") as f:
        yaml.dump(d, f)
