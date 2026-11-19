"""Dataset helpers: make data.yaml point at wherever the data actually is."""
import os

import yaml

from . import CLASSES


def local_yaml(src, out=None):
    """Copy a data.yaml with `path` set to the folder it sits in.

    The generator writes an absolute path, which breaks once the dataset is moved
    (for example from a local disk to Google Drive or to /content). Returns the new path.
    """
    src = os.path.abspath(src)
    with open(src) as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names")
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    if names != CLASSES:
        raise ValueError(f"{src}: classes {names} do not match {CLASSES}")
    cfg["path"] = os.path.dirname(src)
    out = out or os.path.join(os.path.dirname(src), "data.local.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out
