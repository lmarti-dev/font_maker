import os
from pathlib import Path


def ensure_dir(d:Path):
    if "." in d.name:
        p = d.parent
    else:
        p = d
    if not p.is_dir():
        os.makedirs(p)