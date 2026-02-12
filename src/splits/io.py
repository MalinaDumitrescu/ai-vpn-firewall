from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def read_capture_list(path: Path) -> List[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]


def write_capture_list(path: Path, capture_ids: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(capture_ids) + "\n", encoding="utf-8")


def load_splits(train_list: Path, val_list: Path, test_list: Path) -> Dict[str, List[str]]:
    return {
        "train": read_capture_list(train_list),
        "val": read_capture_list(val_list),
        "test": read_capture_list(test_list),
    }
