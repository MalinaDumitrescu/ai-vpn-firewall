from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hashlib
import json
import pickle


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def save_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


@dataclass(frozen=True)
class FeatureArtifacts:
    """
    Stored under artifacts/features/ (as in your project layout).
    """
    feature_columns_json: Path
    scaler_pkl: Path
    feature_config_hash_txt: Path


def default_feature_artifacts(artifacts_features_dir: Path) -> FeatureArtifacts:
    return FeatureArtifacts(
        feature_columns_json=artifacts_features_dir / "feature_columns.json",
        scaler_pkl=artifacts_features_dir / "scaler.pkl",
        feature_config_hash_txt=artifacts_features_dir / "feature_config_hash.txt",
    )
