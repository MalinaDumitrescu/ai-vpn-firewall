from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _find_repo_root(start: Optional[Path] = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for _ in range(30):
        if (cur / "pyproject.toml").exists():
            return cur
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # Fallback: if running in a notebook or script deep inside, try to find 'src'
    # This is a heuristic if .git/pyproject.toml are missing in some envs
    if (Path.cwd() / "src").exists():
        return Path.cwd()
        
    raise RuntimeError(
        "Could not find repo root. Run inside the repo (pyproject.toml or .git must exist)."
    )


def _is_absolute_path_like(s: str) -> bool:
    """
    True for:
    - absolute POSIX paths (/home/..)
    - absolute Windows paths (C:\\..)
    - UNC paths (\\\\server\\share\\..)
    """
    if not s:
        return False
    if s.startswith("\\\\") or s.startswith("//"):
        return True
    p = Path(s)
    if p.is_absolute():
        return True
    # Windows drive letter pattern like "C:..."
    return len(s) >= 2 and s[1] == ":"


def _resolve_path(repo_root: Path, value: str) -> Path:
    if _is_absolute_path_like(value):
        return Path(value).expanduser().resolve()
    return (repo_root / value).expanduser().resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        # Fallback to empty dict if config file is missing, using defaults
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        # If file exists but is invalid, return empty dict to use defaults
        return {}
    return data


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path
    configs_dir: Path

    data_raw_dir: Path
    data_processed_dir: Path
    data_splits: Path

    data_raw_vnat: Path
    data_raw_iscx: Path
    data_raw_captured: Path

    artifacts_root: Path
    artifacts_features: Path
    artifacts_xgb: Path
    artifacts_ensemble: Path
    artifacts_eval: Path

    demo_logs: Path
    reports_dir: Path  # Added reports_dir

    config_paths_yaml: Path
    
    # Alias for backward compatibility if needed, though artifacts_root is preferred
    @property
    def artifacts_dir(self) -> Path:
        return self.artifacts_root

    def ensure_dirs(self, create_raw_dirs: bool = True) -> None:
        """
        Create directory structure.

        create_raw_dirs=True is convenient for first setup.
        If you prefer to keep raw dirs untouched (e.g., mounted volumes), set it False.
        """
        dirs = [
            self.data_processed_dir,
            self.data_splits,
            self.artifacts_root,
            self.artifacts_features,
            self.artifacts_xgb,
            self.artifacts_ensemble,
            self.artifacts_eval,
            self.demo_logs,
            self.reports_dir,
        ]
        if create_raw_dirs:
            dirs = [
                self.data_raw_dir,
                self.data_raw_vnat,
                self.data_raw_iscx,
                self.data_raw_captured,
                *dirs,
            ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def validate_layout(self, require_raw: bool = False) -> None:
        """
        Basic sanity checks to catch 'empty folders' issues early.
        """
        required = [
            self.repo_root / "src",
        ]
        for p in required:
            if not p.exists():
                raise FileNotFoundError(f"Missing required path: {p}")

        if require_raw:
            raw_dirs = [self.data_raw_vnat, self.data_raw_iscx]
            for d in raw_dirs:
                if not d.exists():
                    raise FileNotFoundError(f"Missing raw dataset directory: {d}")


def load_paths(config_path: Optional[Path] = None) -> ProjectPaths:
    repo_root = _find_repo_root()
    
    # Try to find config file, but don't fail if missing
    default_cfg_path = repo_root / "configs" / "paths.yaml"
    cfg_path = config_path or default_cfg_path
    
    cfg = _load_yaml(cfg_path)

    project_root_value = (cfg.get("project") or {}).get("root", None)
    if isinstance(project_root_value, str) and project_root_value.strip():
        # If config specifies a different root, respect it
        # But usually repo_root is correct from _find_repo_root
        pass

    configs_dir = repo_root / "configs"

    data_cfg = cfg.get("data") or {}
    art_cfg = cfg.get("artifacts") or {}
    demo_cfg = cfg.get("demo") or {}
    
    # Reports dir (often not in yaml, so default it)
    reports_dir = _resolve_path(repo_root, str(cfg.get("reports", "reports")))

    data_raw_dir = _resolve_path(repo_root, str(data_cfg.get("raw", "data/raw")))
    data_processed_dir = _resolve_path(repo_root, str(data_cfg.get("processed", "data/processed")))
    data_splits = _resolve_path(repo_root, str(data_cfg.get("splits", "data/splits")))

    datasets_cfg = data_cfg.get("datasets") or {}
    data_raw_vnat = _resolve_path(repo_root, str(datasets_cfg.get("vnat", "data/raw/vnat")))
    data_raw_iscx = _resolve_path(repo_root, str(datasets_cfg.get("iscx", "data/raw/iscx")))
    data_raw_captured = _resolve_path(
        repo_root, str(datasets_cfg.get("captured", "data/raw/captured"))
    )

    artifacts_root = _resolve_path(repo_root, str(art_cfg.get("root", "artifacts")))
    artifacts_features = _resolve_path(repo_root, str(art_cfg.get("features", "artifacts/features")))
    artifacts_xgb = _resolve_path(repo_root, str(art_cfg.get("xgb", "artifacts/xgb")))
    artifacts_ensemble = _resolve_path(repo_root, str(art_cfg.get("ensemble", "artifacts/ensemble")))
    artifacts_eval = _resolve_path(repo_root, str(art_cfg.get("eval", "artifacts/eval")))

    demo_logs = _resolve_path(repo_root, str(demo_cfg.get("logs", "artifacts/demo_logs")))

    return ProjectPaths(
        repo_root=repo_root,
        configs_dir=configs_dir,
        data_raw_dir=data_raw_dir,
        data_processed_dir=data_processed_dir,
        data_splits=data_splits,
        data_raw_vnat=data_raw_vnat,
        data_raw_iscx=data_raw_iscx,
        data_raw_captured=data_raw_captured,
        artifacts_root=artifacts_root,
        artifacts_features=artifacts_features,
        artifacts_xgb=artifacts_xgb,
        artifacts_ensemble=artifacts_ensemble,
        artifacts_eval=artifacts_eval,
        demo_logs=demo_logs,
        reports_dir=reports_dir,
        config_paths_yaml=cfg_path,
    )


if __name__ == "__main__":
    paths = load_paths()
    paths.ensure_dirs(create_raw_dirs=True)
    paths.validate_layout(require_raw=False)

    print("Repo root:", paths.repo_root)
    print("Configs dir:", paths.configs_dir)
    print("Config:", paths.config_paths_yaml)
    print("data/raw:", paths.data_raw_dir)
    print("data/raw/vnat:", paths.data_raw_vnat)
    print("data/raw/iscx:", paths.data_raw_iscx)
    print("data/raw/captured:", paths.data_raw_captured)
    print("data/processed:", paths.data_processed_dir)
    print("data/splits:", paths.data_splits)
    print("artifacts/root:", paths.artifacts_root)
    print("artifacts/features:", paths.artifacts_features)
    print("artifacts/xgb:", paths.artifacts_xgb)
    print("artifacts/ensemble:", paths.artifacts_ensemble)
    print("artifacts/eval:", paths.artifacts_eval)
    print("demo/logs:", paths.demo_logs)
    print("reports:", paths.reports_dir)
