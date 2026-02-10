from __future__ import annotations

import os
import random


def set_seed(seed: int, deterministic_torch: bool = False) -> None:
    """
    Set seeds across common libs.
    Safe if some libs aren't installed.

    Notes:
    - For scikit-learn, also pass random_state=seed explicitly where applicable.
    - deterministic_torch=True can slow training but improves reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
