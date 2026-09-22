from __future__ import annotations

import platform
import sys

import numpy
import pandas
import pytest
import sentencepiece
import sklearn
import torch


def main() -> int:
    print("=== RateMyCollezzz RRM Runtime Check ===")
    print()

    print("Python")
    print("------")
    print("Version:", platform.python_version())
    print("Executable:", sys.executable)
    print()

    print("Core Packages")
    print("-------------")
    print("torch:", torch.__version__)
    print("numpy:", numpy.__version__)
    print("pandas:", pandas.__version__)
    print("scikit-learn:", sklearn.__version__)
    print("sentencepiece:", sentencepiece.__version__)
    print("pytest:", pytest.__version__)
    print()

    print("CUDA")
    print("----")
    print("PyTorch CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)

        print("GPU:", torch.cuda.get_device_name(device))
        print("Compute capability:", f"{properties.major}.{properties.minor}")
        print(
            "VRAM:",
            f"{properties.total_memory / (1024 ** 3):.2f} GB",
        )
    else:
        print(
            "GPU unavailable. CPU execution remains valid for "
            "non-GPU setup, dataset work, and classical baselines."
        )

    print()

    python_ok = sys.version_info[:2] == (3, 11)

    print("Validation")
    print("----------")
    print("Python 3.11:", "PASS" if python_ok else "FAIL")

    if not python_ok:
        print(
            "Expected Python 3.11 for the locked RRM environment, "
            f"but found {platform.python_version()}."
        )
        return 1

    print()
    print("RRM runtime foundation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
