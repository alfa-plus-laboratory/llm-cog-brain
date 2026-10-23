#!/usr/bin/env python3
"""Run the D1 pipeline without installing it: puts src/ on the path, then `python -m d1`.

    data/venv/bin/python scripts/run_d1.py --smoke
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from d1.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
