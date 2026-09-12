#!/usr/bin/env python3
"""Compare local GGUFs on original test splits (no Hugging Face).

    python benchmark.py --model original.gguf --model opendynamic.gguf
    python benchmark.py --suite release --model original.gguf --model opendynamic.gguf
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.run import main

if __name__ == "__main__":
    raise SystemExit(main())
