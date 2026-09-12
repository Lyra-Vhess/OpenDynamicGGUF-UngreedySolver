#!/usr/bin/env bash
# One command: prepare variants (if needed) → eval BF16/Q4/Q5/Q6/ODG → comparison table.
set -euo pipefail
# shellcheck source=/dev/null
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
python -m experiment prepare "$@"
python -m experiment run --all "$@"
exec python -m experiment compare "$@"
