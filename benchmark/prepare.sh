#!/usr/bin/env bash
# Download FunctionGemma 270M, freeze BF16 GGUF, emit Q4_K_M / Q5_K_M / Q6_K / ODG.
# Everything after this step is evaluation only — quantization is the only variable.
set -euo pipefail
# shellcheck source=/dev/null
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
_odg_experiment prepare "$@"
