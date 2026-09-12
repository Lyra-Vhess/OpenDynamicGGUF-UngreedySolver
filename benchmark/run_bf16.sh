#!/usr/bin/env bash
# Original model: FunctionGemma 270M BF16/FP16 via Hugging Face. Same harness pin.
set -euo pipefail
# shellcheck source=/dev/null
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
_odg_experiment run --variant bf16 "$@"
