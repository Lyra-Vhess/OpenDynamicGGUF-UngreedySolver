#!/usr/bin/env bash
# Uniform Q5_K_M baseline. Same model, same tokenizer, same lm-eval config as BF16.
set -euo pipefail
# shellcheck source=/dev/null
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
_odg_experiment run --variant q5_k_m "$@"
