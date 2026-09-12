#!/usr/bin/env bash
# OpenDynamicGGUF candidate. Same model, same tokenizer, same lm-eval config as BF16.
set -euo pipefail
# shellcheck source=/dev/null
source "$(cd "$(dirname "$0")" && pwd)/_common.sh"
_odg_experiment run --variant odg "$@"
