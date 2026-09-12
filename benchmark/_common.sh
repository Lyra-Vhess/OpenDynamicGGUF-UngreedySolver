# Shared launcher for the FunctionGemma 270M experiment scripts.
# Sourced by run_*.sh / prepare.sh / run_all.sh — not meant to be run alone.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "${LLAMA_CPP_DIR:-}" && -x "${HOME}/.unsloth/llama.cpp/build/bin/llama-quantize" ]]; then
  export LLAMA_CPP_DIR="${HOME}/.unsloth/llama.cpp"
fi

_odg_experiment() {
  exec python -m experiment "$@"
}
