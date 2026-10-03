#!/usr/bin/env bash
# Download the five ungated "next round" GGUF models into ~/.local/share/models/.
#
# No Hugging Face token required — all five repos are public (ungated):
#   Qwen3-4B (Q8_0), Poro 2 8B (Q8_0), EuroLLM-9B (Q8_0),
#   Qwen3-32B (Q5_K_M), Poro 2 70B (Q4_K_M).
#
# Resumable (wget -c); re-run to continue any partial download. Total ~82 GiB.
# The gated picks (Gemma 3 27B, Llama 3.3 70B) are intentionally NOT here —
# they need an HF token and `huggingface-cli download`, not plain wget.
#
# The local filenames match the `file:` fields in matrix.yaml. A truncated or
# corrupt download is caught by llama-bench/llama-server on load (and by the
# driver's model-file pre-check, which fails fast before a sweep).
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/.local/share/models}"
mkdir -p "$MODELS_DIR"

download() {
    local name="$1" url="$2"
    echo "== ${name}"
    wget -c -O "${MODELS_DIR}/${name}" "${url}"
}

download "Qwen3-4B-Q8_0.gguf" \
    "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf"

download "Llama-Poro-2-8B-Instruct.Q8_0.gguf" \
    "https://huggingface.co/mradermacher/Llama-Poro-2-8B-Instruct-GGUF/resolve/main/Llama-Poro-2-8B-Instruct.Q8_0.gguf"

download "EuroLLM-9B-Instruct-Q8_0.gguf" \
    "https://huggingface.co/bartowski/EuroLLM-9B-Instruct-GGUF/resolve/main/EuroLLM-9B-Instruct-Q8_0.gguf"

download "Qwen3-32B-Q5_K_M.gguf" \
    "https://huggingface.co/Qwen/Qwen3-32B-GGUF/resolve/main/Qwen3-32B-Q5_K_M.gguf"

download "Llama-Poro-2-70B-Instruct.Q4_K_M.gguf" \
    "https://huggingface.co/mradermacher/Llama-Poro-2-70B-Instruct-GGUF/resolve/main/Llama-Poro-2-70B-Instruct.Q4_K_M.gguf"

echo
echo "Done. Cells ready to sweep:"
echo "  /usr/bin/python3 scripts/run_sweep.py --backend rocm --cells qwen3-4b-q8_0   # smoke"
