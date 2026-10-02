# Results

> Sustained = llama-server streaming 512-token generations. bench = llama-bench (mean of repeats). probe W = GPU power mean during server phase.

## run-20261002-1719 — rocm

| cell | quant | pp2048 t/s | tg512 bench t/s | tg server t/s | TTFT ms | probe W | tok/s/W | VRAM peak GiB |
|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b-q5_k_xl | Q5_K_XL | 341.05 | 10.36 | 10.23 | 145.3 | 85.1 | 0.12 | 19.65 |
