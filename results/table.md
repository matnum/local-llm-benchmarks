# Results

> Rollup built by harness `0f7ca5cea5bf`. Sustained = llama-server streaming 512-token generations. bench = llama-bench (mean of repeats). probe W = GPU power mean during server phase. TTFT mean is inflated where the per-request spread is wide (Vulkan) — compare with p50.

## run-20261002-1719 — rocm

| cell | quant | pp2048 t/s | tg512 bench t/s | tg server t/s | TTFT mean ms | TTFT p50 ms | probe W | tok/s/W | VRAM peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b-q5_k_xl | Q5_K_XL | 341.05 | 10.36 | 10.23 | 145.3 | 137.9 | 85.1 | 0.12 | 19.65 |

## run-20261002-1734 — vulkan

| cell | quant | pp2048 t/s | tg512 bench t/s | tg server t/s | TTFT mean ms | TTFT p50 ms | probe W | tok/s/W | VRAM peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b-mxfp4 | MXFP4 (native) | 780.42 | 55.12 | 53.18 | 124.0 | 143.9 | 85.6 | 0.62 | 59.09 |
| qwen3-coder-30b-a3b-q6_k | Q6_K | 1140.09 | 74.75 | 71.46 | 114.8 | 162.8 | 80.7 | 0.89 | 23.76 |
| qwen3.5-35b-a3b-q6_k | Q6_K | 1117.97 | 61.77 | 60.26 | 576.3 | 688.5 | 75.3 | 0.8 | 27.1 |
| qwen3.8-27b-q5_k_xl | Q5_K_XL | 328.65 | 10.6 | 10.52 | 1350.9 | 1631.9 | 84.1 | 0.13 | 19.06 |
| qwen3.8-27b-q6_k | Q6_K | 317.69 | 10.01 | 9.96 | 1087.7 | 809.1 | 84.5 | 0.12 | 20.09 |

## run-20261002-1751 — rocm

| cell | quant | pp2048 t/s | tg512 bench t/s | tg server t/s | TTFT mean ms | TTFT p50 ms | probe W | tok/s/W | VRAM peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b-mxfp4 | MXFP4 (native) | 906.9 | 51.46 | 49.59 | 68.1 | 67.2 | 86.1 | 0.58 | 59.36 |
| qwen3-coder-30b-a3b-q6_k | Q6_K | 1288.43 | 63.2 | 58.59 | 24.7 | 22.5 | 86.5 | 0.68 | 24.12 |
| qwen3.5-35b-a3b-q6_k | Q6_K | 1140.08 | 52.82 | 50.8 | 57.9 | 58.1 | 85.5 | 0.59 | 27.46 |
| qwen3.8-27b-q5_k_xl | Q5_K_XL | 339.18 | 10.35 | 10.23 | 155.9 | 137.3 | 85.2 | 0.12 | 19.65 |
| qwen3.8-27b-q6_k | Q6_K | 336.27 | 9.84 | 9.73 | 155.7 | 150.0 | 85.4 | 0.11 | 20.59 |

## run-20261003-1543 — vulkan

| cell | quant | pp2048 t/s | tg512 bench t/s | tg server t/s | TTFT mean ms | TTFT p50 ms | probe W | tok/s/W | VRAM peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| eurollm-9b-q8_0 | Q8_0 | 911.37 | 23.54 | 23.06 | 130.2 | 48.7 | 85.1 | 0.27 | 9.48 |
| poro2-8b-q8_0 | Q8_0 | 1025.83 | 27.13 | 26.61 | 169.6 | 234.3 | 84.7 | 0.31 | 8.21 |
| qwen3-32b-q5_k_m | Q5_K_M | 264.51 | 9.71 | 9.64 | 350.7 | 471.9 | 85.0 | 0.11 | 22.52 |
| qwen3-4b-q8_0 | Q8_0 | 2056.79 | 49.1 | 47.69 | 92.9 | 27.7 | 83.5 | 0.57 | 4.81 |

## run-20261003-1556 — rocm

| cell | quant | pp2048 t/s | tg512 bench t/s | tg server t/s | TTFT mean ms | TTFT p50 ms | probe W | tok/s/W | VRAM peak GiB |
|---|---|---|---|---|---|---|---|---|---|
| eurollm-9b-q8_0 | Q8_0 | 1089.07 | 23.02 | 22.57 | 50.5 | 47.2 | 85.2 | 0.26 | 9.74 |
| poro2-8b-q8_0 | Q8_0 | 1277.2 | 26.65 | 26.16 | 43.0 | 41.9 | 85.2 | 0.31 | 8.47 |
| qwen3-32b-q5_k_m | Q5_K_M | 306.88 | 9.47 | 9.31 | 118.3 | 113.2 | 85.3 | 0.11 | 22.79 |
| qwen3-4b-q8_0 | Q8_0 | 2392.52 | 46.83 | 44.97 | 26.0 | 24.4 | 85.2 | 0.53 | 5.06 |
