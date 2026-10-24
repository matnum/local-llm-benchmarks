# Results

> Rollup built by harness `ab44daecbbf0`. Sustained = llama-server streaming 512-token generations. bench = llama-bench (mean of repeats). probe W = GPU power mean during server phase. TTFT = first-token latency of a streaming request; cold TTFT erases the KV cache before each timed request (protocol stamped per row), so mean and p50 agree to within a few percent (worst 1.04x on the published pass). Warm TTFT carries the previous generation's KV work and spreads wide on Vulkan — compare with p50. Peak GPU mem = peak (VRAM + GTT), the residency figure; runs that predate GTT sampling show the VRAM pool only.

## run-20261002-1719 — rocm

| cell | quant | pp2048 t/s | tg512 bench t/s | tg server t/s | TTFT mean ms | TTFT p50 ms | probe W | tok/s/W | peak GPU mem GiB |
|---|---|---|---|---|---|---|---|---|---|
| qwen3.8-27b-q5_k_xl | Q5_K_XL | 341.05 | 10.36 | 10.23 | 145.3 | 137.9 | 85.1 | 0.12 | 19.65 |

## run-20261002-1734 — vulkan

| cell | quant | pp2048 t/s | tg512 bench t/s | tg server t/s | TTFT mean ms | TTFT p50 ms | probe W | tok/s/W | peak GPU mem GiB |
|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b-mxfp4 | MXFP4 (native) | 780.42 | 55.12 | 53.18 | 124.0 | 143.9 | 85.6 | 0.62 | 59.09 |
| qwen3-coder-30b-a3b-q6_k | Q6_K | 1140.09 | 74.75 | 71.46 | 114.8 | 162.8 | 80.7 | 0.89 | 23.76 |
| qwen3.5-35b-a3b-q6_k | Q6_K | 1117.97 | 61.77 | 60.26 | 576.3 | 688.5 | 75.3 | 0.8 | 27.1 |
| qwen3.8-27b-q5_k_xl | Q5_K_XL | 328.65 | 10.6 | 10.52 | 1350.9 | 1631.9 | 84.1 | 0.13 | 19.06 |
| qwen3.8-27b-q6_k | Q6_K | 317.69 | 10.01 | 9.96 | 1087.7 | 809.1 | 84.5 | 0.12 | 20.09 |

## run-20261002-1751 — rocm

| cell | quant | pp2048 t/s | tg512 bench t/s | tg server t/s | TTFT mean ms | TTFT p50 ms | probe W | tok/s/W | peak GPU mem GiB |
|---|---|---|---|---|---|---|---|---|---|
| gpt-oss-120b-mxfp4 | MXFP4 (native) | 906.9 | 51.46 | 49.59 | 68.1 | 67.2 | 86.1 | 0.58 | 59.36 |
| qwen3-coder-30b-a3b-q6_k | Q6_K | 1288.43 | 63.2 | 58.59 | 24.7 | 22.5 | 86.5 | 0.68 | 24.12 |
| qwen3.5-35b-a3b-q6_k | Q6_K | 1140.08 | 52.82 | 50.8 | 57.9 | 58.1 | 85.5 | 0.59 | 27.46 |
| qwen3.8-27b-q5_k_xl | Q5_K_XL | 339.18 | 10.35 | 10.23 | 155.9 | 137.3 | 85.2 | 0.12 | 19.65 |
| qwen3.8-27b-q6_k | Q6_K | 336.27 | 9.84 | 9.73 | 155.7 | 150.0 | 85.4 | 0.11 | 20.59 |

## run-20261003-1543 — vulkan

| cell | quant | pp2048 t/s | tg512 bench t/s | tg server t/s | TTFT mean ms | TTFT p50 ms | probe W | tok/s/W | peak GPU mem GiB |
|---|---|---|---|---|---|---|---|---|---|
| eurollm-9b-q8_0 | Q8_0 | 911.37 | 23.54 | 23.06 | 130.2 | 48.7 | 85.1 | 0.27 | 9.48 |
| poro2-8b-q8_0 | Q8_0 | 1025.83 | 27.13 | 26.61 | 169.6 | 234.3 | 84.7 | 0.31 | 8.21 |
| qwen3-32b-q5_k_m | Q5_K_M | 264.51 | 9.71 | 9.64 | 350.7 | 471.9 | 85.0 | 0.11 | 22.52 |
| qwen3-4b-q8_0 | Q8_0 | 2056.79 | 49.1 | 47.69 | 92.9 | 27.7 | 83.5 | 0.57 | 4.81 |

## run-20261003-1556 — rocm

| cell | quant | pp2048 t/s | tg512 bench t/s | tg server t/s | TTFT mean ms | TTFT p50 ms | probe W | tok/s/W | peak GPU mem GiB |
|---|---|---|---|---|---|---|---|---|---|
| eurollm-9b-q8_0 | Q8_0 | 1089.07 | 23.02 | 22.57 | 50.5 | 47.2 | 85.2 | 0.26 | 9.74 |
| poro2-8b-q8_0 | Q8_0 | 1277.2 | 26.65 | 26.16 | 43.0 | 41.9 | 85.2 | 0.31 | 8.47 |
| qwen3-32b-q5_k_m | Q5_K_M | 306.88 | 9.47 | 9.31 | 118.3 | 113.2 | 85.3 | 0.11 | 22.79 |
| qwen3-4b-q8_0 | Q8_0 | 2392.52 | 46.83 | 44.97 | 26.0 | 24.4 | 85.2 | 0.53 | 5.06 |

## run-20261003-1718 — vulkan

| cell | quant | pp2048 t/s | tg512 bench t/s | tg server t/s | TTFT mean ms | TTFT p50 ms | probe W | tok/s/W | peak GPU mem GiB |
|---|---|---|---|---|---|---|---|---|---|
| poro2-70b-q4_k_m | Q4_K_M | 117.92 | 5.27 | 5.24 | 354.6 | 199.6 | 85.1 | 0.06 | 40.72 |

## run-20261003-1729 — rocm

| cell | quant | pp2048 t/s | tg512 bench t/s | tg server t/s | TTFT mean ms | TTFT p50 ms | probe W | tok/s/W | peak GPU mem GiB |
|---|---|---|---|---|---|---|---|---|---|
| poro2-70b-q4_k_m | Q4_K_M | 131.16 | 5.15 | 5.1 | 210.1 | 200.3 | 85.2 | 0.06 | 41.02 |

## run-20261022-2042 — vulkan

| cell | quant | pp2048 t/s | tg512 bench t/s | tg server t/s | TTFT mean ms | TTFT p50 ms | probe W | tok/s/W | peak GPU mem GiB |
|---|---|---|---|---|---|---|---|---|---|
| eurollm-9b-q8_0 | Q8_0 | 791.03 | 23.57 | 23.28 | 100.5 | 48.8 | 84.7 | 0.27 | 10.67 |
| gpt-oss-120b-mxfp4 | MXFP4 (native) | 735.34 | 55.3 | 53.9 | 108.1 | 116.6 | 85.6 | 0.63 | 59.7 |
| poro2-70b-q4_k_m | Q4_K_M | 81.05 | 5.25 | 5.25 | 226.1 | 200.0 | 85.2 | 0.06 | 42.52 |
| poro2-8b-q8_0 | Q8_0 | 918.43 | 27.16 | 26.84 | 46.9 | 41.6 | 84.8 | 0.32 | 9.26 |
| qwen3-32b-q5_k_m | Q5_K_M | 243.05 | 9.71 | 9.67 | 159.5 | 186.2 | 85.0 | 0.11 | 24.01 |
| qwen3-4b-q8_0 | Q8_0 | 1857.91 | 48.89 | 47.9 | 45.4 | 27.3 | 84.0 | 0.57 | 5.77 |
| qwen3-coder-30b-a3b-q6_k | Q6_K | 1109.95 | 74.86 | 72.52 | 33.7 | 40.7 | 80.4 | 0.9 | 24.39 |
| qwen3.5-35b-a3b-q6_k | Q6_K | 1062.38 | 62.3 | 60.93 | 61.1 | 47.4 | 76.1 | 0.8 | 27.63 |
| qwen3.8-27b-q5_k_xl | Q5_K_XL | 290.34 | 10.56 | 10.51 | 328.9 | 421.8 | 85.0 | 0.12 | 20.09 |
| qwen3.8-27b-q6_k | Q6_K | 273.7 | 9.98 | 9.92 | 1085.1 | 834.9 | 84.1 | 0.12 | 21.13 |

## run-20261022-2124 — rocm

| cell | quant | pp2048 t/s | tg512 bench t/s | tg server t/s | TTFT mean ms | TTFT p50 ms | probe W | tok/s/W | peak GPU mem GiB |
|---|---|---|---|---|---|---|---|---|---|
| eurollm-9b-q8_0 | Q8_0 | 1060.1 | 22.91 | 22.92 | 54.3 | 45.9 | 84.9 | 0.27 | 10.3 |
| gpt-oss-120b-mxfp4 | MXFP4 (native) | 882.6 | 50.91 | 50.31 | 73.0 | 74.4 | 86.0 | 0.59 | 59.31 |
| poro2-70b-q4_k_m | Q4_K_M | 127.06 | 5.18 | 5.18 | 219.3 | 198.3 | 85.2 | 0.06 | 42.15 |
| poro2-8b-q8_0 | Q8_0 | 1239.11 | 26.59 | 26.39 | 42.5 | 41.2 | 84.9 | 0.31 | 8.89 |
| qwen3-32b-q5_k_m | Q5_K_M | 292.88 | 9.54 | 9.52 | 128.1 | 109.9 | 85.4 | 0.11 | 23.68 |
| qwen3-4b-q8_0 | Q8_0 | 2329.53 | 46.83 | 46.23 | 26.9 | 25.3 | 85.4 | 0.54 | 5.53 |
| qwen3-coder-30b-a3b-q6_k | Q6_K | 1246.92 | 61.41 | 59.92 | 30.0 | 21.0 | 87.0 | 0.69 | 24.37 |
| qwen3.5-35b-a3b-q6_k | Q6_K | 1054.68 | 51.32 | 50.54 | 64.6 | 72.8 | 85.0 | 0.59 | 27.4 |
| qwen3.8-27b-q5_k_xl | Q5_K_XL | 308.37 | 10.37 | 10.33 | 168.2 | 137.2 | 85.4 | 0.12 | 19.59 |
| qwen3.8-27b-q6_k | Q6_K | 308.6 | 9.83 | 9.82 | 174.5 | 152.6 | 85.3 | 0.12 | 20.53 |

## run-20261023-1140 — vulkan

| cell | quant | pp2048 t/s | tg512 bench t/s | tg server t/s | TTFT mean ms | TTFT p50 ms | probe W | tok/s/W | peak GPU mem GiB |
|---|---|---|---|---|---|---|---|---|---|
| eurollm-9b-q8_0 | Q8_0 | – | – | 23.32 | 101.7 | 48.3 | – | – | – |
| gpt-oss-120b-mxfp4 | MXFP4 (native) | – | – | 54.16 | 108.0 | 116.5 | – | – | – |
| poro2-70b-q4_k_m | Q4_K_M | – | – | 5.25 | 226.9 | 199.0 | – | – | – |
| poro2-8b-q8_0 | Q8_0 | – | – | 26.82 | 47.3 | 42.3 | – | – | – |
| qwen3-32b-q5_k_m | Q5_K_M | – | – | 9.66 | 159.1 | 185.7 | – | – | – |
| qwen3-4b-q8_0 | Q8_0 | – | – | 48.12 | 43.4 | 27.6 | – | – | – |
| qwen3-coder-30b-a3b-q6_k | Q6_K | – | – | 73.56 | 35.3 | 44.0 | – | – | – |
| qwen3.5-35b-a3b-q6_k | Q6_K | – | – | 60.98 | 63.8 | 48.3 | – | – | – |
| qwen3.8-27b-q5_k_xl | Q5_K_XL | – | – | 10.51 | 331.0 | 422.0 | – | – | – |
| qwen3.8-27b-q6_k | Q6_K | – | – | 9.92 | 1092.6 | 818.5 | – | – | – |

## run-20261023-1159 — rocm

| cell | quant | pp2048 t/s | tg512 bench t/s | tg server t/s | TTFT mean ms | TTFT p50 ms | probe W | tok/s/W | peak GPU mem GiB |
|---|---|---|---|---|---|---|---|---|---|
| eurollm-9b-q8_0 | Q8_0 | – | – | 22.93 | 55.5 | 46.2 | – | – | – |
| gpt-oss-120b-mxfp4 | MXFP4 (native) | – | – | 50.21 | 73.5 | 74.9 | – | – | – |
| poro2-70b-q4_k_m | Q4_K_M | – | – | 5.18 | 219.4 | 196.7 | – | – | – |
| poro2-8b-q8_0 | Q8_0 | – | – | 26.41 | 42.0 | 40.7 | – | – | – |
| qwen3-32b-q5_k_m | Q5_K_M | – | – | 9.52 | 128.9 | 110.9 | – | – | – |
| qwen3-4b-q8_0 | Q8_0 | – | – | 46.29 | 26.5 | 24.8 | – | – | – |
| qwen3-coder-30b-a3b-q6_k | Q6_K | – | – | 59.87 | 30.4 | 21.4 | – | – | – |
| qwen3.5-35b-a3b-q6_k | Q6_K | – | – | 50.62 | 68.8 | 77.2 | – | – | – |
| qwen3.8-27b-q5_k_xl | Q5_K_XL | – | – | 10.33 | 171.7 | 140.4 | – | – | – |
| qwen3.8-27b-q6_k | Q6_K | – | – | 9.81 | 175.3 | 143.6 | – | – | – |

## run-20261024-0910 — vulkan

| cell | quant | pp2048 t/s | tg512 bench t/s | tg server t/s | TTFT mean ms | TTFT p50 ms | probe W | tok/s/W | peak GPU mem GiB |
|---|---|---|---|---|---|---|---|---|---|
| eurollm-9b-q8_0 | Q8_0 | – | – | 23.25 | 140.4 | 140.9 | – | – | – |
| gpt-oss-120b-mxfp4 | MXFP4 (native) | – | – | 54.02 | 376.7 | 370.3 | – | – | – |
| poro2-70b-q4_k_m | Q4_K_M | – | – | 5.24 | 1169.1 | 1179.4 | – | – | – |
| poro2-8b-q8_0 | Q8_0 | – | – | 26.83 | 128.0 | 127.9 | – | – | – |
| qwen3-32b-q5_k_m | Q5_K_M | – | – | 9.66 | 422.9 | 431.0 | – | – | – |
| qwen3-4b-q8_0 | Q8_0 | – | – | 48.09 | 54.8 | 55.3 | – | – | – |
| qwen3-coder-30b-a3b-q6_k | Q6_K | – | – | 73.55 | 180.2 | 179.6 | – | – | – |
| qwen3.5-35b-a3b-q6_k | Q6_K | – | – | 61.12 | 187.1 | 186.9 | – | – | – |
| qwen3.8-27b-q5_k_xl | Q5_K_XL | – | – | 10.48 | 499.9 | 507.9 | – | – | – |
| qwen3.8-27b-q6_k | Q6_K | – | – | 9.91 | 1195.0 | 1211.4 | – | – | – |

## run-20261024-0930 — rocm

| cell | quant | pp2048 t/s | tg512 bench t/s | tg server t/s | TTFT mean ms | TTFT p50 ms | probe W | tok/s/W | peak GPU mem GiB |
|---|---|---|---|---|---|---|---|---|---|
| eurollm-9b-q8_0 | Q8_0 | – | – | 22.9 | 85.2 | 85.3 | – | – | – |
| gpt-oss-120b-mxfp4 | MXFP4 (native) | – | – | 48.88 | 223.7 | 218.9 | – | – | – |
| poro2-70b-q4_k_m | Q4_K_M | – | – | 5.18 | 545.6 | 545.4 | – | – | – |
| poro2-8b-q8_0 | Q8_0 | – | – | 26.48 | 64.5 | 64.4 | – | – | – |
| qwen3-32b-q5_k_m | Q5_K_M | – | – | 9.51 | 221.7 | 225.0 | – | – | – |
| qwen3-4b-q8_0 | Q8_0 | – | – | 46.23 | 43.3 | 43.5 | – | – | – |
| qwen3-coder-30b-a3b-q6_k | Q6_K | – | – | 60.05 | 106.4 | 106.4 | – | – | – |
| qwen3.5-35b-a3b-q6_k | Q6_K | – | – | 50.53 | 134.1 | 129.2 | – | – | – |
| qwen3.8-27b-q5_k_xl | Q5_K_XL | – | – | 10.33 | 338.0 | 338.1 | – | – | – |
| qwen3.8-27b-q6_k | Q6_K | – | – | 9.8 | 356.8 | 358.1 | – | – | – |
