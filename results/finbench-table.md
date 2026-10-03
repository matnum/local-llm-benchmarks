# FIN-bench-v2 — Finnish capability (run `fi-20261003`)

`cf` formulation, prompt variant `p0`, scored by `tools/score_ll` (pinned llama.cpp logits). Cells are **acc / acc_norm**; `acc` = argmax loglikelihood of the choice text, `acc_norm` divides by the choice's character length, exactly as `lm_eval/api/task.py` does. Items per task: 900, 1000, 1004, 2048, 2344.

| model | arc_c | belebele_fin | goldenswag | scandisent | sib200 | mean acc |
|---|---|---|---|---|---|---|
| poro2-70b-q4_k_m | 0.455 / 0.486 | 0.570 / 0.470 | 0.649 / 0.714 | 0.954 / 0.954 | 0.518 / 0.435 | 0.629 |
| eurollm-9b-q8_0 | 0.421 / 0.445 | 0.537 / 0.471 | 0.521 / 0.587 | 0.944 / 0.944 | 0.596 / 0.567 | 0.604 |
| poro2-8b-q8_0 | 0.396 / 0.420 | 0.469 / 0.417 | 0.587 / 0.638 | 0.942 / 0.942 | 0.356 / 0.263 | 0.550 |
| qwen3-8b-q4_k_m | 0.310 / 0.364 | 0.434 / 0.410 | 0.389 / 0.344 | 0.499 / 0.499 | 0.515 / 0.611 | 0.430 |
