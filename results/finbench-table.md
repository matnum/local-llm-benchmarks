# FIN-bench-v2 — Finnish capability (run `pilot`)

> **PILOT — first 150 items per task only.** Not comparable with a full run; a task's score can move several points on the full split.

`cf` formulation, prompt variant `p0`, scored by `tools/score_ll` (pinned llama.cpp logits). Cells are **acc / acc_norm**; `acc` = argmax loglikelihood of the choice text, `acc_norm` divides by the choice's character length, exactly as `lm_eval/api/task.py` does. Items per task: 150.

| model | arc_c | belebele_fin | goldenswag | scandisent | sib200 | mean acc |
|---|---|---|---|---|---|---|
| poro2-8b-q8_0 | 0.340 / 0.413 | 0.400 / 0.420 | 0.573 / 0.633 | 0.947 / 0.947 | 0.340 / 0.300 | 0.520 |
