# FIN-bench-v2 — Finnish capability

`cf` formulation, prompt variant `p0`, scored by `tools/score_ll` (pinned llama.cpp logits). `acc` = argmax loglikelihood of the choice text; `acc_norm` divides by the choice's character length, as the harness does.

| model | arc_c acc | belebele_fin acc | goldenswag acc | scandisent acc | sib200 acc | mean acc |
|---|---|---|---|---|---|---|
| poro2-8b-q8_0 | 0.340 | 0.400 | 0.573 | 0.947 | 0.340 | 0.520 |
