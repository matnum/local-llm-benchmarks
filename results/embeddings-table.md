| Model | Backend | hit@1 | hit@5 | hit@10 | MRR@10 | nDCG@10 | Passages/s | llama.cpp |
|---|---|---|---|---|---|---|---|---|
| bge-m3 | rocm | 0.867 | 0.967 | 1.000 | 0.910 | 0.925 | 88.5 | 5b59b83f4 |
| bge-m3 | vulkan | 0.867 | 0.967 | 1.000 | 0.910 | 0.925 | 73.2 | 5b59b83f4 |
| nomic-embed-text-v1.5 | rocm | 0.600 | 0.833 | 0.900 | 0.700 | 0.723 | 149.2 | 5b59b83f4 |
| nomic-embed-text-v1.5 | vulkan | 0.600 | 0.833 | 0.900 | 0.700 | 0.723 | 153.7 | 5b59b83f4 |
