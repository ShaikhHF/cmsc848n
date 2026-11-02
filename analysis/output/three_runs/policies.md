| Policy | Paired, runs 1/2/3 | Mean paired (95% CI) | Cost, runs 1/2/3 | Mean cost (95% CI) |
|---|---|---:|---|---:|
| Agent | 20 / 17 / 18 | 18.3 (13.0–23.3) | 1.44 / 1.81 / 1.62 | 1.62 (0.92–2.41) |
| Prompt-only | 17 / 18 / 19 | 18.0 (12.7–23.0) | 1.55 / 1.42 / 1.52 | 1.49 (0.86–2.19) |
| Threshold, cutoffs from run 1 | 22 / 20 / 19 | 20.3 (15.0–25.3) | 1.20 / 1.34 / 1.39 | 1.31 (0.64–2.08) |
| Threshold, re-tuned each run | 22 / 10 / 9 | 13.7 (9.3–18.0) | 1.20 / 1.66 / 1.70 | 1.52 (0.94–2.15) |
| Threshold, keeps agent's pairs | 20.0 / 18.2 / 17.8 | 18.7 (11.6–23.8) | 1.28 / 1.50 / 1.43 | 1.40 (0.83–2.16) |
| Always abstain | 0 / 0 / 0 | 0.0 (0.0–0.0) | 1.50 / 1.50 / 1.50 | 1.50 (1.50–1.50) |
| Always act | 0 / 0 / 0 | 0.0 (0.0–0.0) | 4.22 / 4.22 / 4.22 | 4.22 (3.83–4.61) |

"Threshold, keeps agent's pairs" is exploratory: one cutoff for all categories, the cheapest that gets at least as many pairs right as the plain agent. It was written after seeing these pairs. Each cutoff is chosen on half the pairs in one run and scored on the other half in another run, so its run values are averages over many splits.
