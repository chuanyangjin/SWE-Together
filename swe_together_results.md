# SWE-Together Reproduction Results

| Model | pass@1 ↑ | SSR ↑ | pass² ↑ | Mean judge ↑ | U-Corr ↓ | Tok./task | Min./task |
|---|---:|---:|---:|---:|---:|---:|---:|
| Opus 4.8 (paper) | 63% | 59% | 52% | 0.801 | 1.38 | 74.0k | 23.3 |
| Opus 4.8 (ours) | 51.8% | 45.9% | 39.4% | 0.746 | 1.34 | 61.4k | 19.7 |
| qwen3.5-4b | 6.0% | 1.8% | 1.8% | 0.197 | 3.78 | 23.9k | 8.0 |
| sushi_0803_step575 | 14.7% | 9.2% | 5.5% | 0.414 | 7.08 | 64.7k | 27.2 |
| Olmo_0716_step500 | 17.4% | 12.8% | 9.2% | 0.410 | 6.75 | 78.6k | 34.4 |

- **pass@1 ↑:** The percentage of individual runs with a judge score of at least 0.85.
- **SSR ↑:** The percentage of tasks whose average score across the two runs is at least 0.85.
- **pass² ↑:** The percentage of tasks for which both runs score at least 0.85.
- **Mean judge ↑:** The average continuous correctness score, with no-patch runs scored as 0.
- **U-Corr ↓:** The average correction effort, calculated as corrections plus 0.2 times nudges.
- **Tok./task:** The task-averaged mean model output and reasoning tokens across the two runs.
- **Min./task:** The task-averaged mean wall-clock minutes across the two runs.
- **Tagger:** An LLM that labels each follow-up user message as a correction, nudge, or another interaction type.

Our rows are strict-complete 109-task × 2-replicate, non-canonical Podman baselines: substantive patches are judged by Opus 4.6 (no-patch runs score 0), and U-Corr uses the released benchmark's Gemini 3.1 Pro tagger through the Vertex `generateContent` API at temperature 0; other runtime and sampling settings differ from the paper, so the values are not directly comparable.

For Qwen, Sushi, and Olmo, user-simulator temperature 0.5 was requested, but the gateway rejected it and the client retried without an explicit temperature.
