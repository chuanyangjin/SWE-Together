# Opus 4.8 reproduction report

## Bottom line

The benchmark implementation now supports the Sandoq OCI-runner transport and
the paper's strict 109-task × 2-replicate scoring protocol. The production
Sandoq control plane was validated from both the login host and a fresh Slurm
node, including lease creation, health, the unauthenticated `/v1/exec` 401
boundary, and HTTP-404-confirmed deletion. Authenticated execution, nested
Podman + gVisor startup, and a Sandoq-backed Opus cohort remain blocked because
this account has no `OCI_RUNNER_TOKEN_FILE` credential.

The measured row below is therefore an existing **non-canonical Podman
baseline**, not a Sandoq or exact paper reproduction. It is useful for testing
the complete 109×2 metrics path and for estimating the effect of the changed
agent/user setup, but it must not be submitted as the paper's row.

## Protocol comparison

| Setting | Published row / released canonical runner | Measured baseline |
|---|---|---|
| Tasks / replicates | 109 / 2 | 109 / 2 after infra repair |
| Agent | OpenCode 1.15.13, high reasoning | OpenCode 1.15.13, high reasoning |
| Action model route | OpenRouter Claude Opus 4.8 | internal metagen Claude Opus 4.8 |
| User simulator | Gemini 3.1 Pro Preview, temperature 0.5 | Claude Opus 4.8 |
| Agent timeout | 4800 seconds | 2400 seconds |
| Task sandbox | E2B | local rootless Podman |
| Correctness judge | Claude Opus 4.6 | Claude Opus 4.6 (primary row) |
| User Correction tags | single Gemini 3.1 Pro tagger, temperature 0 | single Opus 4.8 tagger, temperature 0, pinned provenance |

## Table 2 comparison

The paper reports the following Claude Opus 4.8 target at judge threshold
`0.85`: pass@1 `63%`, SSR `59%`, pass² `52%`, mean judge `0.801`, User
Correction `1.38`, output+reasoning tokens/task `74.0k`, and minutes/task
`23.3`.

| Row | pass@1 | SSR | pass² | Mean judge | U-Corr | Tok./task | Min./task |
|---|---:|---:|---:|---:|---:|---:|---:|
| Paper Opus 4.8 | 63.0% | 59.0% | 52.0% | 0.801 | 1.38 | 74.0k | 23.3 |
| Podman baseline, Opus 4.6 judge | 51.8% | 45.9% | 39.4% | 0.746 | 1.12* | 61.4k | 19.7 |
| Difference vs displayed paper row | -11.2 pp | -13.1 pp | -12.6 pp | -0.055 | n/a | -12.6k | -3.6 |
| Sensitivity: Podman baseline, Opus 4.8 judge | 53.7% | 47.7% | 42.2% | 0.755 | 1.12* | 61.4k | 19.7 |

`*` User Correction is intentionally not compared numerically: the baseline's
Opus tagger differs from the released evaluator's Gemini tagger. The manuscript
specifies only a multi-label tagger, while the released implementation pins
Gemini; its handling of no-follow-up trials also leaves the exact published
U-Corr denominator underdetermined.
Minutes/task is reported for completeness but is also infrastructure-dependent.
Percentage-point and scalar differences use the paper's displayed rounded
values, not unavailable higher-precision source values.

Both measured rows passed the strict metric-input gate with 109 tasks, 218
trials, two replicates per task, zero infrastructure failures, and zero
completeness issues. There are 216 model-judged trials; the other two had no
substantive patch and are correctly scored as failures without invoking a
judge. Strict metric completeness does not make the changed action/user/sandbox
protocol canonical.

## Validation evidence

- Fresh post-hardening Slurm Sandoq control-plane probe: job `10000963`, exit
  `0`, log
  `pipeline_logs/sandoq_probe.slurm.log`.
- Cohort repair: job `10005958`, exit `0`; exact matrix `218/218`, infra audit
  `218 ok / 0 failed`.
- Finalizer: job `10006834`, exit `0`; Opus 4.6 and Opus 4.8 missing-verdict
  counts both reached zero, and all 218 trials were re-tagged with pinned Opus
  4.8 provenance.
- Full authenticated probe command: `.venv/bin/python sandoq_probe.py`.
- Canonical Sandoq plan: `canonical_full109.json` via `launch.py --env-type
  sandoq`.
- Strict primary metrics artifact:
  `pipeline_logs/opus_k2_table2_opus46_final.json`.
- Sensitivity metrics with an Opus 4.8 judge:
  `pipeline_logs/opus_k2_table2_opus48_final.json`.
- Final local verification: 101/101 unit and integration tests; scoped Ruff,
  Python compilation, shell syntax, and `git diff --check` all pass.
- Artifact security verification: 30,649 relevant files scanned against known
  credentials with zero hits; live model-secret argv scans on both evaluation
  nodes and the final host returned zero; obsolete proxy-header residue was
  removed and the temporary relay was stopped.

Historical infrastructure failures and incomplete result/reward directories
were moved, not deleted, to `trials/opus_k2_infra_archive/` and selectively
replaced through the resumable deficit runner. Strict output requires exactly
two valid trials for every task, complete runtime/token/tag data, no
infrastructure failures, and an exact rubric-backed verdict from the expected
judge for every substantive patch.

## Paper references

- Table 2: <https://arxiv.org/html/2606.29957v1#S3.T2>
- User Correction definition: <https://arxiv.org/html/2606.29957v1#S2.SS3.SSS2>
- Versioned PDF (page 9): <https://arxiv.org/pdf/2606.29957v1#page=9>
