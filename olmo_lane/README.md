# Olmo_0716_step500 SWE-Together lane

This is an independent 109-task x 2-replicate Podman lane for checkpoint
`/checkpoint/ram/chuanyang/autodata/run_0716_single_turn/weights/step_500`.
Its served identity is `Olmo-0716-step500`, so every accepted trial must record
the exact action model `openai/Olmo-0716-step500`.

All mutable trial, endpoint, credential, relay, log, artifact, lock,
invalid-archive, judge, tagger, and sanitizer state is Olmo-specific. The lane
uses its own fail-closed egress relay on port 48837 and its own atomic client
registry. Reused Sushi Python helpers and the shard manifest are read-only.

Before a full run, start four authenticated services and pass the pinned
tool-call smoke on every slot. `run_pilot.sbatch` remains available as an
optional one-cell end-to-end diagnostic. Four full shards use service slots 0
through 3 and disjoint deterministic task sets:

```bash
sbatch olmo_lane/serve_step500.sbatch 0
sbatch olmo_lane/serve_step500.sbatch 1
sbatch olmo_lane/serve_step500.sbatch 2
sbatch olmo_lane/serve_step500.sbatch 3
sbatch olmo_lane/tool_call_smoke.sbatch 0
sbatch olmo_lane/tool_call_smoke.sbatch 1
sbatch olmo_lane/tool_call_smoke.sbatch 2
sbatch olmo_lane/tool_call_smoke.sbatch 3
# Optional deeper diagnostic:
sbatch olmo_lane/run_pilot.sbatch 0
sbatch olmo_lane/run_full_shard.sbatch 0 0 16 3
sbatch olmo_lane/run_full_shard.sbatch 1 1 16 3
sbatch olmo_lane/run_full_shard.sbatch 2 2 16 3
sbatch olmo_lane/run_full_shard.sbatch 3 3 16 3
```

Submit `rolling_judge.sbatch` only for completed-cell patch snapshots. Gemini
tagging stays in finalization because `result.json` can precede the shard-wide
trace sanitizer; the finalizer first holds all producer locks and sanitizes the
complete root. Submit
`finalize_full_k2.sbatch` with `afterok` dependencies on all four strict shard
jobs. The finalizer takes Olmo-only locks, proves the exact 218-cell action
cohort, finishes the Opus 4.6 judge, sanitizes only after action writers have
stopped, tags with Gemini 3.1 Pro, and writes strict Table 2 metrics under
`olmo_lane/artifacts/full_postprocess`.
