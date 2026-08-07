# run_0806 SWE-Together lanes

This directory launches isolated SWE-Together evaluations for one selected
`run_0806_v1` or `run_0806_v2` Hugging Face weight export. It reuses the proven
Sushi service, pinned-sampling proxy, four-way task manifest, archive helper,
and action validator, but owns all mutable endpoints, locks, trials, judge
artifacts, credentials, logs, and relay state below a step-specific lane root.

Do not launch a lane until its checkpoint step has been explicitly selected.
The launcher requires an immutable `weights/step_N` directory with `STABLE`,
Qwen3.5 config identity, tokenizer, safetensor index, and model shards. The lane
name is tied to both the source run and step, for example
`run_0806_v1_step250` for `run_0806_v1/weights/step_250`.

## Protocol

- 109 canonical tasks, two independent replicates (218 trials)
- OpenCode 1.15.13 in the local Podman backend
- Opus 4.8 user simulator, requested temperature 0.5
- 4,800-second agent timeout
- four one-H200 vLLM services and four disjoint 16-worker CPU shards
- Qwen3.5 tool parsing, thinking disabled, 131,072 context
- pinned action sampling: temperature 0.6, top-p 0.95, top-k 20,
  presence penalty 1.5, repetition penalty 1.05, max output 32,768
- overlapping completed-cell judging with Opus 4.6
- final Gemini 3.1 Pro Vertex message tags and strict Table-2 metrics

## Preflight without writes or submissions

Use a dedicated, currently unused relay port. Ports 48835-48837 belong to
older lanes. They also sit inside the host's 32768-60999 ephemeral range, so
new lanes reject that entire range. The planned assignments are 31038 for v1
and 31039 for v2.

```bash
cd /storage/home/chuanyang/ram_multiturn_autodata/SWE-Together

bash rl0806_lane/launch_lane.sh \
  --lane run_0806_v1_step250 \
  --checkpoint /checkpoint/ram/chuanyang/autodata/run_0806_v1/weights/step_250 \
  --relay-port 31038 \
  --dry-run
```

The dry run validates the exact checkpoint and prints all derived roots and
submission shapes. It creates no files and calls no Slurm mutation command.

## Submit after checkpoint selection

Run the same command without `--dry-run`:

```bash
bash rl0806_lane/launch_lane.sh \
  --lane run_0806_v1_step250 \
  --checkpoint /checkpoint/ram/chuanyang/autodata/run_0806_v1/weights/step_250 \
  --relay-port 31038
```

The launcher:

1. creates `rl0806_lane/runs/<lane>/` and `trials/<lane>_k2/` with a private,
   immutable identity protocol;
2. starts a fail-closed login-node relay to the current loopback filtering
   proxy;
3. submits four authenticated one-H200 vLLM services using explicit `h200`
   partition and `h200_ram_high` QOS overrides;
4. waits for their identity preflights, runs a structured tool-call smoke on
   every slot, and requires all four smokes to finish successfully;
5. submits the deterministic 28/27/27/27 task shards with 16 workers, two
   replicates, and up to three repair passes;
6. starts a rolling judge after all shards have started, so immutable completed
   cells are judged while producers continue;
7. submits the finalizer with an `afterok` dependency on all four shards and the
   rolling watcher.

Service jobs are intentionally long-lived and are not used in an `afterok`
dependency. If setup fails before the full graph is committed, the launcher
cancels every job it submitted, stops its relay, and preserves diagnostics.

Operationally, keep the foreground launcher attached while it waits for all
four services and tool smokes (the service wait defaults to two hours). The
secure egress relay is a `setsid` login-node process, not a Slurm job: after a
successful submission it intentionally survives the launcher and must remain
alive until judging and tagging finish. A login-node restart, relay-port
collision, or change to the pinned relay address will interrupt API access and
requires repair before retrying the affected shard/judge job.

The v2 invocation is identical apart from the immutable identity and port:

```bash
bash rl0806_lane/launch_lane.sh \
  --lane run_0806_v2_step325 \
  --checkpoint /checkpoint/ram/chuanyang/autodata/run_0806_v2/weights/step_325 \
  --relay-port 31039 \
  --dry-run
```

Replace the example steps only after selection. Different step labels produce
different state, trial, archive, service, judge, tag, and metric paths.

## Failure and resume

Each shard owns a lane-local lock and deterministic task set. Its producer uses
`--skip-existing`, archives only invalid cells owned by that shard, and retries
up to three passes. If a shard job still fails, the finalizer remains blocked
by `afterok`. Resubmit only that shard with the same `BENCH_*` values recorded
by `launch_lane.sh`/`protocol.json`, then submit a replacement rolling watcher
and finalizer dependency. Never point a repair at another lane's roots.

## Result and cleanup

Final metrics are written to:

```text
rl0806_lane/runs/<lane>/artifacts/full_postprocess/table2_metrics_strict.json
```

After the finalizer completes, stop the four services and relay:

```bash
bash rl0806_lane/stop_lane.sh run_0806_v1_step250
```

Without `--force`, cleanup requires the finalizer and strict 109-task/218-trial
metric gate. It prints the Markdown-ready row. The same row can be extracted
independently with:

```bash
jq -er '
  select(.status == "strict_complete"
    and .metric_complete == true
    and .canonical_u_corr_protocol_complete == true
    and .aggregates.n_tasks == 109
    and .aggregates.n_trials == 218
    and .aggregates.n_tasks_with_full_k == 109)
  | "| " + .row + " |"
' rl0806_lane/runs/<lane>/artifacts/full_postprocess/table2_metrics_strict.json
```

Only after this gate passes should that line be added to
`swe_together_results.md`.

The finalizer deliberately does not stop services or the relay and never
deletes trials or invalid-shard archives. Normal cleanup only cancels the four
services and stops the relay; `--force` also cancels every recorded shard,
watcher, and finalizer job for an intentional abort. A failed shard leaves the
`afterok` finalizer blocked, so use the exact lane-local protocol when manually
resubmitting and replace the watcher/finalizer dependency afterward.
