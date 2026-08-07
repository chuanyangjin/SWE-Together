# Sushi step-575 lane operations

The action root is `trials/sushi_step575_k2`; pilot roots are evidence only and
must never be copied into it.

The unauthenticated prototype jobs `10077615` / `10079809` were security-
contained. Their eight incomplete directories are recoverably quarantined under
`trials/sushi_step575_k2_cancelled_10079809`; four completed, protocol-equivalent
cells remain in the live root. The authenticated service is job `10080689` and
the earlier secure producer `10081486` was cancelled after discovery that its
login-node egress relay was still unauthenticated. Its eight incomplete
directories are under `trials/sushi_step575_k2_cancelled_10081486`; one late
post-cancel fragment is under
`trials/sushi_step575_k2_cancelled_10081486_late`. Neither quarantine contains
a `result.json`.

The replacement exact-source-IP relay runs in tmux session
`sushi-egress-relay`. Its owner-only registry is
`sushi_lane/state/egress_relay_clients.json`; never edit that file directly.
Launchers must use `sushi_lane/register_relay_client.py` with a job-specific
label, then remove that label in `EXIT`, `TERM`, and `INT` cleanup. The relay
reloads the registry for every connection and fails closed for an absent,
malformed, symlinked, non-owner, or group/world-readable registry.

Monolithic producer `10083045` was checkpoint-cancelled after its first fresh,
fully validated result. Its eight unfinished/zero-result directories are under
`trials/sushi_step575_k2_checkpoint_10083045`; the live root contained exactly
five completed cells before scale-out.

The remaining work is owned by four disjoint deterministic shards from
`sushi_lane/shard_manifest_k4.json` (sorted round-robin, counts 28/27/27/27,
canonical union 109, no duplicates):

- shard 0: job `10087780` (ended OOM), original service `10080689` / slot 0
- shard 1: job `10087783`, service `10085828` / slot 1
- shard 2: job `10087785`, service `10085829` / slot 2
- shard 3: job `10087786`, service `10085830` / slot 3

The first multi-CPU wave produced six more valid cells. Its shard 1/2/3 jobs
were then checkpoint-cancelled to remove the single-H100 bottleneck; exactly
eight zero-result directories per stopped shard are under
`trials/sushi_step575_k2_replica_checkpoint_shard{1,2,3}`. Eleven completed
cells remained live at replica cutover. A final checkpoint raised each shard
from 8 to 16 workers; its eight zero-result directories per shard are under
`trials/sushi_step575_k2_workers16_checkpoint_shard{0,1,2,3}`. Two old
`cli-task-2a55af` results freshly classified as infrastructure failures are
recoverably archived at `trials/sushi_step575_k2_invalid_opencode_backend` and
their replacements are active. Two later infra-failed cells are recoverably
archived at `trials/sushi_step575_k2_invalid_workers16`; neither is eligible for
the final matrix. The live matrix at the final cutover was exactly
23 reusable results plus 64 active cells. The active jobs built 49/47/52/47
configs respectively, exactly 195 cells. Each job has a distinct task set,
authenticated service/key, source-IP relay label, loopback sampling proxy,
Podman store, and shard lock. Never run the monolithic launcher concurrently
with these shards.

Shard 0 reached the 300 GB Slurm cgroup limit after preserving its completed
cells. Four bounded repair jobs use 500 GB each, fixed 16-worker concurrency,
and the same dedicated service slots. Each waits for its original producer. They archive
only completed invalid cells owned by that shard plus stale no-result cells
while holding its exclusive producer lock, rerun only the exact deficit, and
require strict scoped validation. They make at most three passes:

- shard 0: repair `10100482` (running; replaces stopped unsafe `10095017`)
- shard 1: repair `10095018` (running)
- shard 2: repair `10100483` (running)
- shard 3: repair `10100484` (running)

Shard 1's legacy post-run sanitizer traversed the shared root while 14 foreign
cells were active. Their exact names are recorded in runtime state
`sushi_lane/state/sanitizer_race_trials.txt`; repairs force-archive and
rerun every owned match even if it later appears complete. Repair-time
sanitization now passes exact scheduled task names plus `--completed-only`, so
active and foreign artifacts are immutable. The finalizer retains one full-root
sanitization only after all four action dependencies pass.

If one shard stops before strict completion, resubmit its same index and
matching service slot:

```bash
sbatch sushi_lane/run_full_shard.sbatch <0|1|2|3> <0|1|2|3> 16 3
```

The launcher always passes `--skip-existing`. Completion counting recomputes
the current infra sentinel when no fresh sidecar exists, preserves valid cells,
and reruns missing/invalid cells until the final exact 109x2 validator passes.
Every resume first atomically registers only its Slurm node's route-selected
source IP, verifies authenticated service health and exact served model and
checkpoint identity, then starts a loopback-only sampling proxy. Sandboxes see
only the non-secret placeholder credential; the strong per-run service bearer
is read from an owner-only file and injected by that proxy. Job cleanup removes
the exact-source registration.

After the action validator passes, submit:

```bash
sbatch sushi_lane/finalize_full_k2.sbatch
```

The finalizer rebuilds a missing-Opus-4.6 plan on every pass, resumes the
immutable Vertex Gemini tag sidecar, and emits strict metrics at
`sushi_lane/artifacts/full_postprocess/table2_metrics_strict.json`.
Finalizer `10101335` completed successfully after repairs `10100482`,
`10095018`, `10100483`, and `10100484` established the exact 218-cell matrix.
Its full action validator passed 218/218 with zero errors, the final Opus-4.6
missing plan reached zero, the single-Gemini tag sidecar reached 218/218, and
strict metrics were written to
`sushi_lane/artifacts/full_postprocess/table2_metrics_strict.json`:

```text
Sushi checkpoint step 575 | 14.7% | 9.2% | 5.5% | 0.414 | 7.08 | 64.7k | 27.2
```

After completion, service jobs `10080689`, `10085828`, `10085829`, and
`10085830` were cancelled. Endpoint/key/placeholder files, proxy processes,
relay registrations, and the dedicated relay listener were removed; trials,
artifacts, and logs remain intact.

Once at least 40 completed cells are live, submit
`sbatch sushi_lane/rolling_judge.sbatch`. Its completed-only plan cannot include
mutable active trials; valid Opus 4.6 verdicts are reused by the finalizer.
Rolling pass 1 is job `10090941` (48-cell snapshot, 47 substantive verdicts).
Catch-up pass 2 is job `10091439` with `afterok:10090941`; it snapshots newly
completed missing verdicts at its own start under the same lock and completed
41/41 verdicts. Catch-up pass 3 (`10095710`) completed 32 valid verdicts and
exposed one deterministic patch-apply rejection. After the shared judge fix,
catch-up pass 4 (`10097217`) repairs that verdict and snapshots the newer
completed gap, finishing 29/29 valid verdicts. Catch-up pass 5 (`10098361`)
snapshots the subsequent completed gap, producing 28 valid verdicts and one
schema-invalid judge response. Catch-up pass 6 (`10099387`) produced 15 valid
verdicts but the same schema-invalid row; it requires an exact-rubric-ID schema
repair and mechanical revalidation before final metrics. Catch-up pass 7
(`10100755`) used the tested path, mechanically validated Rudel plus three
other verdicts, then was checkpoint-cancelled; five unfinished rows wrote no
verdict and remain eligible for finalizer repair.
