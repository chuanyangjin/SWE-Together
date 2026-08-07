#!/bin/bash
#SBATCH --job-name=opus46judge
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=350G
#SBATCH --time=06:00:00
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/pipeline_logs/opus_k2_judge_opus46.slurm.log
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
cd /storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
export http_proxy=http://10.146.5.90:48835 https_proxy=http://10.146.5.90:48835
export HTTP_PROXY="$http_proxy" HTTPS_PROXY="$https_proxy"
export no_proxy=127.0.0.1,localhost,ghcr.io,githubusercontent.com,nodejs.org
export NO_PROXY="$no_proxy"
export HARBOR_PODMAN_NO_PROXY=ghcr.io,githubusercontent.com,nodejs.org
export SWE_NATIVE_PODMAN=1 SWE_PODMAN_STORE_BASE=/dev/shm
export HARBOR_PODMAN_RMI=1 HARBOR_PODMAN_MAX_PULLS=8
export JUDGE_PODMAN_MODEL=anthropic/claude-opus-4-6
echo "host=$(hostname) start=$(date -u +%FT%H:%M:%SZ)"
.venv/bin/python - <<'PY'
import json
import sys
from pathlib import Path

repo = Path("/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together")
sys.path.insert(0, str(repo))
from eval.patch_utils import patch_file_has_changes

trials = repo / "trials" / "opus_k2"
tasks = {path.name: path for path in (repo / "tasks").iterdir() if path.is_dir()}
plan = []
empty = 0
for trial in sorted(trials.glob("*__*")):
    if not trial.is_dir():
        continue
    if not patch_file_has_changes(trial / "agent" / "final.patch"):
        empty += 1
        continue
    prefix = trial.name.rsplit("__", 1)[0]
    task = tasks.get(prefix) or next(
        (path for name, path in tasks.items() if name.startswith(prefix)), None
    )
    if task and (task / "canonical_goals.json").exists():
        plan.append(
            {
                "trial_dir": str(trial),
                "task_dir": str(task),
                "out_name": "judge_verdict_opus46.json",
            }
        )
out = trials / "judge_plan_opus46.json"
out.write_text(json.dumps(plan, indent=2))
print(f"OPUS 4.6 JUDGE PLAN: {len(plan)} trials (empty/no-change={empty})")
PY
bash judge_local.sh \
  --plan trials/opus_k2/judge_plan_opus46.json \
  --workers 8 --skip-phase1
echo "OPUS46_JUDGE_DONE end=$(date -u +%FT%H:%M:%SZ)"
