#!/bin/bash
#SBATCH --job-name=opusk2judge
#SBATCH --account=ram
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=350G
#SBATCH --time=06:00:00
#SBATCH --output=/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together/pipeline_logs/opus_k2_judge.slurm.log
set -x
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
cd /storage/home/chuanyang/ram_multiturn_autodata/SWE-Together
export http_proxy=http://10.146.35.140:48835 https_proxy=http://10.146.35.140:48835
export HTTP_PROXY=http://10.146.35.140:48835 HTTPS_PROXY=http://10.146.35.140:48835
export no_proxy=127.0.0.1,localhost,ghcr.io,githubusercontent.com,nodejs.org
export NO_PROXY=$no_proxy
export HARBOR_PODMAN_NO_PROXY=ghcr.io,githubusercontent.com,nodejs.org
export SWE_NATIVE_PODMAN=1 SWE_PODMAN_STORE_BASE=/dev/shm HARBOR_PODMAN_RMI=1 HARBOR_PODMAN_MAX_PULLS=6
echo "host=$(hostname) start=$(date -u +%FT%H:%M:%SZ)"
.venv/bin/python - <<'PY'
import json
import sys
from pathlib import Path
REPO=Path("/storage/home/chuanyang/ram_multiturn_autodata/SWE-Together")
sys.path.insert(0, str(REPO))
from eval.patch_utils import patch_file_has_changes
tri=REPO/"trials"/"opus_k2"
tasks={p.name:p for p in (REPO/"tasks").iterdir() if p.is_dir()}
plan=[]; empty=0
for d in sorted(tri.glob("*__*")):
    if not d.is_dir(): continue
    fp=d/"agent"/"final.patch"
    if not patch_file_has_changes(fp): empty+=1; continue
    prefix=d.name.rsplit("__",1)[0]
    real=tasks.get(prefix) or next((p for n,p in tasks.items() if n.startswith(prefix)),None)
    if real and (real/"canonical_goals.json").exists():
        plan.append({"trial_dir":str(d),"task_dir":str(real),"out_name":"judge_verdict.json"})
(tri/"judge_plan.json").write_text(json.dumps(plan,indent=2))
print(f"JUDGE PLAN: {len(plan)} trials to score (empty_patch skipped={empty})")
PY
bash judge_local.sh --plan trials/opus_k2/judge_plan.json --workers 8 --skip-phase1
echo "QWEN_K2_JUDGE_DONE rc=$? end=$(date -u +%FT%H:%M:%SZ)"
