#!/usr/bin/env bash
# Pod entrypoint. Expected env: POD_INDEX, NUM_SHARDS, PROCS (default 4).
# The repository has already been cloned to /workspace/repo by the pod start command
# (read/write deploy key supplied as env DEPLOY_KEY_B64 - never stored in the repository).
set -uo pipefail
cd /workspace/repo
git config user.name "feddac-runner"; git config user.email "runner@feddac.invalid"
export FEDDAC_DATA=/workspace/data
BR="results/${BRANCH:-pod-$POD_INDEX}"
# resume: if this shard already pushed results, continue from them
if git ls-remote --exit-code --heads origin "$BR" >/dev/null 2>&1; then
  git fetch -q origin "$BR" && git checkout -q -B "$BR" "origin/$BR"
else
  git checkout -q -B "$BR"
fi
mkdir -p logs results && touch results/.keep
exec > >(tee -a "logs/bootstrap-${POD_INDEX}.txt") 2>&1
hb() { echo "[$(date -u +%H:%M:%S)] $*"; git add -A logs results >/dev/null 2>&1; git commit -qm "heartbeat: $*" >/dev/null 2>&1; git push -q origin "HEAD:$BR" >/dev/null 2>&1; }
hb "cloned, installing requirements"
pip install -q -r requirements.txt 2>&1 | tail -2
hb "requirements installed"
python scripts/make_grid.py && for g in pilot diag scale mia; do python scripts/make_grid.py $g; done
nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; free -g | head -2
# pre-download the small datasets now; the large ones load in the background (file-locked)
prep() { python -c "from feddac.data import load_dataset as l; d=l('$1','cpu'); print('dataset ok','$1',tuple(d.x_train.shape))" || echo "dataset FAILED $1"; }
prep mnist; prep cifar10
(prep cifar100; prep tinyimagenet) &
hb "datasets ready, starting worker"
python scripts/worker.py --jobs "${JOBS:-configs/jobs.jsonl}" --sync_every "${SYNC:-600}" --shard "$POD_INDEX" --num_shards "$NUM_SHARDS" --procs "${PROCS:-4}" --branch "$BR" ${REVERSE:+--reverse} 2>&1 | tee -a "logs/worker-${POD_INDEX}.txt"
git add -A results logs; git commit -qm "pod ${POD_INDEX}: final"; git push -q origin "HEAD:$BR"
# self-terminate to stop billing (runpodctl is pre-installed on RunPod images)
if [ "${AUTO_REMOVE:-1}" = "1" ] && command -v runpodctl >/dev/null; then runpodctl remove pod "$RUNPOD_POD_ID"; fi
sleep infinity
