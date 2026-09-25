#!/usr/bin/env bash
# One-shot pod job: build the Zenodo bundle from the repository branches and upload it as a Zenodo draft.
# Expected env (RunPod secrets): GH_TOKEN, ZENODO_TOKEN. Status and log are pushed to branch 'zenodo-status'.
set -uo pipefail
cd /workspace/repo
git config user.name "feddac-runner"; git config user.email "runner@feddac.invalid"
push_status() {  # $1 = message; pushes the current log/status to branch zenodo-status
  rm -rf /tmp/zs && git worktree prune && git worktree add -q --detach /tmp/zs HEAD
  (cd /tmp/zs && git checkout -q --orphan zs_tmp && git rm -rq --cached . >/dev/null 2>&1
   find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
   cp /tmp/zenodo.log . 2>/dev/null; cp /tmp/zenodo_status.json . 2>/dev/null; cp /tmp/zb_work/bundle/CHECKSUMS.sha256 . 2>/dev/null
   echo "$1" > STATE; git add . && git commit -qm "zenodo: $1" && git push -qf origin HEAD:zenodo-status)
  git worktree remove --force /tmp/zs 2>/dev/null; git branch -D zs_tmp >/dev/null 2>&1
}
echo "started $(date -u) token_len=${#ZENODO_TOKEN}" > /tmp/zenodo.log
push_status "started"
pip install -q -r requirements.txt requests 2>&1 | tail -1
( while sleep 300; do push_status "running $(date -u +%H:%M)"; done ) &
HB=$!
python scripts/zenodo_upload.py >> /tmp/zenodo.log 2>&1; rc=$?
kill $HB 2>/dev/null
echo "EXIT $rc" >> /tmp/zenodo.log
push_status "finished rc=$rc"
if command -v runpodctl >/dev/null; then runpodctl remove pod "$RUNPOD_POD_ID"; fi
sleep infinity
