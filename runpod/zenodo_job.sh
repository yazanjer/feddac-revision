#!/usr/bin/env bash
# One-shot pod job: build the Zenodo bundle from the repository branches and upload it as a Zenodo draft.
# Expected env (RunPod secrets): GH_TOKEN, ZENODO_TOKEN. Status and log are pushed to branch 'zenodo-status'.
set -uo pipefail
cd /workspace/repo
git config user.name "feddac-runner"; git config user.email "runner@feddac.invalid"
pip install -q -r requirements.txt requests 2>&1 | tail -1
python scripts/zenodo_upload.py > /tmp/zenodo.log 2>&1; rc=$?
echo "EXIT $rc" >> /tmp/zenodo.log
rm -rf /tmp/zs && git worktree add -q --detach /tmp/zs HEAD
(cd /tmp/zs && git checkout -q --orphan zs_tmp && git rm -rq --cached . >/dev/null 2>&1
 find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
 cp /tmp/zenodo.log . ; cp /tmp/zenodo_status.json . 2>/dev/null; cp /tmp/zb_work/bundle/CHECKSUMS.sha256 . 2>/dev/null
 git add . && git commit -qm "zenodo upload $(date -u +%H:%M) rc=$rc" && git push -qf origin HEAD:zenodo-status)
if command -v runpodctl >/dev/null; then runpodctl remove pod "$RUNPOD_POD_ID"; fi
sleep infinity
