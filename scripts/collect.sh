#!/usr/bin/env bash
# Collector loop (runs on a pod with the deploy key): merges result JSONs and worker logs of all
# results/* branches into snapshot.zip and force-pushes it to the branch 'snapshots' every $EVERY s.
set -u
EVERY=${EVERY:-900}
cd /workspace/repo
while true; do
  git fetch -q origin '+refs/heads/results/*:refs/remotes/origin/results/*'
  rm -rf /tmp/snap && mkdir -p /tmp/snap
  for r in $(git for-each-ref --format='%(refname)' refs/remotes/origin/results); do
    n=${r##*/results/}; mkdir -p /tmp/snap/$n
    git archive "$r" results logs 2>/dev/null | tar -x -C /tmp/snap/$n --wildcards '*.json' '*.txt' 2>/dev/null
  done
  python3 - <<'PY'
import os, glob, zipfile
seen = set(); n = 0
with zipfile.ZipFile("/tmp/snapshot.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for br in sorted(os.listdir("/tmp/snap")):
        for f in glob.glob(f"/tmp/snap/{br}/results/**/*.json", recursive=True):
            rel = f.split(f"/tmp/snap/{br}/", 1)[1]
            if rel in seen: continue
            seen.add(rel); z.write(f, rel); n += 1
        for f in glob.glob(f"/tmp/snap/{br}/logs/worker-*.txt"):
            z.write(f, f"logs/{br}_{os.path.basename(f)}")
print("snapshot runs:", n)
PY
  rm -rf /tmp/snapwt && git worktree add -q --detach /tmp/snapwt HEAD 2>/dev/null
  (cd /tmp/snapwt && git checkout -q --orphan snapshots_tmp && git rm -rq --cached . >/dev/null 2>&1; \
   find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +; cp /tmp/snapshot.zip . && git add snapshot.zip && \
   git commit -qm "snapshot $(date -u +%H:%M)" && git push -qf origin HEAD:snapshots)
  git worktree remove --force /tmp/snapwt 2>/dev/null; git branch -D snapshots_tmp >/dev/null 2>&1
  echo "[$(date -u +%H:%M)] snapshot pushed"
  sleep "$EVERY"
done
