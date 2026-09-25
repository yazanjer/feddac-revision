#!/usr/bin/env bash
# One-shot model collector (runs on a pod with the deploy key): gathers the saved global models (*.pt)
# of all results/* branches, converts them to a single zip (split into <95 MB parts so that they fit
# GitHub's file-size limit) and pushes the parts to the branch 'models'.
set -u
cd /workspace/repo
git fetch -q origin '+refs/heads/results/*:refs/remotes/origin/results/*'
rm -rf /tmp/models && mkdir -p /tmp/models
for r in $(git for-each-ref --format='%(refname)' refs/remotes/origin/results); do
  git archive "$r" results 2>/dev/null | tar -x -C /tmp/models --wildcards '*.pt' 2>/dev/null
done
echo "models found: $(find /tmp/models -name '*.pt' | wc -l), $(du -sh /tmp/models | cut -f1)"
python3 - <<'PY'
import glob, os, zipfile
fs = sorted(glob.glob("/tmp/models/results/**/*.pt", recursive=True))
with zipfile.ZipFile("/tmp/feddac-models.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in fs:
        z.write(f, os.path.relpath(f, "/tmp/models"))
    z.writestr("MODELS.txt", "\n".join(os.path.relpath(f, "/tmp/models") for f in fs) + "\n")
print("zipped", len(fs))
PY
rm -rf /tmp/mparts && mkdir -p /tmp/mparts && split -b 90m -d /tmp/feddac-models.zip /tmp/mparts/feddac-models.zip.part
(cd /tmp/mparts && sha256sum /tmp/feddac-models.zip | sed 's|/tmp/||' > SHA256 && ls -la >> SHA256)
rm -rf /tmp/mwt && git worktree add -q --detach /tmp/mwt HEAD
(cd /tmp/mwt && git checkout -q --orphan models_tmp && git rm -rq --cached . >/dev/null 2>&1; \
 find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +; cp /tmp/mparts/* . && git add . && \
 git commit -qm "models $(date -u +%H:%M)" && git push -qf origin HEAD:models)
echo "[$(date -u +%H:%M)] models pushed"
