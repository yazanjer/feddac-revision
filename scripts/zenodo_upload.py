#!/usr/bin/env python
"""Build the Zenodo bundle from the repository branches and upload it as a Zenodo DRAFT.

Runs on a pod inside a clone of this repository (env: ZENODO_TOKEN, optional ZENODO_URL, default
https://zenodo.org). Steps:
  1. merge the per-run result JSONs and worker logs of all results/* branches;
  2. reassemble the trained-model archive from the 'models' branch (split parts + SHA256) and add MODELS.md;
  3. regenerate the paper tables/figures (scripts/aggregate.py) and the code/results/assets zips
     (scripts/package_zenodo.py);
  4. create a Zenodo deposition with the metadata of .zenodo.json, pre-reserve a DOI and upload the files.
The deposition is NOT published; the owner reviews and publishes it on zenodo.org.
A status file (deposition id, reserved DOI, file checksums) is written to /tmp/zenodo_status.json.
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile

import requests

REPO = os.getcwd()
WORK = "/tmp/zb_work"
ZURL = os.environ.get("ZENODO_URL", "https://zenodo.org").rstrip("/")
TOKEN = os.environ.get("ZENODO_TOKEN", "")


def sh(cmd, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def merge_results():
    res, logs = os.path.join(WORK, "results"), os.path.join(WORK, "logs")
    os.makedirs(res, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    sh("git fetch -q origin '+refs/heads/results/*:refs/remotes/origin/results/*' '+refs/heads/models:refs/remotes/origin/models'")
    refs = subprocess.run("git for-each-ref --format='%(refname)' refs/remotes/origin/results", shell=True,
                          capture_output=True, text=True).stdout.split()
    n = 0
    for r in sorted(refs):
        br = r.split("/results/", 1)[1]
        tmp = os.path.join(WORK, "br", br)
        os.makedirs(tmp, exist_ok=True)
        subprocess.run(f"git archive {r} results logs 2>/dev/null | tar -x -C {tmp} 2>/dev/null", shell=True)
        for f in glob.glob(f"{tmp}/results/*/*.json"):
            rel = os.path.relpath(f, os.path.join(tmp, "results"))
            dst = os.path.join(res, rel)
            if not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(f, dst)
                n += 1
        for f in glob.glob(f"{tmp}/logs/*.txt") + glob.glob(f"{tmp}/logs/*.log"):
            shutil.copy(f, os.path.join(logs, f"{br}_{os.path.basename(f)}"))
    shutil.copytree(os.path.join(REPO, "results", "analysis"), os.path.join(res, "analysis"), dirs_exist_ok=True)
    print("merged result files:", n, flush=True)
    return res, logs


def build_models(res_dir, out_zip):
    parts_dir = os.path.join(WORK, "mparts")
    os.makedirs(parts_dir, exist_ok=True)
    sh(f"git archive refs/remotes/origin/models | tar -x -C {parts_dir}")
    raw = os.path.join(WORK, "feddac-models-raw.zip")
    with open(raw, "wb") as o:
        for p in sorted(glob.glob(f"{parts_dir}/feddac-models.zip.part*")):
            with open(p, "rb") as f:
                shutil.copyfileobj(f, o)
    expected = open(os.path.join(parts_dir, "SHA256")).read().split()[0]
    got = sha256(raw)
    if got != expected:
        raise SystemExit(f"model archive checksum mismatch: {got} != {expected}")
    idx = {}
    for f in glob.glob(f"{res_dir}/*/*.json"):
        try:
            d = json.load(open(f))
            idx[d.get("run_id")] = d
        except Exception:
            pass
    src = zipfile.ZipFile(raw)
    names = sorted(n for n in src.namelist() if n.endswith(".pt"))
    rows = ["# FedDAC trained global models", "",
            "State dicts of the final global model after the last round (`torch.save`). Load with",
            "`feddac.models.build_model(dataset, num_classes)` and `model.load_state_dict(torch.load(path))`.",
            "ResNet-18-GN weights (CIFAR-100, Tiny-ImageNet) are stored in float16; call `.float()` after loading.",
            "Each file name ends with the run id of the matching result JSON in feddac-results-v1.0.0.zip.", "",
            "| file | experiment | dataset | method | N | DP eps | optimiser | final acc (%) |", "|---|---|---|---|---|---|---|---|"]
    for nme in names:
        rid = os.path.basename(nme)[:-3]
        exp = nme.split("/")[2]
        d = idx.get(rid, {})
        c = d.get("config", {})
        opt = f"{c.get('optimizer', 'adam')} {c.get('lr', '')}" if c else ""
        acc = f"{d['final_acc']:.2f}" if d else "n/a"
        rows.append(f"| {nme.replace('results/', '')} | {exp} | {c.get('dataset', '')} | {c.get('method', '')} | "
                    f"{c.get('clients', '')} | {c.get('dp_eps') or '-'} | {opt} | {acc} |")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for i in src.infolist():
            if i.filename in ("MODELS.md", "MODELS.txt"):
                continue
            with src.open(i) as f, z.open(i, "w") as g:
                shutil.copyfileobj(f, g, 1 << 20)
        z.writestr("MODELS.md", "\n".join(rows) + "\n")
    print("models:", len(names), flush=True)
    return len(names)


def zenodo(files, meta):
    if not TOKEN:
        raise SystemExit("ZENODO_TOKEN is empty")
    H = {"Authorization": f"Bearer {TOKEN}"}
    r = requests.post(f"{ZURL}/api/deposit/depositions", headers=H, json={}, timeout=60)
    r.raise_for_status()
    dep = r.json()
    md = {k: meta[k] for k in ("title", "upload_type", "description", "creators", "license", "keywords", "version") if k in meta}
    md["prereserve_doi"] = True
    md["access_right"] = "open"
    r = requests.put(f"{ZURL}/api/deposit/depositions/{dep['id']}", headers=H, json={"metadata": md}, timeout=60)
    if r.status_code >= 400:
        print("metadata update failed:", r.status_code, r.text[:500], flush=True)
    else:
        dep = r.json()
    bucket = dep["links"]["bucket"]
    up = {}
    for p in files:
        name = os.path.basename(p)
        print("uploading", name, os.path.getsize(p), flush=True)
        with open(p, "rb") as f:
            rr = requests.put(f"{bucket}/{name}", data=f, headers=H, timeout=7200)
        rr.raise_for_status()
        up[name] = {"size": os.path.getsize(p), "zenodo_checksum": rr.json().get("checksum"), "sha256": sha256(p)}
    dep = requests.get(f"{ZURL}/api/deposit/depositions/{dep['id']}", headers=H, timeout=60).json()
    return {"deposition_id": dep["id"], "state": dep.get("state"), "submitted": dep.get("submitted"),
            "html": dep["links"].get("html"), "reserved_doi": dep.get("metadata", {}).get("prereserve_doi", {}).get("doi"),
            "files": up}


def main():
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    os.makedirs(WORK)
    res, logs = merge_results()
    assets = os.path.join(WORK, "paper_assets")
    sh([sys.executable, "scripts/aggregate.py", "--results", res, "--out", assets])
    out = os.path.join(WORK, "bundle")
    sh([sys.executable, "scripts/package_zenodo.py", "--results", res, "--logs", logs, "--assets", assets, "--out", out])
    n_models = build_models(res, os.path.join(out, "feddac-models-v1.0.0.zip"))
    readme = open(os.path.join(out, "README.md")).read()
    readme = readme.replace("trained global models (0 files)", f"trained global models ({n_models} files)")
    open(os.path.join(out, "README.md"), "w").write(readme)
    zips = sorted(glob.glob(os.path.join(out, "*.zip")))
    with open(os.path.join(out, "CHECKSUMS.sha256"), "w") as f:
        for p in zips:
            f.write(f"{sha256(p)}  {os.path.basename(p)}\n")
    meta = json.load(open(os.path.join(REPO, ".zenodo.json")))
    status = zenodo(zips + [os.path.join(out, "README.md"), os.path.join(out, "CHECKSUMS.sha256")], meta)
    status["models"] = n_models
    json.dump(status, open("/tmp/zenodo_status.json", "w"), indent=1)
    print(json.dumps(status, indent=1), flush=True)


if __name__ == "__main__":
    main()
