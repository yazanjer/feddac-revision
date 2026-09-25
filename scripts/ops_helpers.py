"""Operational helpers used from the Composio workbench to monitor/launch RunPod workers (not needed to reproduce results)."""
import json, base64, os, re, io, time, zipfile, contextlib, urllib.request
DK = globals().get("DK", "")  # deploy key is injected at launch time, never stored in the repo
START = "mkdir -p ~/.ssh && echo \"$DEPLOY_KEY_B64\" | base64 -d > ~/.ssh/id_ed25519 && chmod 600 ~/.ssh/id_ed25519 && ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null && (git clone -q git@github.com:yazanjer/feddac-revision.git /workspace/repo || (cd /workspace/repo && git fetch -q)) && bash /workspace/repo/runpod/bootstrap.sh"
REPO = {"owner": "yazanjer", "repo": "feddac-revision"}
MUT = "mutation($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $input) { id name desiredStatus costPerHr machine { gpuDisplayName } } }"
GP = [("NVIDIA GeForce RTX 4090","COMMUNITY"),("NVIDIA GeForce RTX 3090","COMMUNITY"),("NVIDIA RTX A5000","COMMUNITY"),("NVIDIA RTX A4500","COMMUNITY"),("NVIDIA GeForce RTX 3090","SECURE"),("NVIDIA RTX A5000","SECURE"),("NVIDIA GeForce RTX 4090","SECURE")]
def q(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)
def branches():
    r, e = q(run_composio_tool, "GITHUB_LIST_BRANCHES", dict(REPO, per_page=100))
    return [b["name"] for b in r["data"]["branches"]]
def gh_file(path, ref):
    r, e = q(run_composio_tool, "GITHUB_GET_REPOSITORY_CONTENT", dict(REPO, path=path, ref=ref))
    if e: raise RuntimeError(str(e)[:200])
    d = r["data"]; d = d.get("content", d) if isinstance(d.get("content"), dict) else d
    return base64.b64decode(d["content"].replace("\n", "")).decode()
def get_main(path): return gh_file(path, "main")
def commit(files, msg):
    r, e = q(run_composio_tool, "GITHUB_COMMIT_MULTIPLE_FILES", dict(REPO, branch="main", message=msg, upserts=[{"path": p, "content": c} for p, c in files.items()]))
    return e or [(x["path"], x["blob_sha"]) for x in r["data"]["path_status"]]
def gql(query, variables=None):
    body = {"query": query}
    if variables: body["variables"] = variables
    r, e = q(proxy_execute, "POST", "https://api.runpod.io/graphql", "runpod", body=body)
    return r if not e else {"error": str(e)}
def pods():
    r = gql("query { myself { pods { id name desiredStatus costPerHr runtime { uptimeInSeconds gpus { gpuUtilPercent } } } } }")
    return [p for p in r["data"]["myself"]["pods"] if p["name"].startswith("feddac")]
def terminate(pid): return gql('mutation { podTerminate(input: {podId: "%s"}) }' % pid)
def deploy2(name, env, gpu, cloud, vcpu=6, mem=24):
    envl = [{"key": k, "value": v} for k, v in dict(env, DEPLOY_KEY_B64=DK).items()]
    inp = {"cloudType": cloud, "gpuCount": 1, "volumeInGb": 0, "containerDiskInGb": 50, "minVcpuCount": vcpu, "minMemoryInGb": mem,
           "gpuTypeId": gpu, "name": name, "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
           "dockerArgs": "bash -c '" + START + "'", "ports": "22/tcp", "env": envl}
    r = gql(MUT, {"input": inp})
    return (r.get("data") or {}).get("podFindAndDeployOnDemand")
def launch(name, env):
    for g, c in GP:
        p = deploy2(name, env, g, c)
        if p: return p
def last_commit(b):
    r, e = q(run_composio_tool, "GITHUB_LIST_COMMITS", dict(REPO, sha=b, per_page=1))
    c = r["data"]["commits"][0]["commit"]; return c["author"]["date"][11:16], c["message"][:40]
def progress():
    out = {}
    for b in branches():
        if not (b.startswith("results/pod-") or b.startswith("results/scale-") or b.startswith("results/mia-")) or "pilot" in b or "diag" in b: continue
        k = re.sub(r"\D", "", b.split("-")[-1])
        try:
            t = gh_file(f"logs/worker-{k}.txt", b)
            m = re.findall(r"\[(\d+)/(\d+)\] rc=(\d+)", t)
            out[b] = (m[-1][0] + "/" + m[-1][1] if m else "0", sum(1 for x in m if x[2] != "0"), "DONE" if "ALL DONE" in t else "", last_commit(b)[0])
        except Exception as ex:
            out[b] = ("no log", last_commit(b))
    return out
def branch_zip_url(ref):
    r, e = q(run_composio_tool, "GITHUB_DOWNLOAD_A_REPOSITORY_ARCHIVE_ZIP", dict(REPO, ref=ref))
    d = r["data"]; return (d.get("headers") or d["data"]["headers"])["location"]
