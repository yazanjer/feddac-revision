"""CPU smoke tests: python -m pytest -q tests  (uses a synthetic dataset, no download)."""
import os, subprocess, sys, json, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
from feddac import clustering as CL
from feddac.data import partition


def test_partition_and_dac():
    y = np.repeat(np.arange(10), 1000)
    p = partition(y, 20, (0.1, 0.3, 0.6), 0.2, 10, seed=0)
    assert len(p.client_indices) == 20 and p.histograms.sum() > 0
    ext, het, homo = CL.classify_clients(p.histograms)
    assert len(ext) >= 10
    g = CL.dac_cluster(p.histograms, CL.num_clusters(20))
    assert sorted(c for x in g for c in x) == list(range(20))


def test_runs(tmp_path):
    for m in ["fedavg", "feddac", "ppcfl", "fedse", "fedseq"]:
        subprocess.check_call([sys.executable, os.path.join(ROOT, "scripts/run.py"), "--dataset", "synthetic",
                               "--method", m, "--clients", "10", "--rounds", "2", "--out", str(tmp_path)])
    assert len(glob.glob(str(tmp_path / "default" / "*.json"))) == 5
