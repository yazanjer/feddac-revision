"""Dataset loading and federated client partitioning.

All datasets are materialised once as uint8 tensors (N, C, H, W) plus int64 labels and are kept
on the training device.  Normalisation and (optional) augmentation happen on the fly on the GPU,
which removes the DataLoader bottleneck that dominates small-model FL simulation.

Partitioning follows the manuscript (Section 5.1):
  * homogeneous clients      -> balanced (IID) class distribution
  * heterogeneous clients    -> class proportions ~ Dirichlet(alpha)
  * extremely heterogeneous  -> samples from only k_ext randomly chosen classes
                                (k_ext = 2 for 10-class datasets; 20% of classes in general)
Samples are drawn without replacement; if a class pool is exhausted a relaxed-reuse policy
samples the remainder with replacement from that class (the number of reused samples is logged).
"""
from __future__ import annotations

import os
import tarfile
import zipfile
from dataclasses import dataclass, field

import numpy as np
import torch

DATA_ROOT = os.environ.get("FEDDAC_DATA", os.path.expanduser("~/.feddac_data"))

# (mean, std) per channel, used for on-the-fly normalisation
STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "tinyimagenet": ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
}
NUM_CLASSES = {"mnist": 10, "cifar10": 10, "cifar100": 100, "tinyimagenet": 200, "synthetic": 10}


@dataclass
class TensorDataset:
    name: str
    x_train: torch.Tensor  # uint8 (N, C, H, W)
    y_train: torch.Tensor  # int64 (N,)
    x_test: torch.Tensor
    y_test: torch.Tensor
    num_classes: int
    mean: torch.Tensor = field(default=None)
    std: torch.Tensor = field(default=None)
    augment: bool = False

    def to(self, device):
        for k in ("x_train", "y_train", "x_test", "y_test"):
            setattr(self, k, getattr(self, k).to(device))
        c = self.x_train.shape[1]
        m, s = STATS.get(self.name, ((0.5,) * c, (0.5,) * c))
        self.mean = torch.tensor(m, device=device).view(1, -1, 1, 1)
        self.std = torch.tensor(s, device=device).view(1, -1, 1, 1)
        return self

    def prep(self, x_uint8: torch.Tensor, train: bool = False) -> torch.Tensor:
        x = x_uint8.float().div_(255.0)
        if train and self.augment:
            x = _augment(x)
        return (x - self.mean) / self.std


def _augment(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Random crop (with zero padding) + horizontal flip, batched on device."""
    n, c, h, w = x.shape
    xp = torch.nn.functional.pad(x, (pad, pad, pad, pad))
    ix = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    iy = torch.randint(0, 2 * pad + 1, (n,), device=x.device)
    rows = (iy.view(n, 1) + torch.arange(h, device=x.device).view(1, h))  # n,h
    cols = (ix.view(n, 1) + torch.arange(w, device=x.device).view(1, w))  # n,w
    xp = xp[torch.arange(n, device=x.device).view(n, 1, 1), :, rows.view(n, h, 1), cols.view(n, 1, w)]
    xp = xp.permute(0, 3, 1, 2)  # n,c,h,w
    flip = torch.rand(n, device=x.device) < 0.5
    xp[flip] = xp[flip].flip(-1)
    return xp.contiguous()


# --------------------------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------------------------
def _cache(name):
    return os.path.join(DATA_ROOT, f"{name}.npz")


def _from_cache(name):
    p = _cache(name)
    if os.path.exists(p):
        d = np.load(p)
        return d["xtr"], d["ytr"], d["xte"], d["yte"]
    return None


def _save_cache(name, xtr, ytr, xte, yte):
    os.makedirs(DATA_ROOT, exist_ok=True)
    np.savez(_cache(name), xtr=xtr, ytr=ytr, xte=xte, yte=yte)


def _torchvision(name):
    import torchvision

    root = os.path.join(DATA_ROOT, "raw")
    cls = {"mnist": torchvision.datasets.MNIST, "cifar10": torchvision.datasets.CIFAR10,
           "cifar100": torchvision.datasets.CIFAR100}[name]
    tr = cls(root, train=True, download=True)
    te = cls(root, train=False, download=True)

    def arr(ds):
        x = np.asarray(ds.data)
        if x.ndim == 3:  # mnist (N,H,W)
            x = x[:, None]
        else:  # cifar (N,H,W,C)
            x = x.transpose(0, 3, 1, 2)
        return np.ascontiguousarray(x, dtype=np.uint8), np.asarray(ds.targets, dtype=np.int64)

    return (*arr(tr), *arr(te))


def _tinyimagenet():
    """Tiny-ImageNet-200 (train 100k, official val 10k used as test)."""
    from PIL import Image
    import urllib.request

    root = os.path.join(DATA_ROOT, "raw")
    os.makedirs(root, exist_ok=True)
    d = os.path.join(root, "tiny-imagenet-200")
    if not os.path.isdir(d):
        zp = os.path.join(root, "tiny-imagenet-200.zip")
        if not os.path.exists(zp):
            try:
                urllib.request.urlretrieve("http://cs231n.stanford.edu/tiny-imagenet-200.zip", zp)
            except Exception:
                return _tinyimagenet_hf()
        zipfile.ZipFile(zp).extractall(root)
    wnids = [l.strip() for l in open(os.path.join(d, "wnids.txt"))]
    wid = {w: i for i, w in enumerate(sorted(wnids))}
    xs, ys = [], []
    for w in sorted(wnids):
        imgdir = os.path.join(d, "train", w, "images")
        for f in sorted(os.listdir(imgdir)):
            xs.append(np.asarray(Image.open(os.path.join(imgdir, f)).convert("RGB")))
            ys.append(wid[w])
    xtr = np.stack(xs).transpose(0, 3, 1, 2)
    ytr = np.asarray(ys, dtype=np.int64)
    xs, ys = [], []
    for line in open(os.path.join(d, "val", "val_annotations.txt")):
        f, w = line.split("\t")[:2]
        xs.append(np.asarray(Image.open(os.path.join(d, "val", "images", f)).convert("RGB")))
        ys.append(wid[w])
    xte = np.stack(xs).transpose(0, 3, 1, 2)
    yte = np.asarray(ys, dtype=np.int64)
    return np.ascontiguousarray(xtr), ytr, np.ascontiguousarray(xte), yte


def _tinyimagenet_hf():
    import io
    from PIL import Image
    from datasets import load_dataset  # fallback mirror

    ds = load_dataset("zh-plus/tiny-imagenet")

    def arr(split):
        xs = [np.asarray(im.convert("RGB")) for im in split["image"]]
        return np.stack(xs).transpose(0, 3, 1, 2), np.asarray(split["label"], dtype=np.int64)

    xtr, ytr = arr(ds["train"])
    xte, yte = arr(ds["valid"])
    return np.ascontiguousarray(xtr), ytr, np.ascontiguousarray(xte), yte


HF = {  # (repo, config, image column, label column, test split)
    "mnist": ("ylecun/mnist", "mnist", "image", "label", "test"),
    "cifar10": ("uoft-cs/cifar10", "plain_text", "img", "label", "test"),
    "cifar100": ("uoft-cs/cifar100", "cifar100", "img", "fine_label", "test"),
    "tinyimagenet": ("zh-plus/tiny-imagenet", "default", "image", "label", "valid"),
}


def _hf(name):
    """Fast path: parquet files from the Hugging Face Hub CDN (same data as the original sources)."""
    import io
    import json as _json
    import urllib.request
    import pyarrow.parquet as pq
    from PIL import Image

    repo, cfg, icol, lcol, test_split = HF[name]
    root = os.path.join(DATA_ROOT, "hf", name)
    os.makedirs(root, exist_ok=True)

    def split(sp):
        api = f"https://huggingface.co/api/datasets/{repo}/parquet/{cfg}/{sp}"
        urls = _json.loads(urllib.request.urlopen(api, timeout=60).read())
        xs, ys = [], []
        for k, u in enumerate(urls):
            f = os.path.join(root, f"{sp}-{k}.parquet")
            if not os.path.exists(f):
                urllib.request.urlretrieve(u, f + ".part")
                os.replace(f + ".part", f)
            t = pq.read_table(f).to_pydict()
            for im, y in zip(t[icol], t[lcol]):
                img = Image.open(io.BytesIO(im["bytes"]))
                img = img.convert("L") if name == "mnist" else img.convert("RGB")
                a = np.asarray(img, dtype=np.uint8)
                xs.append(a[None] if a.ndim == 2 else a.transpose(2, 0, 1))
                ys.append(int(y))
        return np.ascontiguousarray(np.stack(xs)), np.asarray(ys, dtype=np.int64)

    xtr, ytr = split("train")
    xte, yte = split(test_split)
    return xtr, ytr, xte, yte


def _download(name):
    try:
        return _hf(name)
    except Exception as e:  # fall back to the original distribution sites
        print(f"[data] HF download failed for {name} ({e}); falling back", flush=True)
        return _tinyimagenet() if name == "tinyimagenet" else _torchvision(name)


def _synthetic(n_train=6000, n_test=1000, c=10, seed=0):
    """Small learnable synthetic dataset (for CI / smoke tests without internet)."""
    g = np.random.default_rng(seed)
    protos = g.integers(0, 255, size=(c, 1, 28, 28))
    def make(n):
        y = g.integers(0, c, size=n)
        x = np.clip(protos[y] + g.normal(0, 60, size=(n, 1, 28, 28)), 0, 255).astype(np.uint8)
        return x, y.astype(np.int64)
    return (*make(n_train), *make(n_test))


def load_dataset(name: str, device="cpu") -> TensorDataset:
    name = name.lower()
    if name == "synthetic":
        arrs = _synthetic()
    else:
        arrs = _from_cache(name)
        if arrs is None:
            import fcntl
            os.makedirs(DATA_ROOT, exist_ok=True)
            with open(os.path.join(DATA_ROOT, f"{name}.lock"), "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)  # one process downloads, the others wait
                arrs = _from_cache(name)
                if arrs is None:
                    arrs = _download(name)
                    _save_cache(name, *arrs)
                fcntl.flock(lk, fcntl.LOCK_UN)
    xtr, ytr, xte, yte = arrs
    ds = TensorDataset(name, torch.from_numpy(xtr), torch.from_numpy(ytr), torch.from_numpy(xte),
                       torch.from_numpy(yte), NUM_CLASSES[name],
                       augment=name in ("cifar100", "tinyimagenet"))
    return ds.to(device)


# --------------------------------------------------------------------------------------------
# partitioning
# --------------------------------------------------------------------------------------------
@dataclass
class Partition:
    client_indices: list          # list[np.ndarray] of train indices
    client_type: list             # 'homo' | 'hetero' | 'extreme'  (ground truth, for analysis)
    histograms: np.ndarray        # (N, C) label counts
    reused_samples: int = 0


def partition(labels: np.ndarray, num_clients: int, ratios=(0.1, 0.3, 0.6), alpha: float = 0.2,
              num_classes: int = 10, ext_classes: int | None = None, samples_per_client: int | None = None,
              seed: int = 0) -> Partition:
    """Split training indices into homogeneous / Dirichlet-heterogeneous / extreme clients."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n_homo = int(round(ratios[0] * num_clients))
    n_ext = int(round(ratios[2] * num_clients))
    n_het = num_clients - n_homo - n_ext
    assert n_het >= 0, "ratios must sum to <= 1"
    if ext_classes is None:
        ext_classes = max(2, int(round(0.2 * num_classes)))
    S = samples_per_client or len(labels) // num_clients

    pools = [list(rng.permutation(np.where(labels == c)[0])) for c in range(num_classes)]
    class_all = [np.where(labels == c)[0] for c in range(num_classes)]
    reused = 0

    def draw(counts):
        nonlocal reused
        out = []
        for c, k in enumerate(counts):
            k = int(k)
            if k == 0:
                continue
            take = min(k, len(pools[c]))
            out.extend(pools[c][:take])
            del pools[c][:take]
            if take < k:  # relaxed reuse
                out.extend(rng.choice(class_all[c], size=k - take, replace=True).tolist())
                reused += k - take
        return np.asarray(out, dtype=np.int64)

    types = ["homo"] * n_homo + ["hetero"] * n_het + ["extreme"] * n_ext
    plans = []
    for t in types:
        if t == "homo":
            base = np.full(num_classes, S // num_classes)
            base[: S - base.sum()] += 1
            counts = base
        elif t == "hetero":
            p = rng.dirichlet(np.full(num_classes, alpha))
            counts = rng.multinomial(S, p)
        else:
            cls = rng.choice(num_classes, size=ext_classes, replace=False)
            p = np.zeros(num_classes)
            p[cls] = rng.dirichlet(np.ones(ext_classes))
            counts = rng.multinomial(S, p)
        plans.append(counts)
    idx = [draw(c) for c in plans]
    # shuffle client order so that client id does not reveal its type
    order = rng.permutation(num_clients)
    idx = [idx[i] for i in order]
    types = [types[i] for i in order]
    hist = np.stack([np.bincount(labels[i], minlength=num_classes) for i in idx])
    return Partition(idx, types, hist, reused)
