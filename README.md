# FedDAC — revision code

Code accompanying the revised manuscript **"FedDAC: Federated Learning with Distribution
Anonymization through Clustering on non-IID data"** (major revision of MEAS-D-26-14209).

This is a clean, single-codebase PyTorch re-implementation of FedDAC and every baseline and
ablation used in the revised paper. It replaces the exploratory notebooks of the original
submission, and every table and figure in the paper can be regenerated from it.

## What changed with respect to the original code

| Issue (reviewer) | Original code | This code |
|---|---|---|
| DP not formal (R2-C1) | `AdaptiveDPMechanism`: noise std = ε·0.1‖θ‖·σ/(σ+1e-6)/(1+‖θ‖), δ unused, no accounting | Record-level DP-SGD (Poisson sampling, per-example clipping, Gaussian noise) with σ calibrated by the RDP accountant for a target (ε, δ); histogram release charged to the same budget (`feddac/dp.py`) |
| Identical With/Without-DP numbers (R2-C3) | `use_clustering` flag ignored; DP noise ≈1e-2·‖θ‖/(1+‖θ‖) → negligible | DP and non-DP runs are separate jobs with their own seeds and result files. The accounting report (σ, q, steps, spent ε) is stored in every DP result |
| No attacks (R1-C1, R2-C2) | – | Label-distribution inference (server and peer observers) and membership inference (`feddac/attacks.py`) |
| Cluster-size leakage (R1-C2) | – | Plug-in MI analysis (`scripts/cluster_analysis.py`) + uniform-weight variant (`dac_uniform`) |
| Scale (R1-C3) | MNIST, CIFAR-10 | + CIFAR-100 and Tiny-ImageNet (ResNet-18-GN) |
| Novelty vs FedSe / FedSeq (R2-C4) | – | Both re-implemented (`fedse`, `fedseq`) |
| Ablations / variance / cost (R2-C5) | single run | 5 seeds, ablations `rand_seq`, `dac_par`, `dac_uniform`, `dac_samplew`, `dac_noisyhist`; communication + simulated latency logged |

## Layout
```
feddac/data.py         datasets as GPU tensors; homogeneous / Dirichlet / extreme partitioning
feddac/models.py       CNN_MNIST, CNN_CIFAR (as in the paper), ResNet18-GN
feddac/clustering.py   DAC (Alg. 1), random, FedSe label-balanced, FedSeq superclients, PP-CFL fuzzy c-means
feddac/algorithms.py   unified FL loop (grouping x intra-group schedule x inter-group weights)
feddac/dp.py           DP-SGD calibration / accounting (Opacus RDP accountant)
feddac/attacks.py      label-distribution inference, membership inference
scripts/run.py         one experiment -> results/<exp>/<run_id>.json (+ model .pt)
scripts/make_grid.py   the full experiment grid of the paper (configs/jobs.jsonl, 2 280 runs)
scripts/worker.py      shard runner used on RunPod (N concurrent runs per GPU, git sync)
scripts/aggregate.py   results -> LaTeX tables, CSV, PDF figures (paper_assets/)
scripts/cluster_analysis.py  training-free coverage and cluster-size side-channel analysis
runpod/                pod bootstrap
```

## Quick start
```bash
pip install -r requirements.txt
python -m pytest -q tests                    # CPU smoke test, synthetic data
python scripts/run.py --dataset mnist --method feddac --clients 20 --alpha 0.2 --ratios 0.1 0.3 0.6 --seed 0
python scripts/run.py --dataset cifar10 --method feddac --optimizer sgd --lr 0.02 --seed 0   # CIFAR-10 setting of the paper
python scripts/run.py --dataset cifar10 --method fedavg --dp_eps 4 --seed 0       # DP-SGD, eps=4, delta=1e-5
python scripts/run.py --dataset mnist --method feddac --ldi_rounds 1 25 50 --mia   # privacy attacks
```
Methods: `fedavg fedprox ppcfl fedse fedseq feddac rand_seq dac_par dac_uniform dac_samplew dac_noisyhist`.

## Reproducing the paper
```bash
python scripts/make_grid.py                     # configs/jobs.jsonl (main grid)
for g in scale mia lrsel dpprox attacksgd; do python scripts/make_grid.py $g; done   # additional grids
python scripts/worker.py --shard 0 --num_shards 1 --procs 4 --no_git   # or shard across GPUs/pods
python scripts/cluster_analysis.py
python scripts/aggregate.py --results results --out paper_assets
```
Additional grids: `scale` (CIFAR-100 / Tiny-ImageNet), `mia` (class-matched membership inference),
`lrsel` (CIFAR-10 SGD learning-rate selection), `dpprox` (DP-FedProx after adding the proximal term to the
DP-SGD path), `attacksgd` (CIFAR-10 attacks with the selected optimiser). Run each with
`scripts/worker.py --jobs configs/<grid>.jsonl`. `aggregate.py` documents which runs supersede which
(e.g. CIFAR-10 Adam runs are kept only for the optimiser sensitivity table).

On RunPod we used 8-16 pods (1 GPU each), `runpod/start_cmd.txt` as the container start command and
env `POD_INDEX`, `NUM_SHARDS`, `PROCS`, `DEPLOY_KEY_B64`.

## Default hyper-parameters (Table 1 of the paper)
Optimiser: MNIST Adam lr 1e-3; CIFAR-10 plain SGD lr 0.02 (selected by the `lrsel` grid; with Adam the
sequential methods drift, see the optimiser table); CIFAR-100 / Tiny-ImageNet SGD lr 0.05; all DP runs Adam
lr 1e-3. 1 local epoch, fresh optimiser state per local update, full participation; batch 10
(MNIST, CIFAR-10) or 32 (CIFAR-100, Tiny-ImageNet); 50 / 100 / 100 / 60 rounds; FedProx μ = 0.01;
PP-CFL randomized-response ε = 0.8; L = round(√(N/2)) groups for DAC, FedSe, random grouping and PP-CFL;
FedSeq K_S,max = 11, min 800 samples. Sequential order within a group is re-drawn every round. DP: lot size 64,
clip 1.0, δ = 1e-5, ε_hist = 0.5 for methods that use label statistics.

## License
MIT (code). Trained models and result files released on Zenodo under CC-BY-4.0.
