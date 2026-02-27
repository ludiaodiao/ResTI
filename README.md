# ReSTI (RepMTCN-style) – Middle-Gap Time-Series Reconstruction

This repository contains the code and example datasets to reproduce the **ReSTI / RepMTCN-style** baseline you provided:
a lightweight 1D-conv + segment-wise linear model for **middle-gap imputation / reconstruction** on multivariate time-series.

## What this repo includes

- **Model**: `ReSTI.Model` (single-channel version; choose different channels from a multivariate CSV)  
- **Re-parameterizable conv block**: `RepODCCB.Rep_1d_block` with `switch_to_deploy()` for inference-friendly conversion  
- **One-file training entry**: `train.py` (no CLI; edit `DATASET` and `CHANNELS`)  
- **Datasets (CSV)** under `./dataset/`

## Quickstart

### 1) Install

```bash
pip install -r requirements.txt
```

### 2) Train (edit-only, no CLI)

Open `train.py` and set:

- `DATASET = "ETTh1"` or `DATASET = "powerconsumption"`
- `CHANNELS = [1,2,3,4]` (ETTh1) or `CHANNELS = [4,5,6,7]` (powerconsumption), etc.

Then run:

```bash
python train.py
```

> Tip: if you set `DATA_USE_RATE` too small, the test split may be shorter than one window and you will get “No samples were constructed”. Increase `DATA_USE_RATE` or reduce `left/right/impu` lengths.


Outputs are saved to `./outputs_unified/...` (see `train.py` for the exact folder naming).

### 3) Deploy conversion

During training, the script will also export a deployable state dict via:

- `repmtcn_model_convert(...)` → calls `switch_to_deploy()` for each re-parameterizable block

The exported weights are placed under the same run folder as `deploy_net.pth`.

## Data format & channel indexing

`utils.get_dataloader()` loads a CSV from `./dataset/{DATASET}.csv` and selects one **column by integer index**:

- Column **0** is typically a datetime string.
- Numeric channels start at **1**.

So, for ETTh1:
- channel=1 → HUFL
- channel=7 → OT (target in many forecasting papers)

For powerconsumption:
- channel=1 → Temperature
- channel=6/7/8 → PowerConsumption_Zone1/2/3

## Task definition (middle-gap reconstruction)

For each sliding window:

- window length = `right_len + impu_len + left_len`
- input `x`:
  - **bidirectional**: concat(first `right_len`, last `left_len`)
  - **causal**: take first `right_len + left_len`
- label `y`: the middle segment of length `impu_len`

This matches the “predict the missing middle” visualization in your original utilities.

## Repository layout

```
.
├── ReSTI.py
├── RepODCCB.py
├── activation_function.py
├── utils.py
├── train.py
├── dataset/
│   ├── ETTh1.csv
│   └── powerconsumption.csv
└── scripts/
    └── sanity_check.py
```

## License

- **Code**: MIT License (see `LICENSE`)
- **Datasets**: see `DATA_LICENSE.md` (they come with their own upstream licenses)

## Citation

If you use this repository in academic work, please cite it (see `CITATION.cff`) and also cite the upstream datasets listed in `DATA_LICENSE.md`.
