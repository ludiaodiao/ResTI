"""
Unified training script (no CLI) for ReSTI / RepMTCN-style model.

Goal:
- Replace the 3 messy per-dataset scripts with ONE clean script.
- You ONLY edit:
    1) DATASET
    2) CHANNELS
  (and optionally a few hyperparameters below)
- It runs a SINGLE configuration per channel (no huge sweeps), while keeping:
    - training loop
    - checkpoint save
    - deploy model convert (repmtcn_model_convert)
    - metrics logging (MSE/MAE)
    - config json dump

This script is designed to be safe against small signature differences in:
- utils.get_dataloader(...)
- ReSTI.Model(...)

by filtering kwargs using inspect.signature.
"""

import os
import json
import time
import random
import inspect
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ReSTI import Model, repmtcn_model_convert, generate_odd_numbers
from utils import get_dataloader


# =========================
# 1) USER CONFIG (EDIT HERE)
# =========================

# Choose dataset (supports aliases below)
DATASET: str = "powerconsumption"          # "ETTh1" | "powerconsumption"  (aliases: "PC")
CHANNELS: List[int] = [5]  # e.g. ETTh1: [1,2,3,4], AirQualityUCI: [1], powerconsumption: [4,5,6,7]

# One-key reproducibility
SEED: int = 2026

# Output folder
OUT_ROOT: str = "./outputs_unified"

# Training knobs (keep defaults to match your original scripts)
NUM_EPOCHS: int = 400
BATCH_SIZE: int = 3072
LR: float = 1e-3

# Evaluation strategy:
# - evaluate on test set during last K epochs only (like your AirQuality / powerconsumption scripts)
EVAL_LAST_K_EPOCHS: int = 5

# Frequency-domain loss (optional)
USE_FREQUENCY_DOMAIN_LOSS: bool = False
ALPHA_FREQ: float = 0.5   # only used if USE_FREQUENCY_DOMAIN_LOSS=True

# Model / data options (pick ONE config, no sweeping)
CLASS_CONV: str = "rep_conv"  # "normal" or "rep_conv"
CONV_WITH_ACTIVATION: bool = False
CONV_WITH_BIAS: bool = False
LINEAR_WITH_BIAS: bool = False
USE_BIDIRECTIONAL_FLOW: bool = True
IS_CONVOLUTION_FIRST: bool = False  # if USE_BIDIRECTIONAL_FLOW=False, will be forced to False

# How many repeated runs per channel (kept as an option, but default is 1)
REPEATS: int = 1

# If your get_dataloader supports data_use_rate, you can sub-sample for quick runs.
# Use None to NOT pass data_use_rate at all.
DATA_USE_RATE: Optional[float] = None


# =========================
# 2) DATASET DEFAULTS
# =========================
DATASET_ALIASES = {
    "PC": "powerconsumption",
    "PowerConsumption": "powerconsumption",
}

DATASET_DEFAULTS = {
    # From your ETTh1 script: w_list=[24] and channels=[1,2,3,4]
    "ETTh1": dict(period_len=24, data_use_rate=None),

    # From your powerconsumption script: w_list=[144], channels=[4,5,6,7], data_use_rate=.2
    "powerconsumption": dict(period_len=144, data_use_rate=0.2),
}


# =========================
# 3) CORE IMPLEMENTATION
# =========================

@dataclass
class RunConfig:
    dataset: str
    channel: int
    period_len: int

    # lengths (derived)
    left_len: int
    impu_len: int
    right_len: int

    # model hyperparams
    class_conv: str
    conv_with_activation_function: bool
    conv_with_bias: bool
    linear_with_bias: bool
    is_use_bidirectional_flow: bool
    is_convolution_first: bool

    # training
    seed: int
    num_epochs: int
    batch_size: int
    lr: float

    # loss
    use_frequency_domain_loss: bool
    alpha_freq: float

    # data
    data_use_rate: Optional[float]

    # branch sampling (filled per repeat)
    branch_num: Optional[int] = None
    aside_kernel_size_list: Optional[List[int]] = None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_dataset(name: str) -> str:
    name = name.strip()
    return DATASET_ALIASES.get(name, name)


def _safe_kwargs(func_or_cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs by signature to tolerate minor API differences."""
    try:
        sig = inspect.signature(func_or_cls)
    except (TypeError, ValueError):
        # best effort: return all kwargs
        return kwargs
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _make_out_dir(cfg: RunConfig, repeat_idx: int) -> str:
    # human-readable run name, similar to your "case_dir" but cleaner
    alpha_str = f"{cfg.alpha_freq}" if cfg.use_frequency_domain_loss else "0"
    run_name = (
        f"af_{cfg.conv_with_activation_function}"
        f"_conv_{cfg.class_conv}"
        f"_cb_{cfg.conv_with_bias}"
        f"_lb_{cfg.linear_with_bias}"
        f"_fl_{cfg.use_frequency_domain_loss}"
        f"_periodlen_{cfg.period_len}"
        f"_bidir_{cfg.is_use_bidirectional_flow}"
        f"_convfirst_{cfg.is_convolution_first}"
        f"_alpha_{alpha_str}"
        f"_repeat_{repeat_idx}"
    )

    out_dir = os.path.join(
        OUT_ROOT,
        cfg.dataset,
        f"period_{cfg.period_len}",
        f"channel_{cfg.channel}",
        run_name,
    )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _build_datasets(cfg: RunConfig):
    """Call get_dataloader with flexible signature."""
    kwargs = dict(
        DATASET=cfg.dataset,
        channel=cfg.channel,
        left_len=cfg.left_len,
        impu_len=cfg.impu_len,
        right_len=cfg.right_len,
        is_use_bidirectional_flow=cfg.is_use_bidirectional_flow,
    )

    # IMPORTANT: some codebases expect data_use_rate to be a float and will
    # crash on comparisons if you pass None.
    # So we ONLY pass it when it is explicitly set.
    if cfg.data_use_rate is not None:
        kwargs["data_use_rate"] = float(cfg.data_use_rate)
    kwargs = _safe_kwargs(get_dataloader, kwargs)
    return get_dataloader(**kwargs)


def _build_model(cfg: RunConfig) -> nn.Module:
    kwargs = dict(
        leftlen=cfg.left_len,
        pred_len=cfg.impu_len,
        rightlen=cfg.right_len,
        period_len=cfg.period_len,
        branch_num=cfg.branch_num,
        aside_kernel_size_list=cfg.aside_kernel_size_list,
        conv_with_activation_function=cfg.conv_with_activation_function,
        class_conv=cfg.class_conv,
        conv_with_bias=cfg.conv_with_bias,
        linear_with_bias=cfg.linear_with_bias,
        is_use_bidirectional_flow=cfg.is_use_bidirectional_flow,
        is_convolution_first=cfg.is_convolution_first,
    )
    kwargs = _safe_kwargs(Model, kwargs)
    return Model(**kwargs)


def _loss_fn(cfg: RunConfig, outputs: torch.Tensor, labels: torch.Tensor, mse: nn.Module) -> torch.Tensor:
    if not cfg.use_frequency_domain_loss:
        return mse(outputs, labels)

    # time-domain MSE
    loss_time = ((outputs - labels) ** 2).mean()
    # frequency-domain L1 on rFFT difference (same as your scripts)
    loss_freq = (torch.fft.rfft(outputs, dim=2) - torch.fft.rfft(labels, dim=2)).abs().mean()
    return cfg.alpha_freq * loss_freq + (1.0 - cfg.alpha_freq) * loss_time


@torch.no_grad()
def _eval_model(net: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    net.eval()
    mse = nn.MSELoss()
    total_mse = 0.0
    total_mae = 0.0
    n_batches = 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = net(inputs)
        total_mse += mse(outputs, labels).item()
        total_mae += torch.mean(torch.abs(outputs - labels)).item()
        n_batches += 1

    if n_batches == 0:
        return float("nan"), float("nan")
    return total_mse / n_batches, total_mae / n_batches


def train_one(cfg: RunConfig, repeat_idx: int) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # branch sampling per repeat (and save it for reproducibility)
    num, branch_list = generate_odd_numbers()
    cfg.branch_num = int(num)
    cfg.aside_kernel_size_list = list(branch_list)

    out_dir = _make_out_dir(cfg, repeat_idx)

    # skip if already done (config exists)
    config_path = os.path.join(out_dir, "config.json")
    if os.path.exists(config_path):
        print(f"[SKIP] already trained: {out_dir}")
        return

    # data
    train_dataset, test_dataset = _build_datasets(cfg)
    nw = 0
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=nw, pin_memory=True)

    # model
    net = _build_model(cfg).to(device)
    mse = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)

    # logs
    test_mse_hist: List[float] = []
    test_mae_hist: List[float] = []

    print(f"\n[RUN] {cfg.dataset} | channel={cfg.channel} | period_len={cfg.period_len} | repeat={repeat_idx}")
    print(f"[OUT] {out_dir}")
    print(f"[MODEL] class_conv={cfg.class_conv}, act={cfg.conv_with_activation_function}, "
          f"cb={cfg.conv_with_bias}, lb={cfg.linear_with_bias}, "
          f"bidir={cfg.is_use_bidirectional_flow}, convfirst={cfg.is_convolution_first}, "
          f"freq_loss={cfg.use_frequency_domain_loss}, alpha={cfg.alpha_freq if cfg.use_frequency_domain_loss else 0}")

    # train
    for epoch in range(cfg.num_epochs):
        net.train()
        running = 0.0
        n_batches = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = net(inputs)
            loss = _loss_fn(cfg, outputs, labels, mse)
            loss.backward()
            optimizer.step()

            running += loss.item()
            n_batches += 1

        mean_loss = running / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{cfg.num_epochs} | train_loss={mean_loss:.6f}")

        # evaluate only in last K epochs (to match your original style)
        if cfg.num_epochs - (epoch + 1) < EVAL_LAST_K_EPOCHS:
            tmse, tmae = _eval_model(net, test_loader, device)
            test_mse_hist.append(float(tmse))
            test_mae_hist.append(float(tmae))
            print(f"  Test MSE={tmse:.8f} | Test MAE={tmae:.8f}")

    # save checkpoint
    ckpt_path = os.path.join(out_dir, "net.pth")
    torch.save(net.state_dict(), ckpt_path)

    # deploy conversion (CPU)
    deploy_path = os.path.join(out_dir, "deploy_net.pth")
    repmtcn_model_convert(net.to("cpu"), save_path=deploy_path)

    # reload deploy keys (like your scripts)
    state_dict = torch.load(deploy_path, map_location="cpu")
    with open(os.path.join(out_dir, "deploy_keys.txt"), "w", encoding="utf-8") as f:
        for k in state_dict.keys():
            f.write(k + "\n")

    # save metrics
    np.savetxt(os.path.join(out_dir, "test_mse.txt"), np.array(test_mse_hist, dtype=np.float64), fmt="%.8e")
    np.savetxt(os.path.join(out_dir, "test_mae.txt"), np.array(test_mae_hist, dtype=np.float64), fmt="%.8e")
    np.save(os.path.join(out_dir, "test_mse.npy"), np.array(test_mse_hist, dtype=np.float64))
    np.save(os.path.join(out_dir, "test_mae.npy"), np.array(test_mae_hist, dtype=np.float64))

    # save config (full)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    print(f"[DONE] saved to {out_dir}\n")


def main() -> None:
    torch.backends.cudnn.benchmark = True

    # device info
    if torch.cuda.is_available():
        try:
            print(torch.cuda.get_device_properties(0).name)
        except Exception:
            print("[GPU] available")
    else:
        print("[GPU] not available, using CPU")

    dataset = _resolve_dataset(DATASET)
    if dataset not in DATASET_DEFAULTS:
        raise ValueError(
            f"Unknown DATASET='{DATASET}'. Supported: {list(DATASET_DEFAULTS.keys())} "
            f"(aliases: {list(DATASET_ALIASES.keys())})"
        )

    defaults = DATASET_DEFAULTS[dataset]
    period_len = int(defaults["period_len"])

    # choose data_use_rate:
    # - if user explicitly sets DATA_USE_RATE, use that
    # - else use dataset default
    data_use_rate = DATA_USE_RATE if DATA_USE_RATE is not None else defaults.get("data_use_rate", None)

    # handle convfirst logic
    is_conv_first = bool(IS_CONVOLUTION_FIRST)
    if not USE_BIDIRECTIONAL_FLOW:
        is_conv_first = False

    # derived lengths (same as your scripts)
    left_len = 2 * period_len
    impu_len = 2 * period_len
    right_len = 2 * period_len

    # run per channel
    for ch in CHANNELS:
        for r in range(REPEATS):
            seed = SEED + 1000 * ch + r
            _set_seed(seed)

            cfg = RunConfig(
                dataset=dataset,
                channel=int(ch),
                period_len=period_len,
                left_len=left_len,
                impu_len=impu_len,
                right_len=right_len,
                class_conv=str(CLASS_CONV),
                conv_with_activation_function=bool(CONV_WITH_ACTIVATION),
                conv_with_bias=bool(CONV_WITH_BIAS),
                linear_with_bias=bool(LINEAR_WITH_BIAS),
                is_use_bidirectional_flow=bool(USE_BIDIRECTIONAL_FLOW),
                is_convolution_first=is_conv_first,
                seed=int(seed),
                num_epochs=int(NUM_EPOCHS),
                batch_size=int(BATCH_SIZE),
                lr=float(LR),
                use_frequency_domain_loss=bool(USE_FREQUENCY_DOMAIN_LOSS),
                alpha_freq=float(ALPHA_FREQ),
                data_use_rate=data_use_rate,
            )

            train_one(cfg, repeat_idx=r)


if __name__ == "__main__":
    main()
