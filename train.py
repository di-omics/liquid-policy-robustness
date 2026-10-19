"""Step 2: train policies to imitate the expert (behavior cloning).

Each network learns to output the expert's torques from the same clean
demonstrations, with the same optimizer, batch size and number of passes. The only
thing that differs between runs is the architecture and the random seed.

Run one:
    uv run train.py --arch cfc --seed 0
Run all four architectures for seeds 0 to 4:
    uv run train.py --arch all --seeds 5
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import ARCHS, Normalized, n_effective_params, n_params

parser = argparse.ArgumentParser()
parser.add_argument("--arch", default="all", choices=[*ARCHS, "all"])
parser.add_argument("--seed", type=int, default=None, help="train one seed")
parser.add_argument("--seeds", type=int, default=5, help="train seeds 0..N-1")
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--lr", type=float, default=3e-3)
parser.add_argument("--batch", type=int, default=32, help="episodes per batch")
args = parser.parse_args()

torch.set_num_threads(4)
demos = np.load("data/demos.npz")
train_obs, train_act = torch.tensor(demos["train_obs"]), torch.tensor(demos["train_act"])
val_obs, val_act = torch.tensor(demos["val_obs"]), torch.tensor(demos["val_act"])
# Standardize each feature with statistics of the training demonstrations only.
obs_mean = train_obs.reshape(-1, train_obs.shape[-1]).mean(0)
obs_std = train_obs.reshape(-1, train_obs.shape[-1]).std(0)


def train_one(arch, seed):
    torch.manual_seed(seed)
    model = Normalized(ARCHS[arch](), obs_mean, obs_std)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    gen = torch.Generator().manual_seed(seed)
    best, history, start = (float("inf"), None, -1), [], time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        for idx in torch.randperm(len(train_obs), generator=gen).split(args.batch):
            pred, _ = model(train_obs[idx])
            loss = torch.mean((pred - train_act[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val = torch.mean((model(val_obs)[0] - val_act) ** 2).item()
        history.append(val)
        # Keep the weights with the lowest validation loss (validation seeds, not eval).
        if val < best[0]:
            best = (val, {k: v.clone() for k, v in model.state_dict().items()}, epoch)
    seconds = time.perf_counter() - start
    model.load_state_dict(best[1])
    with torch.no_grad():
        max_torque = model(train_obs)[0].abs().max().item()

    Path("runs").mkdir(exist_ok=True)
    ckpt = {
        "arch": arch, "seed": seed, "n_params": n_params(model.policy),
        "n_effective_params": n_effective_params(model.policy), "max_torque": max_torque,
        "obs_mean": obs_mean, "obs_std": obs_std, "state_dict": best[1],
        "best_val_mse": best[0], "best_epoch": best[2], "val_history": history,
        "epochs": args.epochs, "lr": args.lr, "batch": args.batch, "train_seconds": seconds,
    }
    torch.save(ckpt, f"runs/{arch}_seed{seed}.pt")
    print(f"{arch:>4} seed {seed}: {ckpt['n_effective_params']} effective params, best validation "
          f"MSE {best[0]:.5f} at epoch {best[2]}, largest torque {max_torque:.3f}, {seconds:.0f} s")
    # The teacher's torques reach the +/-1 limit, so a student that cannot get close
    # to 1 is handicapped by its output layer, not by what it learned.
    if max_torque < 0.95:
        print(f"  WARNING: {arch} never outputs more than {max_torque:.3f}; the teacher uses up to 1.0")
    return {k: v for k, v in ckpt.items() if k not in ("state_dict", "obs_mean", "obs_std")}


archs = list(ARCHS) if args.arch == "all" else [args.arch]
seeds = [args.seed] if args.seed is not None else list(range(args.seeds))
summary = [train_one(a, s) for a in archs for s in seeds]
Path("results").mkdir(exist_ok=True)
Path(f"results/train_{args.arch}.json").write_text(json.dumps(summary, indent=2) + "\n")
