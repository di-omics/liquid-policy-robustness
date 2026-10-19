"""Step 3, second half: plot and tabulate the noise sweep.

Three summaries of the final distances, because a few bad episodes can dominate a
mean: the mean, the median (the typical episode) and the 90th percentile (the bad
tail). For a student, each training run gives one number per summary, computed over
the evaluation seeds. The line is the mean across its training runs, and the interval
is a 95% interval for that mean across training runs. It shows how much a result
depends on which training run you happened to get. It does not include the
variation from using only these 100 evaluation seeds, which every policy shares. The
teacher is a single deterministic run, so it gets no interval.

Run it (after evaluate.py):
    uv run plot.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sweep = json.loads(Path("results/noise_sweep.json").read_text())
levels = sweep["noise_levels"]
names = ["expert", "mlp", "lstm", "cfc", "ncp"]
labels = {"expert": "scripted teacher", "mlp": "MLP (no memory)", "lstm": "LSTM",
          "cfc": "CfC (liquid)", "ncp": "CfC + NCP wiring (liquid)"}
summaries = {"mean": np.mean, "median": np.median, "90th percentile": lambda d: np.quantile(d, 0.9)}


def summarize(name, summary):
    """Per noise level: (mean across training runs, interval half-width or None, runs)."""
    rows = [r for r in sweep["runs"] if r["policy"] == name]
    out = []
    for sigma in levels:
        scores = np.array([summaries[summary](r["final_distances"]) for r in rows if r["sigma"] == sigma])
        half = (stats.t.ppf(0.975, len(scores) - 1) * scores.std(ddof=1) / np.sqrt(len(scores))
                if len(scores) > 1 else None)
        out.append((scores.mean(), half, len(scores)))
    return out


present = [n for n in names if any(r["policy"] == n for r in sweep["runs"])]
for summary in summaries:
    print(f"\nFinal distance to target (m), {summary} over evaluation seeds; "
          "+/- is the 95% interval across training runs")
    print(f"{'policy':<34}" + "".join(f"{'sigma ' + format(s, 'g'):<17}" for s in levels))
    for name in present:
        per_level = summarize(name, summary)
        cells = [f"{m:.4f}" + (f"+/-{h:.4f}" if h is not None else "") for m, h, _ in per_level]
        tag = f"{labels[name]} (n={per_level[0][2]})"
        print(f"{tag:<34}" + "".join(f"{c:<17}" for c in cells))



def run_scores(name, sigma):
    """Mean final distance of each training run of one network at one noise level."""
    return np.array([np.mean(r["final_distances"]) for r in sweep["runs"]
                     if r["policy"] == name and r["sigma"] == sigma])


# Each network against the LSTM, the memory control. The training runs of two networks
# are independent, so this is a Welch interval for a difference of two means.
if "lstm" in present:
    others = [n for n in present if n not in ("expert", "lstm")]
    print("\nMean final distance minus the LSTM's (m), with its 95% interval; "
          "negative means closer to the target than the LSTM")
    print(f"{'sigma':<8}" + "".join(f"{name:<29}" for name in others))
    for sigma in levels:
        cells = []
        for name in others:
            a, b = run_scores(name, sigma), run_scores("lstm", sigma)
            d = a.mean() - b.mean()
            if len(a) < 2 or len(b) < 2:
                cells.append(f"{d:+.4f}")
                continue
            va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
            if va + vb == 0:
                half = 0.0
            else:
                dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
                half = stats.t.ppf(0.975, dof) * np.sqrt(va + vb)
            cells.append(f"{d:+.4f} [{d - half:+.4f}, {d + half:+.4f}]")
        print(f"{format(sigma, 'g'):<8}" + "".join(f"{c:<29}" for c in cells))

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
for ax, summary in zip(axes, ["mean", "median"]):
    for name in present:
        per_level = summarize(name, summary)
        m = np.array([p[0] for p in per_level])
        label = f"{labels[name]} (n={per_level[0][2]})"
        if name == "expert":
            ax.plot(levels, m, marker="o", color="black", linestyle="--", label=label)
            continue
        h = np.array([p[1] if p[1] is not None else 0.0 for p in per_level])
        line, = ax.plot(levels, m, marker="o", label=label)
        ax.fill_between(levels, m - h, m + h, alpha=0.2, color=line.get_color())
    ax.set_title(f"{summary} final distance")
    ax.set_xlabel("sensor noise (multiple of each reading's spread in the demonstrations)")
    ax.set_ylabel("final distance to target (m)")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.suptitle("Reacher-v5: imitation policies under sensor noise (bands: 95% across training runs)")
fig.tight_layout()
Path("media").mkdir(exist_ok=True)
fig.savefig("media/noise_sweep.png", dpi=150)
print("\nSaved media/noise_sweep.png")
