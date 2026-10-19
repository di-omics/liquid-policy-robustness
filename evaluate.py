"""Step 3: test every trained policy, and the expert, under increasing sensor noise.

For each noise level, each policy runs one episode on each evaluation seed. The noise
draws depend only on the seed, so every policy faces identical noise. The score is the
true fingertip-to-target distance at the end of the episode, taken from the simulator,
never from the noisy observation.

Run it (after train.py):
    uv run evaluate.py
"""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from common import ENV_ID, EVAL_SEEDS, NOISE_LEVELS, SensorNoise, expert_action, load_policy

demos = np.load("data/demos.npz")
feature_std = demos["train_obs"].reshape(-1, demos["train_obs"].shape[-1]).std(0)


def check_sensor_noise():
    """Checks the SensorNoise TODO before spending a minute on the sweep."""
    def first_obs(seed, sigma=0.5):
        wrapped = SensorNoise(gym.make(ENV_ID), sigma, feature_std)
        noisy = wrapped.reset(seed=seed)[0]
        return noisy, wrapped.unwrapped._get_obs()

    a, _ = first_obs(EVAL_SEEDS[0])
    b, _ = first_obs(EVAL_SEEDS[0])
    assert np.array_equal(a, b), (
        "Same seed gave different noise: SensorNoise must draw from self.rng (TODO in common.py)")
    diffs = np.array([np.subtract(*first_obs(s)) for s in range(200)])
    ratio = diffs.std(0) / (0.5 * feature_std)
    assert np.all(np.abs(ratio - 1) < 0.2), (
        f"Noise size per feature is {np.round(ratio, 2)} times sigma * feature_std, expected 1 "
        "(TODO in common.py)")
    z = diffs / (0.5 * feature_std)
    assert np.all(np.abs(z.mean(0)) < 0.3), "Noise must average to zero (TODO in common.py)"
    corr = np.corrcoef(z.T)
    assert np.abs(corr - np.eye(len(feature_std))).max() < 0.5, (
        "Every feature got the same noise draw: draw one number per feature with "
        "self.rng.standard_normal(obs.shape) (TODO in common.py)")


check_sensor_noise()


def run_episodes(act_fn, sigma):
    """Runs one episode per evaluation seed, all in lockstep, and returns final and
    mean distances. act_fn maps a batch of observations and a state to actions."""
    envs = [SensorNoise(gym.make(ENV_ID), sigma, feature_std) for _ in EVAL_SEEDS]
    obs = np.stack([env.reset(seed=s)[0] for env, s in zip(envs, EVAL_SEEDS)])
    state, done, distances = None, False, []
    while not done:
        actions, state = act_fn(obs, state)
        steps = [env.step(a) for env, a in zip(envs, actions)]
        obs = np.stack([s[0] for s in steps])
        distances.append([-s[4]["reward_dist"] for s in steps])
        done = all(s[2] or s[3] for s in steps)
    for env in envs:
        env.close()
    distances = np.array(distances)  # (steps, episodes)
    return distances[-1].tolist(), distances.mean(0).tolist()


def policy_fn(model):
    def act(obs, state):
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(1)  # one time step
            a, state = model(x, state)
        return a[:, 0].numpy(), state
    return act


def expert_fn(obs, state):
    return expert_action(obs), None


torch.set_num_threads(4)
results = []
checkpoints = sorted(Path("runs").glob("*.pt"))
policies = [("expert", None, expert_fn)] + [
    (ckpt["arch"], ckpt["seed"], policy_fn(model))
    for model, ckpt in (load_policy(p) for p in checkpoints)
]
for name, seed, fn in policies:
    for sigma in NOISE_LEVELS:
        final, mean = run_episodes(fn, sigma)
        results.append({"policy": name, "train_seed": seed, "sigma": sigma,
                        "final_distances": final, "mean_distances": mean})
    tag = name if seed is None else f"{name} seed {seed}"
    print(f"evaluated {tag} at {len(NOISE_LEVELS)} noise levels")

Path("results").mkdir(exist_ok=True)
Path("results/noise_sweep.json").write_text(json.dumps(
    {"eval_seeds": [EVAL_SEEDS[0], EVAL_SEEDS[-1]], "noise_levels": NOISE_LEVELS,
     "feature_std": feature_std.tolist(), "runs": results}, indent=1) + "\n")
print("Saved results/noise_sweep.json")
