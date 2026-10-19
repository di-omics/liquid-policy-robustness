"""Step 1: record demonstrations from the scripted expert.

Runs the expert on the training and validation seeds, with clean observations, and
saves every observation and the expert's action at that observation. It also checks
the expert itself on the evaluation seeds, since a student cannot be judged without
knowing how good its teacher is. Last, it saves three expert episodes as
media/expert.gif.

Run it:
    uv run collect.py
"""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

from common import ENV_ID, EVAL_SEEDS, TRAIN_SEEDS, VAL_SEEDS, expert_action


def rollout(env, seed):
    obs, info = env.reset(seed=seed)
    observations, actions, done = [], [], False
    while not done:
        action = expert_action(obs)[0]
        observations.append(obs)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    # info["reward_dist"] is minus the true fingertip-to-target distance after the step.
    return np.array(observations), np.array(actions), -info["reward_dist"]


env = gym.make(ENV_ID)
data, final = {}, {}
for split, seeds in [("train", TRAIN_SEEDS), ("val", VAL_SEEDS), ("eval", EVAL_SEEDS)]:
    episodes = [rollout(env, s) for s in seeds]
    final[split] = [float(e[2]) for e in episodes]
    if split != "eval":  # evaluation seeds are never stored as training data
        data[f"{split}_obs"] = np.stack([e[0] for e in episodes]).astype(np.float32)
        data[f"{split}_act"] = np.stack([e[1] for e in episodes]).astype(np.float32)
env.close()

Path("data").mkdir(exist_ok=True)
np.savez_compressed("data/demos.npz", **data)
n_train, steps, _ = data["train_obs"].shape
print(f"Saved data/demos.npz: {n_train} training and {len(VAL_SEEDS)} validation episodes "
      f"of {steps} steps")

d = np.array(final["eval"])
print(f"Expert on {len(EVAL_SEEDS)} evaluation seeds, clean observations: final distance "
      f"mean {d.mean():.4f} m, median {np.median(d):.4f} m, 90th percentile {np.quantile(d, 0.9):.4f} m")
Path("results").mkdir(exist_ok=True)
Path("results/expert_clean.json").write_text(json.dumps({"final_distances": final["eval"]}, indent=2) + "\n")

# Three training-seed episodes of the expert, top-down, as a GIF.
TOP_DOWN = {"distance": 0.75, "elevation": -90.0, "azimuth": 90.0, "lookat": np.zeros(3)}
env = gym.make(ENV_ID, render_mode="rgb_array", default_camera_config=TOP_DOWN)
frames = []
for seed in TRAIN_SEEDS[:3]:
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        frames.append(Image.fromarray(env.render()).resize((240, 240)))
        obs, reward, terminated, truncated, info = env.step(expert_action(obs)[0])
        done = terminated or truncated
env.close()
Path("media").mkdir(exist_ok=True)
# 40 ms per frame is half real speed (one step is 20 ms).
frames[0].save("media/expert.gif", save_all=True, append_images=frames[1:], duration=40, loop=0)
print("Saved media/expert.gif")
