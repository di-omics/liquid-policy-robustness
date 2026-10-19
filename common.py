"""Shared pieces: the arm, the scripted expert, the policy networks, and sensor noise.

Every other script imports from here, so the seeds, the noise model and the network
sizes are defined once.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP

ENV_ID = "Reacher-v5"
EPISODE_STEPS = 50

# Three disjoint seed ranges. Demonstrations come from the training seeds, model
# selection uses the validation seeds, and every reported number comes from the
# evaluation seeds, which no training step ever sees.
TRAIN_SEEDS = list(range(0, 500))
VAL_SEEDS = list(range(500, 600))
EVAL_SEEDS = list(range(1000, 1100))

# Sensor noise, in units of each observation feature's standard deviation in the
# demonstrations. 0.1 means noise one tenth as large as that feature's normal spread.
NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]

# --- Scripted expert -------------------------------------------------------------
# Link lengths from reacher.xml: shoulder to elbow 0.10 m, elbow to fingertip 0.11 m.
L1, L2 = 0.10, 0.11
# Gains picked on training seeds 0-299. KB is the elbow-bend push described below.
KP, KD, KB = 150.0, 0.15, 10.0


def expert_action(obs):
    """Jacobian-transpose controller: push the fingertip toward the target, damp the joints.

    A pure Jacobian-transpose controller stalls when the arm is straight and the target
    lies along it, closer to the base: pulling the fingertip straight inward then asks
    for no joint torque at all. Every episode starts with the arm nearly straight, and
    with KB set to 0 the teacher ends more than 2 cm from the target in 14 of the 100
    evaluation episodes, mostly from this stall or slow progress near it. The KB
    term pushes the elbow to bend while the arm is near straight, scaled by the
    remaining distance so it fades at the target. With it, the teacher still ends more
    than 2 cm away in about 6 of 100.

    obs can be one observation (10,) or a batch (n, 10).
    """
    obs = np.atleast_2d(obs)
    c0, c1, s0, s1 = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
    c01 = c0 * c1 - s0 * s1  # cos(shoulder + elbow)
    s01 = s0 * c1 + c0 * s1  # sin(shoulder + elbow)
    # How the fingertip (x, y) moves per unit of shoulder and elbow rotation.
    dx_dq0, dx_dq1 = -L1 * s0 - L2 * s01, -L2 * s01
    dy_dq0, dy_dq1 = L1 * c0 + L2 * c01, L2 * c01
    ex, ey = -obs[:, 8], -obs[:, 9]  # target minus fingertip
    tau0 = KP * (dx_dq0 * ex + dy_dq0 * ey) - KD * obs[:, 6]
    tau1 = KP * (dx_dq1 * ex + dy_dq1 * ey) - KD * obs[:, 7]
    tau1 += KB * np.hypot(ex, ey) * c1  # elbow-bend push, largest when the arm is straight
    return np.clip(np.stack([tau0, tau1], axis=1), -1.0, 1.0).astype(np.float32)


# --- Sensor noise ----------------------------------------------------------------
class SensorNoise(gym.ObservationWrapper):
    """Adds Gaussian noise to every observation the policy receives.

    The noise generator is seeded per episode, so every policy evaluated on the same
    seed and noise level receives exactly the same noise draws.
    """

    def __init__(self, env, sigma, feature_std):
        super().__init__(env)
        self.sigma = float(sigma)
        self.feature_std = np.asarray(feature_std, dtype=np.float64)
        self.rng = np.random.default_rng(0)

    def reset(self, *, seed=None, options=None):
        self.rng = np.random.default_rng([seed if seed is not None else 0, 7])
        return super().reset(seed=seed, options=options)

    def observation(self, obs):
        # TODO: one line. Return obs plus Gaussian noise whose size for feature i is
        # self.sigma * self.feature_std[i]. Draw the noise from self.rng
        # (self.rng.standard_normal(obs.shape) gives one standard normal number per
        # feature), never from np.random, or the policies stop facing identical noise.
        raise NotImplementedError("TODO: add the sensor noise in common.py, SensorNoise.observation")


# --- Policies ----------------------------------------------------------------------
# All four map a sequence of observations (batch, time, 10) to torques (batch, time, 2)
# and carry a hidden state from one step to the next. The MLP has no memory: it sees
# one observation at a time and ignores the state. Sizes are chosen so each network
# has close to 5,000 parameters that can change its output (see n_effective_params).
# Every network ends in a linear layer and a tanh, so each can output any torque in
# (-1, 1).
OBS_DIM, ACT_DIM = 10, 2


class MLPPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, ACT_DIM)
        )

    def forward(self, x, state=None):
        return torch.tanh(self.net(x)), None


class LSTMPolicy(nn.Module):
    def __init__(self, hidden=29):
        super().__init__()
        self.rnn = nn.LSTM(OBS_DIM, hidden, batch_first=True)
        self.head = nn.Linear(hidden, ACT_DIM)

    def forward(self, x, state=None):
        y, state = self.rnn(x, state)
        return torch.tanh(self.head(y)), state


class CfCPolicy(nn.Module):
    """Closed-form continuous-time (CfC) liquid network, fully connected.

    The ncps CfC passes the input and state through one 40-unit layer (the backbone)
    before the liquid neurons, and proj_size adds a linear readout to 2 torques.
    """

    def __init__(self, units=22, backbone_units=40):
        super().__init__()
        self.rnn = CfC(OBS_DIM, units, proj_size=ACT_DIM, batch_first=True,
                       backbone_units=backbone_units, backbone_layers=1)

    def forward(self, x, state=None):
        y, state = self.rnn(x, state)
        return torch.tanh(y), state


class NCPPolicy(nn.Module):
    """CfC neurons with sparse neural-circuit-policy wiring, the design used in the
    liquid-network drone-navigation work.

    The wiring zeroes about half of the connections from one layer of neurons to the
    next (12 of the 30 into the 2 motor neurons); connections within a layer stay fully
    connected, and there is no backbone layer. Its 2 motor neurons are squashed into (-1, 1) already, so without the linear
    readout (proj_size) the outer tanh would cap every torque at tanh(1) = 0.76. All
    training runs share one wiring (seed 0).
    """

    def __init__(self, units=40):
        super().__init__()
        self.rnn = CfC(OBS_DIM, AutoNCP(units, ACT_DIM, seed=0), proj_size=ACT_DIM, batch_first=True)

    def forward(self, x, state=None):
        y, state = self.rnn(x, state)
        return torch.tanh(y), state


ARCHS = {"mlp": MLPPolicy, "lstm": LSTMPolicy, "cfc": CfCPolicy, "ncp": NCPPolicy}


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def n_effective_params(model):
    """Trainable parameters minus the weights a sparse wiring holds at zero.

    ncps multiplies each wired layer's two input weight matrices by a 0/1 mask, so a
    masked weight gets zero gradient and never affects the output.
    """
    total = n_params(model)
    for module in model.modules():
        mask = getattr(module, "sparsity_mask", None)
        if mask is not None:
            total -= int((mask == 0).sum()) * (2 if hasattr(module, "ff2") else 1)
    return total


class Normalized(nn.Module):
    """Wraps a policy so it takes raw observations and standardizes them itself."""

    def __init__(self, policy, mean, std):
        super().__init__()
        self.policy = policy
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))

    def forward(self, x, state=None):
        return self.policy((x - self.mean) / self.std, state)


def load_policy(path):
    ckpt = torch.load(path, weights_only=True)
    model = Normalized(ARCHS[ckpt["arch"]](), ckpt["obs_mean"], ckpt["obs_std"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
