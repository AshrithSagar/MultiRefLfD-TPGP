"""
lfd/run.py \n
Trial run
"""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from shapely import LineString

from lfd.utils import append_progress_values
from lfd.utils.alignment import resample
from lfd.utils.frames import DemonstrationSet, Frame, GlobalFrame
from lfd.utils.gp import FrameRelevanceGP, LocalPolicyGP
from lfd.utils.lasa import load_data


def prepare_data(dset: DemonstrationSet, include_phi: bool = True):
    """
    Turn a DemonstrationSet of shape (N demos, L points, 3 coords)
    into X and Y matrices for GP training.

    :return X: [xi, phi] at time t
    :return Y: Delta xi = xi(t+1) − xi(t), Delta phi = phi(t+1) − phi(t)
    """
    arr = np.stack([np.array(d.coords) for d in dset])  # Shape (N, L, 3)
    X = arr[:, :-1, : 3 if include_phi else 2]  # Shape (N, L−1, 2 or 3)
    Y = arr[:, 1:, : 3 if include_phi else 2] - X  # Shape (N, L−1, 2 or 3)

    # Flatten demos & timesteps
    X_flat = X.reshape(-1, 3)  # Shape (N*(L−1), 2 or 3)
    Y_flat = Y.reshape(-1, 3)  # Shape (N*(L−1), 2 or 3)
    return X_flat, Y_flat


data, x, xd = load_data("s")
dset = append_progress_values([LineString(traj) for traj in data])
f1 = Frame(index=1, rotation=10, translation=(5, 5))
al_dset = resample(dset, frames=[f1])

frames = [GlobalFrame, f1]
local_policies: Dict[int, LocalPolicyGP] = {}

for m, frame in enumerate(frames):
    # Transform the aligned global demos into frame m
    dset_m = frame.transform(al_dset)
    X_m, Y_m = prepare_data(dset_m, include_phi=False)

    # Pick M inducing points randomly from X_m
    idx = np.random.choice(len(X_m), size=50, replace=False)
    Xu = X_m[idx]

    print(f"Frame {m}: {len(X_m)} points, {len(Xu)} inducing points")
    gp_m = LocalPolicyGP(X_m, Y_m, Xu, noise=1e-2, lr=5e-3)
    gp_m.train(num_steps=1000, log_every=200)
    local_policies[m] = gp_m

# Re‑prepare the global (aligned) dataset into inputs X_glob and true deltas Delta_glob
# X_glob: (N, 3) : [x, y, phi],  Y_glob: (N, 3) : Delta [x,y,phi]
X_glob, Y_glob = prepare_data(al_dset, include_phi=True)

# Split into features
phi = X_glob[:, 2]  # Shape (N,)
deltas = Y_glob  # Shape (N, D_out)

# Query each local GP on the same X_glob to get local means & vars
all_means = []
all_vars = []
for m, gp_m in sorted(local_policies.items()):
    mu_m, var_m = gp_m.predict(X_glob)  # Each (N, D_out)
    all_means.append(mu_m)
    all_vars.append(var_m)

# Stack into shape (N, num_frames, D_out)
local_means = np.stack(all_means, axis=1)
local_vars = np.stack(all_vars, axis=1)

# Choose M inducing‐points along phi in [0,1], evenly spaced
Xu_phi = np.linspace(0, 1, 20)
num_frames = len(local_policies)

frgp = FrameRelevanceGP(
    phi=phi,
    num_frames=num_frames,
    Xu=Xu_phi,
    noise=1e-2,
    lr=1e-2,
)

# Self‑supervised train relevance GP
frgp.train(
    phi=phi,
    local_means=local_means,
    local_vars=local_vars,
    deltas=deltas,
    num_steps=2000,
    log_every=200,
)

# Inspect learned relevance alpha(phi)
phi_test = np.linspace(0, 1, 100)
alpha = frgp.predict(phi_test)  # Shape (100, num_frames)

# Plot of each column of alpha vs. phi to see which frame is active when.
plt.figure(figsize=(10, 5))
for i in range(num_frames):
    plt.plot(phi_test, alpha[:, i], label=f"Frame {i}")
plt.xlabel("Progress $\\varphi$")
plt.ylabel("Relevance $\\alpha(\\varphi)$")
plt.title("Frame Relevance vs. Progress")
plt.legend()
plt.grid()
plt.show()
