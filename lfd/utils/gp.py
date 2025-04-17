"""
lfd/utils/gp.py \n
Gaussian processes utils
"""

from typing import List, Tuple, cast

import numpy as np
import pyro.contrib.gp as gp
import pyro.contrib.gp.kernels as kernels
import torch
from numpy.typing import NDArray
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam, PyroOptim
from torch import Tensor


# TODO: Verify
class LocalPolicyGP:
    """
    Local policy modeled as independent Gaussian Processes for each output dimension.
    Uses a Matérn 5/2 kernel and sparse variational approximation.
    """

    def __init__(
        self,
        X: NDArray,
        Y: NDArray,
        Xu: NDArray,
        noise: float = 1e-2,
        lr: float = 1e-2,
    ) -> None:
        """
        :param X: Training inputs, shape (N, D_in)
        :param Y: Training outputs, shape (N, D_out)
        :param Xu: Inducing inputs, shape (M, D_in)
        :param noise: Observation noise
        :param lr: Learning rate for optimizer
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.output_dim = self.Y.shape[1]

        # Build one sparse GP per output dimension
        self.models: List[gp.models.VariationalSparseGP] = []
        for i in range(self.output_dim):
            kernel = kernels.Matern52(input_dim=self.X.shape[1])
            gpr = gp.models.VariationalSparseGP(
                self.X,
                self.Y[:, i],
                kernel,
                Xu=torch.tensor(Xu, dtype=torch.float32),
                noise=torch.tensor(noise),
            )
            self.models.append(gpr)

        # Set up optimizer and SVI
        self.optim: PyroOptim = Adam({"lr": lr})
        self.elbo = Trace_ELBO()
        self.svis = [
            SVI(m.model, m.guide, self.optim, loss=self.elbo) for m in self.models
        ]

    def train(self, num_steps: int = 1000, log_every: int = 100) -> None:
        """
        Train each local GP by maximizing the ELBO.
        """
        for step in range(num_steps):
            losses = [svi.step() for svi in self.svis]
            if log_every and step % log_every == 0:
                total = sum(losses)
                print(f"[LocalPolicyGP] Step {step} ELBO loss: {total:.4f}")

    def predict(self, X_new: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Predict mean and variance for new inputs.
        :param X_new: Inputs, shape (K, D_in)
        :return: mean (K, D_out), var (K, D_out)
        """
        mu: Tensor
        var: Tensor

        X_new_t = torch.tensor(X_new, dtype=torch.float32)
        means, vars_ = [], []
        for m in self.models:
            mu, var = m(X_new_t, full_cov=False, noiseless=False)
            means.append(mu.detach().numpy())
            vars_.append(var.detach().numpy())
        return np.stack(means, axis=1), np.stack(vars_, axis=1)


# TODO: Verify
class FrameRelevanceGP:
    """
    Frame relevance modeled as independent Gaussian Processes over the progress scalar phi.
    Outputs softmax-normalized relevance weights for each frame.
    """

    def __init__(
        self,
        phi: NDArray,
        num_frames: int,
        Xu: NDArray,
        noise: float = 1e-2,
        lr: float = 1e-2,
    ) -> None:
        """
        :param phi: Progress values, shape (N,)
        :param num_frames: Number of frames (i.e., GP outputs)
        :param Xu: Inducing inputs for the GP, shape (M, 1)
        :param noise: Observation noise
        :param lr: Learning rate
        """
        # Reshape progress to column vector
        self.phi = torch.tensor(phi.reshape(-1, 1), dtype=torch.float32)
        self.num_frames = num_frames

        # Build one sparse GP per frame
        self.models: List[gp.models.VariationalSparseGP] = []
        for _ in range(num_frames):
            kernel = kernels.Matern52(input_dim=1)
            gpr = gp.models.VariationalSparseGP(
                self.phi,
                torch.zeros(self.phi.shape[0]),  # placeholder; labels will be implicit
                kernel,
                Xu=torch.tensor(Xu.reshape(-1, 1), dtype=torch.float32),
                noise=torch.tensor(noise),
            )
            self.models.append(gpr)

        # Set up optimizer and SVI
        self.optim: PyroOptim = Adam({"lr": lr})
        self.elbo = Trace_ELBO()
        self.svis = [
            SVI(m.model, m.guide, self.optim, loss=self.elbo) for m in self.models
        ]

    def train(self, *args, **kwargs) -> None:
        """
        Self-supervised training is performed externally by maximizing
        the joint likelihood of weighted local policy predictions matching
        the demonstration deltas. Implement training loop elsewhere.
        """
        raise NotImplementedError(
            "FrameRelevanceGP self-supervised training to be implemented externally."
        )

    def predict(self, phi_new: NDArray) -> NDArray:
        """
        Compute relevance weights for new progress values.
        :param phi_new: New progress array, shape (K,)
        :return: relevance weights, shape (K, num_frames)
        """
        phi_t = torch.tensor(phi_new.reshape(-1, 1), dtype=torch.float32)
        mus: List[Tensor] = []
        for m in self.models:
            mu, _ = m(phi_t, full_cov=False, noiseless=True)
            mus.append(mu)
        M = torch.stack(mus, dim=1)  # (K, num_frames)
        alpha = torch.softmax(M, dim=1)
        return alpha.detach().numpy()
