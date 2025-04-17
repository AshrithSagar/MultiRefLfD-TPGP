"""
lfd/utils/gp.py \n
Gaussian processes utils
"""

from typing import List, Tuple

import pyro.contrib.gp as gp
import pyro.contrib.gp.kernels as kernels
import pyro.contrib.gp.likelihoods as likelihoods
import torch
import torch.optim as optim
from numpy.typing import NDArray
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam, PyroOptim
from torch import Tensor


# TODO: Verify
class LocalPolicyGP:
    """
    Local policy modeled as independent Gaussian Processes for each output dimension.
    Uses a Matérn 5/2 kernel and sparse variational approximation with Gaussian likelihood.
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
            lik = likelihoods.Gaussian(variance=torch.tensor(noise))
            gpr = gp.models.VariationalSparseGP(
                self.X,
                self.Y[:, i],
                kernel,
                Xu=torch.tensor(Xu, dtype=torch.float32),
                likelihood=lik,
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
                print(f"[LocalPolicyGP] Step {step} ELBO loss: {sum(losses):.4f}")

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
            mu, var = m(X_new_t, full_cov=False)
            means.append(mu.detach())
            vars_.append(var.detach())
        means_t = torch.stack(means, dim=1)  # (K, D_out)
        vars_t = torch.stack(vars_, dim=1)
        return means_t.numpy(), vars_t.numpy()


# TODO: Verify
class FrameRelevanceGP:
    """
    Frame relevance modeled as a single sparse GP over the progress scalar phi,
    with separate variational inducing weights. Outputs softmax-normalized
    relevance weights for each frame.
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
        self.phi = torch.tensor(phi.reshape(-1, 1), dtype=torch.float32)
        self.num_frames = num_frames
        self.Xu = torch.tensor(Xu.reshape(-1, 1), dtype=torch.float32)

        # Variational parameters (u) for each frame: shape (num_frames, M)
        M = self.Xu.shape[0]
        self.u = torch.nn.Parameter(torch.zeros(num_frames, M))

        # Kernel and prior covariance
        self.kernel = kernels.Matern52(input_dim=1)
        Kuu = self.kernel(self.Xu, self.Xu) + noise * torch.eye(M)
        self.Kuu_inv = torch.inverse(Kuu)

        # Optimizer for variational weights
        self.optimizer = optim.Adam([self.u], lr=lr)

    def predict_raw(self, phi_new: Tensor) -> Tensor:
        """
        Compute raw (unnormalized) relevance scores for new progress values.

        :param phi_new: Torch tensor, shape (K, 1)
        :return: raw scores tensor of shape (K, num_frames)
        """
        # Compute cross-covariances
        Kxz = self.kernel(phi_new, self.Xu)  # (K, M)
        # Sparse GP predictive mean: Kxz @ Kuu_inv @ u[m]
        # For each frame m, use u[m] row
        raw = Kxz @ (self.Kuu_inv @ self.u.T)  # (K, num_frames)
        return raw

    def predict(self, phi_new: NDArray) -> NDArray:
        """
        Compute normalized relevance weights.

        :param phi_new: New progress array, shape (K,)
        :return: relevance weights, shape (K, num_frames)
        """
        phi_t = torch.tensor(phi_new.reshape(-1, 1), dtype=torch.float32)
        raw = self.predict_raw(phi_t)
        alpha = torch.softmax(raw, dim=1)
        return alpha.detach().numpy()

    def train(
        self,
        phi: NDArray,
        local_means: NDArray,
        local_vars: NDArray,
        deltas: NDArray,
        num_steps: int = 1000,
        log_every: int = 100,
    ) -> None:
        """
        Self-supervised training using the local means and variances of the GP outputs.
        The training objective is to minimize the negative log-likelihood of the
        deltas under the Gaussian distribution defined by the local means and variances.
        The relevance weights are learned by maximizing the ELBO.
        This is done by optimizing the variational parameters u.

        :param phi: Progress values, shape (N,)
        :param local_means: Local means, shape (N, num_frames, D_out)
        :param local_vars: Local variances, shape (N, num_frames, D_out)
        :param deltas: Deltas, shape (N, D_out)
        :param num_steps: Number of training steps
        :param log_every: Log every log_every steps
        """
        phi_t = torch.tensor(phi.reshape(-1, 1), dtype=torch.float32)
        mus_t = torch.tensor(local_means, dtype=torch.float32)
        vars_t = torch.tensor(local_vars, dtype=torch.float32)
        deltas_t = torch.tensor(deltas, dtype=torch.float32)

        for step in range(num_steps):
            # Predict relevance
            raw = self.predict_raw(phi_t)  # (N, num_frames)
            alpha = torch.softmax(raw, dim=1)  # (N, num_frames)

            # Mixture mean and variance
            # expand alpha to match output dim
            alpha_exp = alpha.unsqueeze(2)  # (N, num_frames, 1)
            mu_mix = (alpha_exp * mus_t).sum(dim=1)  # (N, D_out)
            var_mix = (alpha_exp**2 * vars_t).sum(dim=1)  # (N, D_out)

            # Negative log-likelihood under Gaussian
            nll = 0.5 * (((deltas_t - mu_mix) ** 2) / var_mix).sum()
            nll = nll + 0.5 * torch.log(var_mix).sum()

            # Optimize
            self.optimizer.zero_grad()
            nll.backward()
            self.optimizer.step()

            if log_every and step % log_every == 0:
                print(f"[FrameRelevanceGP] Step {step} NLL: {nll.item():.4f}")
