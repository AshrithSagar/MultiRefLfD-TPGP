"""
lfd/utils/gp.py \n
Gaussian processes utils
"""

from typing import List, Tuple

import gpytorch
import torch
import torch.nn as nn
from torch import Tensor


class SingleOutputGPModel(gpytorch.models.ApproximateGP):
    """Model for a single output dimension"""

    def __init__(self, inducing_points: Tensor):
        """
        :param inducing_points: Inducing points for the GP model
        """
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )

    def forward(self, x: Tensor):
        """
        Compute the forward pass of the GP model.

        :param x: Input tensor of shape (batch_size, input_dim)
        :return: MultivariateNormal distribution
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class LocalPolicyGP(nn.Module):
    """
    Multi-output GP model for local policy learning.
    Trains one GP per output dimension.
    """

    def __init__(self, input_dim: int, output_dim: int, num_inducing=128):
        """
        :param input_dim: Input dimension
        :param output_dim: Output dimension
        :param num_inducing: Number of inducing points for each GP
        """
        super().__init__()
        self.models = nn.ModuleList(
            [
                SingleOutputGPModel(torch.randn(num_inducing, input_dim))
                for _ in range(output_dim)
            ]
        )
        self.likelihoods = nn.ModuleList(
            [gpytorch.likelihoods.GaussianLikelihood() for _ in range(output_dim)]
        )

    def train_model(
        self, train_x: Tensor, train_y: Tensor, num_epochs: int = 100, lr: float = 0.01
    ):
        """
        Train the model using the provided training data.

        :param train_x: Input tensor of shape (batch_size, input_dim)
        :param train_y: Output tensor of shape (batch_size, output_dim)
        :param num_epochs: Number of training epochs
        :param lr: Learning rate for the optimizer
        """
        self.train()
        mlls = [
            gpytorch.mlls.VariationalELBO(likelihood, model, train_y.size(0))
            for likelihood, model in zip(self.likelihoods, self.models)
        ]
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss: Tensor = 0
            for i in range(len(self.models)):
                output = self.models[i](train_x)
                loss += -mlls[i](output, train_y[:, i])
            loss.backward()
            optimizer.step()

    def predict(self, test_x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Predict mean and variance for each output dimension.

        :param test_x: Input tensor of shape (batch_size, input_dim)
        :return mean: Mean tensor of shape (batch_size, output_dim)
        :return var: Variance tensor of shape (batch_size, output_dim)
        """
        self.eval()
        pred_means: List[Tensor] = []
        pred_vars: List[Tensor] = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for model, likelihood in zip(self.models, self.likelihoods):
                preds: gpytorch.likelihoods.Likelihood = likelihood(model(test_x))
                pred_means.append(preds.mean.unsqueeze(1))
                pred_vars.append(preds.variance.unsqueeze(1))
        mean = torch.cat(pred_means, dim=1)
        var = torch.cat(pred_vars, dim=1)
        return mean, var

    def stabilized_predict(self, test_x, beta: float = 2.0) -> Tensor:
        """
        Predict with stabilization based on the gradient of the variance.

        :param test_x: Input tensor of shape (batch_size, input_dim)
        :param beta: Scaling factor for stabilization
        :return: Stabilized mean prediction
        """
        mean, var = self.predict(test_x)
        grad_var = torch.autograd.grad(
            outputs=var.sum(),
            inputs=test_x,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        direction = -grad_var / (grad_var.norm(dim=1, keepdim=True) + 1e-8)
        stabilization = beta * var.sum(dim=1, keepdim=True) * direction
        return mean + stabilization


class FrameRelevanceGP(nn.Module):
    """
    Frame relevance model in multi-frame learning.
    Predicts relevance scores for each frame based on progress values.
    """

    def __init__(self, n_frames: int, num_inducing: int = 32):
        """
        :param n_frames: Number of frames
        :param num_inducing: Number of inducing points for each GP
        """
        super().__init__()
        self.n_frames = n_frames
        self.models = nn.ModuleList(
            [SingleOutputGPModel(torch.randn(num_inducing, 1)) for _ in range(n_frames)]
        )
        self.likelihoods = nn.ModuleList(
            [gpytorch.likelihoods.GaussianLikelihood() for _ in range(n_frames)]
        )

    def predict_alpha(self, phi):
        """Predict raw relevance scores"""
        self.eval()
        alphas = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for model, likelihood in zip(self.models, self.likelihoods):
                preds: gpytorch.likelihoods.Likelihood = likelihood(model(phi))
                alphas.append(preds.mean.squeeze(-1))  # Each alpha is (batch_size,)
        alphas = torch.stack(alphas, dim=1)  # Shape: (batch_size, n_frames)
        # Softmax normalization so they sum to 1 across frames
        return torch.softmax(alphas, dim=1)

    def train_self_supervised(
        self,
        phi_train,
        val_x_global,
        val_delta_x,
        local_policies,
        frame_transforms,
        num_epochs: int = 100,
        lr: float = 0.01,
    ):
        """
        Self-supervised training

        :param phi_train: Progress values (n_samples, 1)
        :param val_x_global: Validation states in global frame (n_samples, input_dim)
        :param val_delta_x: Validation true delta in global frame (n_samples, output_dim)
        :param local_policies: List of trained LocalPolicyGP, one per frame
        :param frame_transforms: List of (rotation, translation) matrices per frame
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Predict frame relevance alpha(phi)
            alpha_weights = self.predict_alpha(phi_train)

            # Predict local deltas for each frame
            local_preds = []
            for m, local_policy in enumerate(local_policies):
                # Transform global x into local frame
                R, t = frame_transforms[m]

                # Only position, not progress
                x_local = (val_x_global[:, :-1] - t) @ R.T

                x_local_phi = torch.cat([x_local, val_x_global[:, -1:]], dim=1)
                mean_local, _ = local_policy.predict(x_local_phi)

                # Transform delta back to global
                mean_global = mean_local @ R

                local_preds.append(mean_global)

            # (n_samples, output_dim, n_frames)
            local_preds = torch.stack(local_preds, dim=2)

            # Weighted sum of predictions
            weighted_mean = (local_preds * alpha_weights.unsqueeze(1)).sum(dim=2)

            # Negative log likelihood loss (assumes isotropic Gaussian)
            loss = nn.functional.mse_loss(weighted_mean, val_delta_x)
            loss.backward()
            optimizer.step()
