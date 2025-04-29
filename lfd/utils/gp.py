"""
lfd/utils/gp.py \n
Gaussian processes utils
"""

from dataclasses import dataclass
from typing import Dict

import gpytorch
import torch
import tqdm
from deprecated import deprecated
from torch import Tensor


class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        train_x: Tensor,  # (n_data, n_dim)
        num_tasks: int,
        num_inducing: int = 500,
        matern_nu: float = 2.5,
    ):
        inducing_points = train_x[torch.randperm(train_x.size(0))[:num_inducing]]
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=torch.Size([num_tasks])
        )
        base_variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        variational_strategy = (
            gpytorch.variational.IndependentMultitaskVariationalStrategy(
                base_variational_strategy, num_tasks=num_tasks
            )
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_tasks])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=matern_nu, batch_shape=torch.Size([num_tasks])
            ),
            batch_shape=torch.Size([num_tasks]),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class LocalPolicyGP:
    """GP model for learning local policies"""

    def __init__(self, X_train: Tensor, Y_train: Tensor, num_inducing=500):
        self.X_train = X_train
        self.Y_train = Y_train
        self.num_inducing = num_inducing

    def _train_frame(
        self, X_frame: Tensor, Y_frame: Tensor, num_epochs: int = 100, lr: float = 0.1
    ):
        train_x = X_frame.reshape(-1, 3)  # (n_traj * n_length, 3)
        train_y = Y_frame.reshape(-1, 3)

        model = MultitaskGPModel(train_x, num_tasks=3, num_inducing=self.num_inducing)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters()},
                {"params": likelihood.parameters()},
            ],
            lr=lr,
        )

        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

        epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
        for i in epochs_iter:
            optimizer.zero_grad()
            output = model(train_x)
            loss: Tensor = -mll(output, train_y)
            epochs_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()

        return model, likelihood

    def train(self, num_epochs=100, lr=0.1):
        @dataclass
        class LocalGP:
            model: MultitaskGPModel
            likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood

        local_gps: Dict[int, LocalGP] = {}
        for m in range(self.X_train.shape[0]):
            print(f"Training GP for frame {m + 1}...")
            model, likelihood = self._train_frame(
                self.X_train[m], self.Y_train[m], num_epochs=num_epochs, lr=lr
            )
            local_gps[m] = LocalGP(model, likelihood)

        self.local_gps = local_gps
        return local_gps

    def predict(self):
        n_frames, n_traj, n_length, n_dim = self.X_train.shape

        mean_preds = torch.empty_like(self.X_train).clone()
        covar_preds = torch.empty_like(self.X_train).clone()
        covar_preds = covar_preds.unsqueeze(-1).expand(-1, -1, -1, -1, n_dim).clone()

        for m in range(n_frames):
            model = self.local_gps[m].model
            likelihood = self.local_gps[m].likelihood

            model.eval()
            likelihood.eval()

            with torch.no_grad():
                for i in range(n_traj):
                    traj = self.X_train[m, i]
                    preds_traj = likelihood(model(traj))
                    mean_preds[m, i] = preds_traj.mean
                    covar = preds_traj.covariance_matrix.view(
                        n_length, n_dim, n_length, n_dim
                    )
                    covar_per_point = covar.diagonal(dim1=0, dim2=2).permute(2, 0, 1)
                    covar_preds[m, i] = covar_per_point

        self.mean_preds = mean_preds
        self.covar_preds = covar_preds

    @deprecated
    def predict_2(self):
        n_frames, n_traj, n_length, n_dim = self.X_train.shape

        mean_preds = torch.empty_like(self.X_train).clone()
        covar_preds = torch.empty_like(self.X_train).clone()
        covar_preds = covar_preds.unsqueeze(-1).expand(-1, -1, -1, -1, n_dim).clone()

        for m in range(n_frames):
            model = self.local_gps[m].model
            likelihood = self.local_gps[m].likelihood

            model.eval()
            likelihood.eval()

            with torch.no_grad():
                for i in range(n_traj):
                    for j in range(n_length):
                        point = self.X_train[m, i, j]
                        preds_point = likelihood(model(point))
                        mean_preds[m, i, j] = preds_point.mean
                        covar_preds[m, i, j] = preds_point.covariance_matrix


class GPModel(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        train_x: Tensor,  # (n_data, n_dim)
        num_inducing: int = 500,
    ):
        inducing_points = train_x[:num_inducing]
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class FrameRelevanceGP:
    """
    Frame relevance model in multi-frame learning.
    Predicts relevance scores for each frame based on progress values.
    """
