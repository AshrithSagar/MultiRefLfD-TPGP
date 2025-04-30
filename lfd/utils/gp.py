"""
lfd/utils/gp.py \n
Gaussian processes utils
"""

from dataclasses import dataclass
from typing import Dict, Optional

import gpytorch
import torch
import tqdm
from deprecated import deprecated
from torch import Tensor


class DataNormalizer:
    """Normalization utilities"""

    def __init__(self, X: Tensor, Y: Tensor):
        self.X_mean = X.mean(dim=0)
        self.X_std = X.std(dim=0) + 1e-6
        self.Y_mean = Y.mean(dim=0)
        self.Y_std = Y.std(dim=0) + 1e-6

    def normalize_inputs(self, X: Tensor) -> Tensor:
        return (X - self.X_mean) / self.X_std

    def normalize_outputs(self, Y: Tensor) -> Tensor:
        return (Y - self.Y_mean) / self.Y_std

    def denormalize_outputs(self, Y_norm: Tensor) -> Tensor:
        return Y_norm * self.Y_std + self.Y_mean

    def rescale_covariance(self, covar: Tensor) -> Tensor:
        D = torch.diag(self.Y_std)
        return torch.einsum("ij,njk,kl->nil", D, covar, D)


class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        train_x: Tensor,  # (n_data, n_dim)
        num_tasks: int,
        num_inducing: Optional[int] = 64,
        matern_nu: float = 2.5,
    ):
        inducing_points = train_x[torch.randperm(train_x.size(0))[:num_inducing]]
        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(
            num_inducing_points=inducing_points.size(0),
            batch_shape=torch.Size([num_tasks]),
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

    def __init__(
        self, X_train: Tensor, Y_train: Tensor, num_inducing: Optional[int] = 64
    ):
        self.X_train = X_train
        self.Y_train = Y_train
        self.num_inducing = num_inducing

    def _train_frame(
        self,
        X_frame: Tensor,
        Y_frame: Tensor,
        num_epochs: int = 100,
        lr: float = 0.01,
        notebook=False,
    ):
        train_x = X_frame.reshape(-1, 3)  # (n_traj * n_length, 3)
        train_y = Y_frame.reshape(-1, 3)

        normalizer = DataNormalizer(train_x, train_y)
        train_x_norm = normalizer.normalize_inputs(train_x)
        train_y_norm = normalizer.normalize_outputs(train_y)

        model = MultitaskGPModel(
            train_x_norm, num_tasks=3, num_inducing=self.num_inducing
        )
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)

        model.covar_module.base_kernel.lengthscale = torch.tensor(1.0)
        model.covar_module.outputscale = torch.tensor(1.0)
        model.normalizer = normalizer

        model.train()
        likelihood.train()

        variational_ngd_optimizer = gpytorch.optim.NGD(
            model.variational_parameters(), num_data=train_y_norm.size(0), lr=0.1
        )
        hyperparameter_optimizer = torch.optim.Adam(
            [
                {"params": model.hyperparameters()},
                {"params": likelihood.parameters()},
            ],
            lr=lr,
        )

        mll = gpytorch.mlls.VariationalELBO(
            likelihood, model, num_data=train_y_norm.size(0)
        )

        _tqdm = tqdm.notebook.tqdm if notebook else tqdm.tqdm
        epochs_iter = _tqdm(range(num_epochs), desc="Epoch")
        for i in epochs_iter:
            variational_ngd_optimizer.zero_grad()
            hyperparameter_optimizer.zero_grad()
            output = model(train_x_norm)
            loss: Tensor = -mll(output, train_y_norm)
            epochs_iter.set_postfix(loss=loss.item())
            loss.backward()
            variational_ngd_optimizer.step()
            hyperparameter_optimizer.step()

        return model, likelihood

    def train(self, num_epochs=100, lr=0.1, notebook=False):
        @dataclass
        class LocalGP:
            model: MultitaskGPModel
            likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood

        local_gps: Dict[int, LocalGP] = {}
        for m in range(self.X_train.shape[0]):
            print(f"Training GP for frame {m + 1}...")
            model, likelihood = self._train_frame(
                self.X_train[m],
                self.Y_train[m],
                num_epochs=num_epochs,
                lr=lr,
                notebook=notebook,
            )
            local_gps[m] = LocalGP(model, likelihood)

        self.local_gps = local_gps

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

            normalizer = model.normalizer
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                for i in range(n_traj):
                    traj = self.X_train[m, i]
                    traj_norm = normalizer.normalize_inputs(traj)
                    preds_traj = likelihood(model(traj_norm))

                    mean_preds[m, i] = normalizer.denormalize_outputs(preds_traj.mean)

                    covar = preds_traj.covariance_matrix.view(
                        n_length, n_dim, n_length, n_dim
                    )
                    covar_per_point = covar.diagonal(dim1=0, dim2=2).permute(2, 0, 1)
                    covar_rescaled = normalizer.rescale_covariance(covar_per_point)
                    covar_preds[m, i] = covar_rescaled

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
        num_inducing: int = 64,
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
