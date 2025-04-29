"""
lfd/utils/gp.py \n
Gaussian processes utils
"""

import gpytorch
import torch
import tqdm
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


def train_local_gp(X_frame: Tensor, Y_frame: Tensor, num_epochs=100):
    train_x = X_frame.reshape(-1, 3)  # (n_traj * n_length, 3)
    train_y = Y_frame.reshape(-1, 3)

    model = MultitaskGPModel(train_x, num_tasks=3)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=0.1,
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


class FrameRelevanceGP:
    """
    Frame relevance model in multi-frame learning.
    Predicts relevance scores for each frame based on progress values.
    """
