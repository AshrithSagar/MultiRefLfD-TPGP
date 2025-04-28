"""
lfd/utils/gp.py \n
Gaussian processes utils
"""

from typing import Tuple

import gpytorch
import torch
import tqdm
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    LMCVariationalStrategy,
    VariationalStrategy,
)
from torch import Tensor


class MultitaskGPModel(ApproximateGP):
    def __init__(
        self,
        num_latents: int,
        num_tasks: int,
        num_inducing: int = 16,
        matern_nu: float = 2.5,
    ):
        inducing_points = torch.rand(num_latents, num_inducing, 1)
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )
        variational_strategy = LMCVariationalStrategy(
            VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1,
        )
        super().__init__(variational_strategy)

        self.mean_module = ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = ScaleKernel(
            MaternKernel(nu=matern_nu, batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents]),
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class LocalPolicyGP:
    """GP model for learning local policies"""

    def __init__(self, num_latents: int, num_tasks: int):
        self.model = MultitaskGPModel(num_latents=num_latents, num_tasks=num_tasks)
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def train(
        self, train_x: Tensor, train_y: Tensor, num_epochs: int = 100, lr: float = 0.01
    ):
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(
            [
                {"params": self.model.parameters()},
                {"params": self.likelihood.parameters()},
            ],
            lr=lr,
        )

        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=train_y.size(0)
        )

        epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
        for i in epochs_iter:
            # Within each iteration, we will go over each minibatch of data
            optimizer.zero_grad()
            output = self.model(train_x)
            loss: Tensor = -mll(output, train_y)
            epochs_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()

    def predict(self, test_x: Tensor) -> Tuple[Tensor, Tensor]:
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(test_x))
            mean = predictions.mean
            lower, upper = predictions.confidence_region()

        return mean, lower, upper


class FrameRelevanceGP:
    """
    Frame relevance model in multi-frame learning.
    Predicts relevance scores for each frame based on progress values.
    """

    pass
