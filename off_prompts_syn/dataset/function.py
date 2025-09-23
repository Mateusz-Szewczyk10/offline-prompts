"""Class to handle synthetic dataset generation."""
from dataclasses import dataclass
from typing import Optional

import torch
from torch.distributions import Uniform, MultivariateNormal, Normal

from ..utils import torch_seed, check_tensor


@dataclass
class ContextQueryGenerator:
    """Generator of context and query.

    Parameters
    -------
    dim_context: int, default=5 (> 0)
        Dimension of context.

    dim_query: int, default=5 (> 0)
        Dimension of query.

    context_query_covariance: torch.Tensor, shape (dim_context + dim_query, dim_context + dim_query), default = None
        Covariance between context and query vectors.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    dim_context: int = 5
    dim_query: int = 5
    context_query_covariance: Optional[torch.Tensor] = None
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.random_state is not None:
            torch_seed(self.random_state, device=self.device)

        self.dim = self.dim_context + self.dim_query

        if self.context_query_covariance is None:
            logit = Uniform(
                torch.full((self.dim, self.dim), -1.0, device=self.device),
                torch.full((self.dim, self.dim), 1.0, device=self.device),
            ).sample()

            self.context_query_covariance = logit @ logit.T

        else:
            check_tensor(
                self.context_query_covariance,
                name="context_query_covariance",
                expected_dim=2,
            )
            if self.context_query_covariance.shape != (self.dim, self.dim):
                raise ValueError(
                    "The shape of context_query_covariance must be (dim_context + dim_query, "
                    "dim_context + dim_query), but found False."
                )

        self.dist = MultivariateNormal(
            torch.zeros((self.dim,), device=self.device),
            covariance_matrix=self.context_query_covariance,
        )

    def sample_context_and_query(self, n_samples: int):
        """Sample context and query.

        Parameters
        -------
        n_samples: int
            Number of samples.

        Return
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            Feature vector of each user.

        query: torch.Tensor, shape (n_samples, dim_query)
            Query vector given by users.

        """
        return self.dist.sample((n_samples,))


@dataclass
class CandidateActionsGenerator:
    """Generator of candidate actions.

    Parameters
    -------
    n_actions: int, default=1000 (> 0)
        Number of discrete actions.

    dim_action_embedding: int, default=5 (> 0)
        Dimension of action embedding.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    n_actions: int = 1000
    dim_action_embedding: int = 5
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.random_state is not None:
            torch_seed(self.random_state, device=self.device)

        self.action_embedding = torch.normal(
            mean=torch.zeros(
                (self.n_actions, self.dim_action_embedding), device=self.device
            ),
            std=torch.ones(
                (self.n_actions, self.dim_action_embedding), device=self.device
            ),
        )


@dataclass
class AuxiliaryOutputGenerator:
    """Generator of auxiliary output.

    Parameters
    -------
    dim_query: int, default=5 (> 0)
        Dimension of query.

    dim_action_embdding: int, default=5 (> 0)
        Dimension of action embedding.

    dim_auxiliary_output: int, default=5 (> 0)
        Dimension of auxiliary output.

    query_coefficient_matrix: torch.Tensor, shape (dim_query, dim_auxiliary_output), default=None.
        Coefficient vector of query.

    action_coefficient_matrix: torch.Tensor, shape (dim_action_embedding, dim_auxiliary_output), default=None.
        Coefficient vector of action embedding.

    noise_level: float = 0.0
        Noise level on auxiliary output.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    dim_query: int = 5
    dim_action_embedding: int = 5
    dim_auxiliary_output: int = 5
    query_coefficient_matrix: Optional[torch.Tensor] = None
    action_coefficient_matrix: Optional[torch.Tensor] = None
    noise_level: float = 0.0
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.random_state is not None:
            torch_seed(self.random_state, device=self.device)

        if self.query_coefficient_matrix is None:
            self.query_coefficient_matrix = torch.randn(
                (self.dim_query, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.query_coefficient_matrix,
                name="query_coefficient_matrix",
                expected_dim=2,
            )
            if self.query_coefficient_matrix.shape != (
                self.dim_query,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of query_coefficient_matrix must be (dim_query, dim_auxiliary_output), but found False"
                )

        if self.action_coefficient_matrix is None:
            self.action_coefficient_matrix = torch.randn(
                (self.dim_action_embedding, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.action_coefficient_matrix,
                name="action_coefficient_matrix",
                expected_dim=2,
            )
            if self.action_coefficient_matrix.shape != (
                self.dim_action_embedding,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of action_coefficient_matrix must be (dim_action_embedding, dim_auxiliary_output), but found False"
                )

    def _sample_auxiliary_output(
        self,
        query: torch.Tensor,
        action_embedding: torch.Tensor,
    ):
        """Sample auxiliary output.

        Parameters
        -------
        query: torch.Tensor, shape (n_samples, dim_query)
            Query.

        action_embedding: torch.Tensor, shape (n_samples, dim_action_embedding)
            Action embedding.

        Return
        -------
        auxiliary_output: torch.Tensor, shape (n_samples, dim_auxiliary_output)
            Auxiliary output.

        """
        n_samples = len(query)

        output = (
            query @ self.query_coefficient_matrix
            + action_embedding @ self.action_coefficient_matrix
        )

        noise = torch.normal(
            mean=output,
            std=torch.full(
                (n_samples, self.dim_auxiliary_output),
                self.noise_level,
                device=self.device,
            ),
        )

        return output + noise

    def sample_auxiliary_output(
        self,
        query: torch.Tensor,
        action_embedding: torch.Tensor,
        enumerate_actions: bool = False,
    ):
        """Sample auxiliary output.

        Parameters
        -------
        query: torch.Tensor, shape (n_samples, dim_query)
            Query.

        action_embedding: torch.Tensor, shape (n_samples, dim_action_embedding) or (n_samples, n_actions, dim_action_embedding) or (n_actions, dim_action_embedding)
            Action embedding.

        enumerate_actions: bool = False.
            Whether to return auxiliary embedding for all actions.

        Return
        -------
        auxiliary_output: torch.Tensor, shape (n_samples, dim_auxiliary_output) or (m_samples, n_actions, dim_auxiliary_output)
            Auxiliary output.

        """
        n_samples = len(query)

        if enumerate_actions:
            n_actions = action_embedding.shape[0]
            auxiliary_output = torch.zeros(
                (n_samples, n_actions, self.dim_auxiliary_output),
                device=self.device,
            )

            for k in range(n_actions):
                auxiliary_output[:, k] = self._sample_auxiliary_output(
                    query=query,
                    action_embedding=torch.tile(action_embedding[k], (n_samples, 1)),
                )

        else:
            if action_embedding.dim() == 2:
                auxiliary_output = self._sample_auxiliary_output(
                    query=query,
                    action_embedding=action_embedding,
                )

            elif action_embedding.dim() == 3:
                n_actions = action_embedding.shape[1]
                auxiliary_output = torch.zeros(
                    (n_samples, n_actions, self.dim_auxiliary_output),
                    device=self.device,
                )

                for k in range(n_actions):
                    auxiliary_output[:, k] = self._sample_auxiliary_output(
                        query=query,
                        action_embedding=action_embedding[:, k],
                    )

        return auxiliary_output


@dataclass
class RewardSimulator:
    """Simulator of expected reward.

    Parameters
    -------
    dim_context: int, default=5 (> 0)
        Dimension of context.

    dim_query: int, default=5 (> 0)
        Dimension of query.

    dim_auxiliary_output: int, default=5 (> 0)
        Dimension of auxiliary output.

    context_coefficient_matrix: torch.Tensor, shape (dim_context, dim_auxiliary_output), default=None.
        Coefficient vector of context.

    query_coefficient_matrix: torch.Tensor, shape (dim_query, dim_auxiliary_output), default=None.
        Coefficient vector of query.

    normalizer: float, default=None.
        Normalization term of expected reward.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    dim_context: int = 5
    dim_query: int = 5
    dim_auxiliary_output: int = 5
    context_coefficient_matrix: Optional[torch.Tensor] = None
    query_coefficient_matrix: Optional[torch.Tensor] = None
    normalizer: Optional[float] = None
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.random_state is not None:
            torch_seed(self.random_state, device=self.device)

        if self.context_coefficient_matrix is None:
            self.context_coefficient_matrix = torch.randn(
                (self.dim_context, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.context_coefficient_matrix,
                name="context_coefficient_matrix",
                expected_dim=2,
            )
            if self.context_coefficient_matrix.shape != (
                self.dim_context,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of context_coefficient_matrix must be (dim_context, dim_auxiliary_output), but found False"
                )

        if self.query_coefficient_matrix is None:
            self.query_coefficient_matrix = torch.randn(
                (self.dim_query, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.query_coefficient_matrix,
                name="query_coefficient_matrix",
                expected_dim=2,
            )
            if self.query_coefficient_matrix.shape != (
                self.dim_query,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of query_coefficient_matrix must be (dim_query, dim_auxiliary_output), but found False"
                )

        if self.normalizer is None:
            self.normalizer = (
                (self.dim_context + self.dim_query) * self.dim_auxiliary_output * 5
            )

    def calc_expected_reward(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        auxiliary_output: torch.Tensor,
    ):
        """Sample auxiliary output.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            Context.

        query: torch.Tensor, shape (n_samples, dim_query)
            Query.

        auxiliary_output: torch.Tensor, shape (n_samples, dim_auxiliary_output)
            Auxiliary output.

        Return
        -------
        expected_reward: torch.Tensor, shape (n_samples, )
            Expected reward of given auxiliary output on given context.

        """
        n_samples = len(context)
        context = context[:, None, :]
        query = query[:, None, :]
        auxiliary_output = auxiliary_output[:, :, None]

        context_coefficient_matrix = torch.tile(
            self.context_coefficient_matrix, (n_samples, 1, 1)
        )
        query_coefficient_matrix = torch.tile(
            self.query_coefficient_matrix, (n_samples, 1, 1)
        )

        coefficient = torch.bmm(context, context_coefficient_matrix) + torch.bmm(
            query, query_coefficient_matrix
        )
        expected_reward = torch.bmm(coefficient, auxiliary_output).squeeze()
        expected_reward = expected_reward / self.normalizer
        return expected_reward


##### non-linear #####


@dataclass
class MixtureOfGaussianCandidateActionsGenerator(CandidateActionsGenerator):
    """Generator of candidate actions.

    Parameters
    -------
    n_actions: int, default=1000 (> 0)
        Number of discrete actions.

    n_clusters: int, default=100 (> 0)
        Number of gaussian distribution.

    dim_action_embedding: int, default=5 (> 0)
        Dimension of action embedding.

    cluster_centers: torch.Tensor, shape (n_clusters, dim_action_embedding), default=None.
        Feature vector of cluster centers.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    n_clusters: int = 100
    n_actions_per_cluster: int = 1
    dim_action_embedding: int = 5
    cluster_centers: Optional[torch.Tensor] = None
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        self.n_actions = self.n_clusters * self.n_actions_per_cluster

        if self.random_state is not None:
            torch_seed(self.random_state, device=self.device)

        if self.cluster_centers is not None:
            if self.cluster_centers.shape != (
                self.n_clusters,
                self.dim_action_embedding,
            ):
                raise ValueError(
                    f"The shape of cluster_centers must be equal to (n_clusters, dim_action_embedding), but found {self.cluster_centers.shape}"
                )
        else:
            cluster_centers_dist = Normal(
                loc=torch.zeros(
                    (self.n_clusters, self.dim_action_embedding),
                    device=self.device,
                ),
                scale=torch.ones(
                    (self.n_clusters, self.dim_action_embedding),
                    device=self.device,
                ),
            )
            self.cluster_centers = cluster_centers_dist.sample()

        action_embedding_dist = Normal(
            loc=self.cluster_centers,
            scale=torch.full(
                (self.n_clusters, self.dim_action_embedding),
                0.1,
            ),
        )
        action_embedding = action_embedding_dist.sample((self.n_actions_per_cluster,))
        self.action_embedding = action_embedding.reshape(
            (-1, self.dim_action_embedding)
        )


@dataclass
class ConfoundedAuxiliaryOutputGenerator(AuxiliaryOutputGenerator):
    """Generator of auxiliary output with confounder.

    Parameters
    -------
    dim_query: int, default=5 (> 0)
        Dimension of query.

    dim_action_embdding: int, default=5 (> 0)
        Dimension of action embedding.

    dim_auxiliary_output: int, default=5 (> 0)
        Dimension of auxiliary output.

    dim_confounded_auxiliary_output: int, default=0 (>= 0)
        Dimension of the confounder in the auxiliary output.

    query_coefficient_matrix: torch.Tensor, shape (dim_query, dim_auxiliary_output), default=None.
        Coefficient vector of query.

    action_coefficient_matrix: torch.Tensor, shape (dim_action_embedding, dim_auxiliary_output), default=None.
        Coefficient vector of action embedding.

    noise_level: float, default=0.0 (> 0)
        Noise level on auxiliary output.

    confounder_scale: float, default=1.0 (> 0)
        Scale of the confounder.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    dim_query: int = 5
    dim_action_embedding: int = 5
    dim_auxiliary_output: int = 5
    dim_confounded_auxiliary_output: Optional[int] = None
    query_coefficient_matrix: Optional[torch.Tensor] = None
    action_coefficient_matrix: Optional[torch.Tensor] = None
    noise_level: float = 0.0
    confounder_scale: float = 1.0
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.random_state is not None:
            torch_seed(self.random_state, device=self.device)

        if self.dim_confounded_auxiliary_output is None:
            self.dim_confounded_auxiliary_output = 0
        elif self.dim_confounded_auxiliary_output > self.dim_auxiliary_output:
            raise ValueError(
                "dim_confounded_auxiliary_output must be equal to or smaller than dim_auxiliary_output, but found False"
            )

        if self.query_coefficient_matrix is None:
            self.query_coefficient_matrix = torch.randn(
                (self.dim_query, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.query_coefficient_matrix,
                name="query_coefficient_matrix",
                expected_dim=2,
            )
            if self.query_coefficient_matrix.shape != (
                self.dim_query,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of query_coefficient_matrix must be (dim_query, dim_auxiliary_output), but found False"
                )

        if self.action_coefficient_matrix is None:
            self.action_coefficient_matrix = torch.randn(
                (self.dim_action_embedding, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.action_coefficient_matrix,
                name="action_coefficient_matrix",
                expected_dim=2,
            )
            if self.action_coefficient_matrix.shape != (
                self.dim_action_embedding,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of action_coefficient_matrix must be (dim_action_embedding, dim_auxiliary_output), but found False"
                )

    def _sample_auxiliary_output(
        self,
        query: torch.Tensor,
        action_embedding: torch.Tensor,
    ):
        """Sample auxiliary output.

        Parameters
        -------
        query: torch.Tensor, shape (n_samples, dim_query)
            Query.

        action_embedding: torch.Tensor, shape (n_samples, dim_action_embedding)
            Action embedding.

        Return
        -------
        auxiliary_output: torch.Tensor, shape (n_samples, dim_auxiliary_output)
            Auxiliary output.

        """
        n_samples = len(query)

        output = (
            query @ self.query_coefficient_matrix
            + action_embedding @ self.action_coefficient_matrix
        )

        if self.dim_confounded_auxiliary_output > 0:
            output[:, -self.dim_confounded_auxiliary_output :] = torch.normal(
                mean=torch.zeros(
                    (n_samples, self.dim_confounded_auxiliary_output),
                    device=self.device,
                ),
                std=torch.full(
                    (n_samples, self.dim_confounded_auxiliary_output),
                    self.confounder_scale,
                    device=self.device,
                ),
            )

        noise = torch.normal(
            mean=output,
            std=torch.full(
                (n_samples, self.dim_auxiliary_output),
                self.noise_level,
                device=self.device,
            ),
        )

        return output + noise


@dataclass
class RationalAuxiliaryOutputGenerator(AuxiliaryOutputGenerator):
    """Generator of auxiliary output.

    Parameters
    -------
    dim_query: int, default=5 (> 0)
        Dimension of query.

    dim_action_embdding: int, default=5 (> 0)
        Dimension of action embedding.

    dim_auxiliary_output: int, default=5 (> 0)
        Dimension of auxiliary output.

    query_coefficient_matrix: torch.Tensor, shape (dim_query, dim_auxiliary_output), default=None.
        Coefficient vector of query.

    action_coefficient_matrix: torch.Tensor, shape (dim_action_embedding, dim_auxiliary_output), default=None.
        Coefficient vector of action embedding.

    noise_level: float = 0.0
        Noise level on auxiliary output.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    dim_query: int = 5
    dim_action_embedding: int = 5
    dim_auxiliary_output: int = 5
    query_coefficient_matrix: Optional[torch.Tensor] = None
    action_coefficient_matrix: Optional[torch.Tensor] = None
    bias: Optional[float] = None
    noise_level: float = 0.0
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.random_state is not None:
            torch_seed(self.random_state, device=self.device)

        if self.query_coefficient_matrix is None:
            self.query_coefficient_matrix = torch.randn(
                (self.dim_query, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.query_coefficient_matrix,
                name="query_coefficient_matrix",
                expected_dim=2,
            )
            if self.query_coefficient_matrix.shape != (
                self.dim_query,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of query_coefficient_matrix must be (dim_query, dim_auxiliary_output), but found False"
                )

        if self.action_coefficient_matrix is None:
            self.action_coefficient_matrix = torch.randn(
                (self.dim_action_embedding, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.action_coefficient_matrix,
                name="action_coefficient_matrix",
                expected_dim=2,
            )
            if self.action_coefficient_matrix.shape != (
                self.dim_action_embedding,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of action_coefficient_matrix must be (dim_action_embedding, dim_auxiliary_output), but found False"
                )

        if self.bias is None:
            self.bias = torch.randn((1, 1)).item()

    def _sample_auxiliary_output(
        self,
        query: torch.Tensor,
        action_embedding: torch.Tensor,
    ):
        """Sample auxiliary output.

        Parameters
        -------
        query: torch.Tensor, shape (n_samples, dim_query)
            Query.

        action_embedding: torch.Tensor, shape (n_samples, dim_action_embedding)
            Action embedding.

        Return
        -------
        auxiliary_output: torch.Tensor, shape (n_samples, dim_auxiliary_output)
            Auxiliary output.

        """
        n_samples = len(query)

        linear = action_embedding @ self.action_coefficient_matrix + self.bias
        non_linear_mapping = linear / (1 + self.bias + linear)
        non_linear_mapping = torch.clip(non_linear_mapping, -5.0, 5.0)

        output = query @ self.query_coefficient_matrix + non_linear_mapping

        noise = torch.normal(
            mean=output,
            std=torch.full(
                (n_samples, self.dim_auxiliary_output),
                self.noise_level,
                device=self.device,
            ),
        )

        return output + noise


@dataclass
class SigmoidAuxiliaryOutputGenerator(AuxiliaryOutputGenerator):
    """Generator of auxiliary output.

    Parameters
    -------
    dim_query: int, default=5 (> 0)
        Dimension of query.

    dim_action_embdding: int, default=5 (> 0)
        Dimension of action embedding.

    dim_auxiliary_output: int, default=5 (> 0)
        Dimension of auxiliary output.

    query_coefficient_matrix: torch.Tensor, shape (dim_query, dim_auxiliary_output), default=None.
        Coefficient vector of query.

    action_coefficient_matrix: torch.Tensor, shape (dim_action_embedding, dim_auxiliary_output), default=None.
        Coefficient vector of action embedding.

    noise_level: float = 0.0
        Noise level on auxiliary output.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    dim_query: int = 5
    dim_action_embedding: int = 5
    dim_auxiliary_output: int = 5
    query_coefficient_matrix: Optional[torch.Tensor] = None
    action_coefficient_matrix: Optional[torch.Tensor] = None
    max_abs_val: float = 5.0
    scaler: float = 1.0
    noise_level: float = 0.0
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.random_state is not None:
            torch_seed(self.random_state, device=self.device)

        if self.query_coefficient_matrix is None:
            self.query_coefficient_matrix = torch.randn(
                (self.dim_query, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.query_coefficient_matrix,
                name="query_coefficient_matrix",
                expected_dim=2,
            )
            if self.query_coefficient_matrix.shape != (
                self.dim_query,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of query_coefficient_matrix must be (dim_query, dim_auxiliary_output), but found False"
                )

        if self.action_coefficient_matrix is None:
            self.action_coefficient_matrix = torch.randn(
                (self.dim_action_embedding, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.action_coefficient_matrix,
                name="action_coefficient_matrix",
                expected_dim=2,
            )
            if self.action_coefficient_matrix.shape != (
                self.dim_action_embedding,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of action_coefficient_matrix must be (dim_action_embedding, dim_auxiliary_output), but found False"
                )

    def _sample_auxiliary_output(
        self,
        query: torch.Tensor,
        action_embedding: torch.Tensor,
    ):
        """Sample auxiliary output.

        Parameters
        -------
        query: torch.Tensor, shape (n_samples, dim_query)
            Query.

        action_embedding: torch.Tensor, shape (n_samples, dim_action_embedding)
            Action embedding.

        Return
        -------
        auxiliary_output: torch.Tensor, shape (n_samples, dim_auxiliary_output)
            Auxiliary output.

        """
        n_samples = len(query)

        linear = (action_embedding @ self.action_coefficient_matrix) * self.scaler
        non_linear_mapping = self.max_abs_val * torch.sigmoid(linear)

        output = query @ self.query_coefficient_matrix + non_linear_mapping

        noise = torch.normal(
            mean=output,
            std=torch.full(
                (n_samples, self.dim_auxiliary_output),
                self.noise_level,
                device=self.device,
            ),
        )

        return output + noise


@dataclass
class TrigonometricAuxiliaryOutputGenerator(AuxiliaryOutputGenerator):
    """Generator of auxiliary output.

    Parameters
    -------
    dim_query: int, default=5 (> 0)
        Dimension of query.

    dim_action_embdding: int, default=5 (> 0)
        Dimension of action embedding.

    dim_auxiliary_output: int, default=5 (> 0)
        Dimension of auxiliary output.

    query_coefficient_matrix: torch.Tensor, shape (dim_query, dim_auxiliary_output), default=None.
        Coefficient vector of query.

    action_coefficient_matrix: torch.Tensor, shape (dim_action_embedding, dim_auxiliary_output), default=None.
        Coefficient vector of action embedding.

    noise_level: float = 0.0
        Noise level on auxiliary output.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    dim_query: int = 5
    dim_action_embedding: int = 5
    dim_auxiliary_output: int = 5
    query_coefficient_matrix: Optional[torch.Tensor] = None
    action_coefficient_matrix: Optional[torch.Tensor] = None
    max_abs_val: float = 5.0
    scaler: float = 1.0
    noise_level: float = 0.0
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.random_state is not None:
            torch_seed(self.random_state)

        if self.query_coefficient_matrix is None:
            self.query_coefficient_matrix = torch.randn(
                (self.dim_query, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.query_coefficient_matrix,
                name="query_coefficient_matrix",
                expected_dim=2,
            )
            if self.query_coefficient_matrix.shape != (
                self.dim_query,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of query_coefficient_matrix must be (dim_query, dim_auxiliary_output), but found False"
                )

        if self.action_coefficient_matrix is None:
            self.action_coefficient_matrix = torch.randn(
                (self.dim_action_embedding, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.action_coefficient_matrix,
                name="action_coefficient_matrix",
                expected_dim=2,
            )
            if self.action_coefficient_matrix.shape != (
                self.dim_action_embedding,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of action_coefficient_matrix must be (dim_action_embedding, dim_auxiliary_output), but found False"
                )

    def _sample_auxiliary_output(
        self,
        query: torch.Tensor,
        action_embedding: torch.Tensor,
    ):
        """Sample auxiliary output.

        Parameters
        -------
        query: torch.Tensor, shape (n_samples, dim_query)
            Query.

        action_embedding: torch.Tensor, shape (n_samples, dim_action_embedding)
            Action embedding.

        Return
        -------
        auxiliary_output: torch.Tensor, shape (n_samples, dim_auxiliary_output)
            Auxiliary output.

        """
        n_samples = len(query)

        linear = (action_embedding @ self.action_coefficient_matrix) * self.scaler
        non_linear_mapping = self.max_abs_val * torch.sin(linear)

        output = query @ self.query_coefficient_matrix + non_linear_mapping

        noise = torch.normal(
            mean=output,
            std=torch.full(
                (n_samples, self.dim_auxiliary_output),
                self.noise_level,
                device=self.device,
            ),
        )

        return output + noise


@dataclass
class PowerAuxiliaryOutputGenerator(AuxiliaryOutputGenerator):
    """Generator of auxiliary output.

    Parameters
    -------
    dim_query: int, default=5 (> 0)
        Dimension of query.

    dim_action_embdding: int, default=5 (> 0)
        Dimension of action embedding.

    dim_auxiliary_output: int, default=5 (> 0)
        Dimension of auxiliary output.

    query_coefficient_matrix: torch.Tensor, shape (dim_query, dim_auxiliary_output), default=None.
        Coefficient vector of query.

    action_coefficient_matrix: torch.Tensor, shape (dim_action_embedding, dim_auxiliary_output), default=None.
        Coefficient vector of action embedding.

    noise_level: float = 0.0
        Noise level on auxiliary output.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    dim_query: int = 5
    dim_action_embedding: int = 5
    dim_auxiliary_output: int = 5
    query_coefficient_matrix: Optional[torch.Tensor] = None
    action_coefficient_matrix: Optional[torch.Tensor] = None
    max_abs_val: float = 5.0
    scaler: float = 1.0
    noise_level: float = 0.0
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.random_state is not None:
            torch_seed(self.random_state, device=self.device)

        if self.query_coefficient_matrix is None:
            self.query_coefficient_matrix = torch.randn(
                (self.dim_query, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.query_coefficient_matrix,
                name="query_coefficient_matrix",
                expected_dim=2,
            )
            if self.query_coefficient_matrix.shape != (
                self.dim_query,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of query_coefficient_matrix must be (dim_query, dim_auxiliary_output), but found False"
                )

        if self.action_coefficient_matrix is None:
            self.action_coefficient_matrix = torch.randn(
                (self.dim_action_embedding, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.action_coefficient_matrix,
                name="action_coefficient_matrix",
                expected_dim=2,
            )
            if self.action_coefficient_matrix.shape != (
                self.dim_action_embedding,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of action_coefficient_matrix must be (dim_action_embedding, dim_auxiliary_output), but found False"
                )

    def _sample_auxiliary_output(
        self,
        query: torch.Tensor,
        action_embedding: torch.Tensor,
    ):
        """Sample auxiliary output.

        Parameters
        -------
        query: torch.Tensor, shape (n_samples, dim_query)
            Query.

        action_embedding: torch.Tensor, shape (n_samples, dim_action_embedding)
            Action embedding.

        Return
        -------
        auxiliary_output: torch.Tensor, shape (n_samples, dim_auxiliary_output)
            Auxiliary output.

        """
        n_samples = len(query)

        x = (action_embedding @ self.action_coefficient_matrix) * self.scaler
        x = torch.where(x < 5, x, 0)
        x = torch.where(x > -5, x, 0)

        non_linear_mapping = (
            (1 / 4)
            * x
            * torch.pow(torch.abs((x - 3) * (x + 3) * (x - 5) * (x + 5)), 1 / 5)
        )

        output = query @ self.query_coefficient_matrix + non_linear_mapping

        noise = torch.normal(
            mean=output,
            std=torch.full(
                (n_samples, self.dim_auxiliary_output),
                self.noise_level,
                device=self.device,
            ),
        )

        return output + noise


@dataclass
class ExponentialAuxiliaryOutputGenerator(AuxiliaryOutputGenerator):
    """Generator of auxiliary output.

    Parameters
    -------
    dim_query: int, default=5 (> 0)
        Dimension of query.

    dim_action_embdding: int, default=5 (> 0)
        Dimension of action embedding.

    dim_auxiliary_output: int, default=5 (> 0)
        Dimension of auxiliary output.

    query_coefficient_matrix: torch.Tensor, shape (dim_query, dim_auxiliary_output), default=None.
        Coefficient vector of query.

    action_coefficient_matrix: torch.Tensor, shape (dim_action_embedding, dim_auxiliary_output), default=None.
        Coefficient vector of action embedding.

    noise_level: float = 0.0
        Noise level on auxiliary output.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    dim_query: int = 5
    dim_action_embedding: int = 5
    dim_auxiliary_output: int = 5
    query_coefficient_matrix: Optional[torch.Tensor] = None
    action_coefficient_matrix: Optional[torch.Tensor] = None
    max_abs_val: float = 5.0
    scaler: float = 1.0
    noise_level: float = 0.0
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.random_state is not None:
            torch_seed(self.random_state, device=self.device)

        if self.query_coefficient_matrix is None:
            self.query_coefficient_matrix = torch.randn(
                (self.dim_query, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.query_coefficient_matrix,
                name="query_coefficient_matrix",
                expected_dim=2,
            )
            if self.query_coefficient_matrix.shape != (
                self.dim_query,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of query_coefficient_matrix must be (dim_query, dim_auxiliary_output), but found False"
                )

        if self.action_coefficient_matrix is None:
            self.action_coefficient_matrix = torch.randn(
                (self.dim_action_embedding, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.action_coefficient_matrix,
                name="action_coefficient_matrix",
                expected_dim=2,
            )
            if self.action_coefficient_matrix.shape != (
                self.dim_action_embedding,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of action_coefficient_matrix must be (dim_action_embedding, dim_auxiliary_output), but found False"
                )

    def _sample_auxiliary_output(
        self,
        query: torch.Tensor,
        action_embedding: torch.Tensor,
    ):
        """Sample auxiliary output.

        Parameters
        -------
        query: torch.Tensor, shape (n_samples, dim_query)
            Query.

        action_embedding: torch.Tensor, shape (n_samples, dim_action_embedding)
            Action embedding.

        Return
        -------
        auxiliary_output: torch.Tensor, shape (n_samples, dim_auxiliary_output)
            Auxiliary output.

        """
        n_samples = len(query)

        x = (action_embedding @ self.action_coefficient_matrix) * self.scaler
        non_linear_mapping = 25 * x / (1 + torch.exp(x) + torch.exp(-x))
        output = query @ self.query_coefficient_matrix + non_linear_mapping

        noise = torch.normal(
            mean=output,
            std=torch.full(
                (n_samples, self.dim_auxiliary_output),
                self.noise_level,
                device=self.device,
            ),
        )

        return output + noise


@dataclass
class ConfoundedRationalAuxiliaryOutputGenerator(AuxiliaryOutputGenerator):
    """Generator of auxiliary output.

    Parameters
    -------
    dim_query: int, default=5 (> 0)
        Dimension of query.

    dim_action_embdding: int, default=5 (> 0)
        Dimension of action embedding.

    dim_auxiliary_output: int, default=5 (> 0)
        Dimension of auxiliary output.

    dim_confounded_auxiliary_output: int, default=0 (>= 0)
        Dimension of the confounder in the auxiliary output.

    query_coefficient_matrix: torch.Tensor, shape (dim_query, dim_auxiliary_output), default=None.
        Coefficient vector of query.

    action_coefficient_matrix: torch.Tensor, shape (dim_action_embedding, dim_auxiliary_output), default=None.
        Coefficient vector of action embedding.

    noise_level: float = 0.0
        Noise level on auxiliary output.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    dim_query: int = 5
    dim_action_embedding: int = 5
    dim_auxiliary_output: int = 5
    dim_confounded_auxiliary_output: Optional[int] = None
    query_coefficient_matrix: Optional[torch.Tensor] = None
    action_coefficient_matrix: Optional[torch.Tensor] = None
    bias: Optional[float] = None
    noise_level: float = 0.0
    confounder_scale: float = 1.0
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.random_state is not None:
            torch_seed(self.random_state, device=self.device)

        if self.dim_confounded_auxiliary_output is None:
            self.dim_confounded_auxiliary_output = 0
        elif self.dim_confounded_auxiliary_output > self.dim_auxiliary_output:
            raise ValueError(
                "dim_confounded_auxiliary_output must be equal to or smaller than dim_auxiliary_output, but found False"
            )

        if self.query_coefficient_matrix is None:
            self.query_coefficient_matrix = torch.randn(
                (self.dim_query, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.query_coefficient_matrix,
                name="query_coefficient_matrix",
                expected_dim=2,
            )
            if self.query_coefficient_matrix.shape != (
                self.dim_query,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of query_coefficient_matrix must be (dim_query, dim_auxiliary_output), but found False"
                )

        if self.action_coefficient_matrix is None:
            self.action_coefficient_matrix = torch.randn(
                (self.dim_action_embedding, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.action_coefficient_matrix,
                name="action_coefficient_matrix",
                expected_dim=2,
            )
            if self.action_coefficient_matrix.shape != (
                self.dim_action_embedding,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of action_coefficient_matrix must be (dim_action_embedding, dim_auxiliary_output), but found False"
                )

        if self.bias is None:
            self.bias = torch.randn((1, 1)).item()

    def _sample_auxiliary_output(
        self,
        query: torch.Tensor,
        action_embedding: torch.Tensor,
    ):
        """Sample auxiliary output.

        Parameters
        -------
        query: torch.Tensor, shape (n_samples, dim_query)
            Query.

        action_embedding: torch.Tensor, shape (n_samples, dim_action_embedding)
            Action embedding.

        Return
        -------
        auxiliary_output: torch.Tensor, shape (n_samples, dim_auxiliary_output)
            Auxiliary output.

        """
        n_samples = len(query)

        linear = action_embedding @ self.action_coefficient_matrix + self.bias
        non_linear_mapping = linear / (1 + self.bias + linear)
        non_linear_mapping = torch.clip(non_linear_mapping, -5.0, 5.0)

        output = query @ self.query_coefficient_matrix + non_linear_mapping

        if self.dim_confounded_auxiliary_output > 0:
            output[:, -self.dim_confounded_auxiliary_output :] = torch.normal(
                mean=torch.zeros(
                    (n_samples, self.dim_confounded_auxiliary_output),
                    device=self.device,
                ),
                std=torch.full(
                    (n_samples, self.dim_confounded_auxiliary_output),
                    self.confounder_scale,
                    device=self.device,
                ),
            )

        noise = torch.normal(
            mean=output,
            std=torch.full(
                (n_samples, self.dim_auxiliary_output),
                self.noise_level,
                device=self.device,
            ),
        )
        return output + noise


@dataclass
class ConfoundedTrigonometricAuxiliaryOutputGenerator(AuxiliaryOutputGenerator):
    """Generator of auxiliary output.

    Parameters
    -------
    dim_query: int, default=5 (> 0)
        Dimension of query.

    dim_action_embdding: int, default=5 (> 0)
        Dimension of action embedding.

    dim_auxiliary_output: int, default=5 (> 0)
        Dimension of auxiliary output.

    dim_confounded_auxiliary_output: int, default=0 (>= 0)
        Dimension of the confounder in the auxiliary output.

    query_coefficient_matrix: torch.Tensor, shape (dim_query, dim_auxiliary_output), default=None.
        Coefficient vector of query.

    action_coefficient_matrix: torch.Tensor, shape (dim_action_embedding, dim_auxiliary_output), default=None.
        Coefficient vector of action embedding.

    noise_level: float = 0.0
        Noise level on auxiliary output.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    dim_query: int = 5
    dim_action_embedding: int = 5
    dim_auxiliary_output: int = 5
    dim_confounded_auxiliary_output: int = 0
    query_coefficient_matrix: Optional[torch.Tensor] = None
    action_coefficient_matrix: Optional[torch.Tensor] = None
    max_abs_val: float = 5.0
    scaler: float = 1.0
    noise_level: float = 0.0
    confounder_scale: float = 1.0
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.random_state is not None:
            torch_seed(self.random_state)

        if self.dim_confounded_auxiliary_output is None:
            self.dim_confounded_auxiliary_output = 0
        elif self.dim_confounded_auxiliary_output > self.dim_auxiliary_output:
            raise ValueError(
                "dim_confounded_auxiliary_output must be equal to or smaller than dim_auxiliary_output, but found False"
            )

        if self.query_coefficient_matrix is None:
            self.query_coefficient_matrix = torch.randn(
                (self.dim_query, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.query_coefficient_matrix,
                name="query_coefficient_matrix",
                expected_dim=2,
            )
            if self.query_coefficient_matrix.shape != (
                self.dim_query,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of query_coefficient_matrix must be (dim_query, dim_auxiliary_output), but found False"
                )

        if self.action_coefficient_matrix is None:
            self.action_coefficient_matrix = torch.randn(
                (self.dim_action_embedding, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.action_coefficient_matrix,
                name="action_coefficient_matrix",
                expected_dim=2,
            )
            if self.action_coefficient_matrix.shape != (
                self.dim_action_embedding,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of action_coefficient_matrix must be (dim_action_embedding, dim_auxiliary_output), but found False"
                )

    def _sample_auxiliary_output(
        self,
        query: torch.Tensor,
        action_embedding: torch.Tensor,
    ):
        """Sample auxiliary output.

        Parameters
        -------
        query: torch.Tensor, shape (n_samples, dim_query)
            Query.

        action_embedding: torch.Tensor, shape (n_samples, dim_action_embedding)
            Action embedding.

        Return
        -------
        auxiliary_output: torch.Tensor, shape (n_samples, dim_auxiliary_output)
            Auxiliary output.

        """
        n_samples = len(query)

        linear = (action_embedding @ self.action_coefficient_matrix) * self.scaler
        non_linear_mapping = self.max_abs_val * torch.sin(linear)

        output = query @ self.query_coefficient_matrix + non_linear_mapping

        if self.dim_confounded_auxiliary_output > 0:
            output[:, -self.dim_confounded_auxiliary_output :] = torch.normal(
                mean=torch.zeros(
                    (n_samples, self.dim_confounded_auxiliary_output),
                    device=self.device,
                ),
                std=torch.full(
                    (n_samples, self.dim_confounded_auxiliary_output),
                    self.confounder_scale,
                    device=self.device,
                ),
            )

        noise = torch.normal(
            mean=output,
            std=torch.full(
                (n_samples, self.dim_auxiliary_output),
                self.noise_level,
                device=self.device,
            ),
        )

        return output + noise


@dataclass
class ConfoundedPowerAuxiliaryOutputGenerator(AuxiliaryOutputGenerator):
    """Generator of auxiliary output.

    Parameters
    -------
    dim_query: int, default=5 (> 0)
        Dimension of query.

    dim_action_embdding: int, default=5 (> 0)
        Dimension of action embedding.

    dim_auxiliary_output: int, default=5 (> 0)
        Dimension of auxiliary output.

    dim_confounded_auxiliary_output: int, default=0 (>= 0)
        Dimension of the confounder in the auxiliary output.

    query_coefficient_matrix: torch.Tensor, shape (dim_query, dim_auxiliary_output), default=None.
        Coefficient vector of query.

    action_coefficient_matrix: torch.Tensor, shape (dim_action_embedding, dim_auxiliary_output), default=None.
        Coefficient vector of action embedding.

    noise_level: float = 0.0
        Noise level on auxiliary output.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    dim_query: int = 5
    dim_action_embedding: int = 5
    dim_auxiliary_output: int = 5
    dim_confounded_auxiliary_output: Optional[int] = None
    query_coefficient_matrix: Optional[torch.Tensor] = None
    action_coefficient_matrix: Optional[torch.Tensor] = None
    max_abs_val: float = 5.0
    scaler: float = 1.0
    noise_level: float = 0.0
    confounder_scale: float = 1.0
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.random_state is not None:
            torch_seed(self.random_state, device=self.device)

        if self.dim_confounded_auxiliary_output is None:
            self.dim_confounded_auxiliary_output = 0
        elif self.dim_confounded_auxiliary_output > self.dim_auxiliary_output:
            raise ValueError(
                "dim_confounded_auxiliary_output must be equal to or smaller than dim_auxiliary_output, but found False"
            )

        if self.query_coefficient_matrix is None:
            self.query_coefficient_matrix = torch.randn(
                (self.dim_query, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.query_coefficient_matrix,
                name="query_coefficient_matrix",
                expected_dim=2,
            )
            if self.query_coefficient_matrix.shape != (
                self.dim_query,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of query_coefficient_matrix must be (dim_query, dim_auxiliary_output), but found False"
                )

        if self.action_coefficient_matrix is None:
            self.action_coefficient_matrix = torch.randn(
                (self.dim_action_embedding, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.action_coefficient_matrix,
                name="action_coefficient_matrix",
                expected_dim=2,
            )
            if self.action_coefficient_matrix.shape != (
                self.dim_action_embedding,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of action_coefficient_matrix must be (dim_action_embedding, dim_auxiliary_output), but found False"
                )

    def _sample_auxiliary_output(
        self,
        query: torch.Tensor,
        action_embedding: torch.Tensor,
    ):
        """Sample auxiliary output.

        Parameters
        -------
        query: torch.Tensor, shape (n_samples, dim_query)
            Query.

        action_embedding: torch.Tensor, shape (n_samples, dim_action_embedding)
            Action embedding.

        Return
        -------
        auxiliary_output: torch.Tensor, shape (n_samples, dim_auxiliary_output)
            Auxiliary output.

        """
        n_samples = len(query)

        x = (action_embedding @ self.action_coefficient_matrix) * self.scaler
        x = torch.where(x < 5, x, 0)
        x = torch.where(x > -5, x, 0)

        non_linear_mapping = (
            (1 / 4)
            * x
            * torch.pow(torch.abs((x - 3) * (x + 3) * (x - 5) * (x + 5)), 1 / 5)
        )

        output = query @ self.query_coefficient_matrix + non_linear_mapping

        if self.dim_confounded_auxiliary_output > 0:
            output[:, -self.dim_confounded_auxiliary_output :] = torch.normal(
                mean=torch.zeros(
                    (n_samples, self.dim_confounded_auxiliary_output),
                    device=self.device,
                ),
                std=torch.full(
                    (n_samples, self.dim_confounded_auxiliary_output),
                    self.confounder_scale,
                    device=self.device,
                ),
            )

        noise = torch.normal(
            mean=output,
            std=torch.full(
                (n_samples, self.dim_auxiliary_output),
                self.noise_level,
                device=self.device,
            ),
        )

        return output + noise


@dataclass
class ConfoundedExponentialAuxiliaryOutputGenerator(AuxiliaryOutputGenerator):
    """Generator of auxiliary output.

    Parameters
    -------
    dim_query: int, default=5 (> 0)
        Dimension of query.

    dim_action_embdding: int, default=5 (> 0)
        Dimension of action embedding.

    dim_auxiliary_output: int, default=5 (> 0)
        Dimension of auxiliary output.

    dim_confounded_auxiliary_output: int, default=0 (>= 0)
        Dimension of the confounder in the auxiliary output.

    query_coefficient_matrix: torch.Tensor, shape (dim_query, dim_auxiliary_output), default=None.
        Coefficient vector of query.

    action_coefficient_matrix: torch.Tensor, shape (dim_action_embedding, dim_auxiliary_output), default=None.
        Coefficient vector of action embedding.

    noise_level: float = 0.0
        Noise level on auxiliary output.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    dim_query: int = 5
    dim_action_embedding: int = 5
    dim_auxiliary_output: int = 5
    dim_confounded_auxiliary_output: Optional[int] = None
    query_coefficient_matrix: Optional[torch.Tensor] = None
    action_coefficient_matrix: Optional[torch.Tensor] = None
    max_abs_val: float = 5.0
    scaler: float = 1.0
    noise_level: float = 0.0
    confounder_scale: float = 1.0
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.random_state is not None:
            torch_seed(self.random_state, device=self.device)

        if self.dim_confounded_auxiliary_output is None:
            self.dim_confounded_auxiliary_output = 0
        elif self.dim_confounded_auxiliary_output > self.dim_auxiliary_output:
            raise ValueError(
                "dim_confounded_auxiliary_output must be equal to or smaller than dim_auxiliary_output, but found False"
            )

        if self.query_coefficient_matrix is None:
            self.query_coefficient_matrix = torch.randn(
                (self.dim_query, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.query_coefficient_matrix,
                name="query_coefficient_matrix",
                expected_dim=2,
            )
            if self.query_coefficient_matrix.shape != (
                self.dim_query,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of query_coefficient_matrix must be (dim_query, dim_auxiliary_output), but found False"
                )

        if self.action_coefficient_matrix is None:
            self.action_coefficient_matrix = torch.randn(
                (self.dim_action_embedding, self.dim_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.action_coefficient_matrix,
                name="action_coefficient_matrix",
                expected_dim=2,
            )
            if self.action_coefficient_matrix.shape != (
                self.dim_action_embedding,
                self.dim_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of action_coefficient_matrix must be (dim_action_embedding, dim_auxiliary_output), but found False"
                )

    def _sample_auxiliary_output(
        self,
        query: torch.Tensor,
        action_embedding: torch.Tensor,
    ):
        """Sample auxiliary output.

        Parameters
        -------
        query: torch.Tensor, shape (n_samples, dim_query)
            Query.

        action_embedding: torch.Tensor, shape (n_samples, dim_action_embedding)
            Action embedding.

        Return
        -------
        auxiliary_output: torch.Tensor, shape (n_samples, dim_auxiliary_output)
            Auxiliary output.

        """
        n_samples = len(query)

        x = (action_embedding @ self.action_coefficient_matrix) * self.scaler
        non_linear_mapping = 25 * x / (1 + torch.exp(x) + torch.exp(-x))
        output = query @ self.query_coefficient_matrix + non_linear_mapping

        if self.dim_confounded_auxiliary_output > 0:
            output[:, -self.dim_confounded_auxiliary_output :] = torch.normal(
                mean=torch.zeros(
                    (n_samples, self.dim_confounded_auxiliary_output),
                    device=self.device,
                ),
                std=torch.full(
                    (n_samples, self.dim_confounded_auxiliary_output),
                    self.confounder_scale,
                    device=self.device,
                ),
            )

        noise = torch.normal(
            mean=output,
            std=torch.full(
                (n_samples, self.dim_auxiliary_output),
                self.noise_level,
                device=self.device,
            ),
        )

        return output + noise


@dataclass
class SparseRewardSimulator:
    """Simulator of expected reward.

    Parameters
    -------
    dim_context: int, default=5 (> 0)
        Dimension of context.

    dim_query: int, default=5 (> 0)
        Dimension of query.

    dim_auxiliary_output: int, default=5 (> 0)
        Dimension of auxiliary output.

    context_coefficient_matrix: torch.Tensor, shape (dim_context, dim_auxiliary_output), default=None.
        Coefficient vector of context.

    query_coefficient_matrix: torch.Tensor, shape (dim_query, dim_auxiliary_output), default=None.
        Coefficient vector of query.

    normalizer: float, default=None.
        Normalization term of expected reward.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    dim_context: int = 5
    dim_query: int = 5
    dim_auxiliary_output: int = 5
    dim_sparse_auxiliary_output: Optional[int] = None
    context_coefficient_matrix: Optional[torch.Tensor] = None
    query_coefficient_matrix: Optional[torch.Tensor] = None
    normalizer: Optional[float] = None
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.random_state is not None:
            torch_seed(self.random_state)

        if self.dim_sparse_auxiliary_output is None:
            self.dim_sparse_auxiliary_output = self.dim_auxiliary_output
        elif self.dim_sparse_auxiliary_output > self.dim_auxiliary_output:
            raise ValueError(
                "dim_sparse_auxiliary_output must be equal to or smaller than dim_auxiliary_output, but found False"
            )

        if self.context_coefficient_matrix is None:
            self.context_coefficient_matrix = torch.randn(
                (self.dim_context, self.dim_sparse_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.context_coefficient_matrix,
                name="context_coefficient_matrix",
                expected_dim=2,
            )
            if self.context_coefficient_matrix.shape != (
                self.dim_context,
                self.dim_sparse_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of context_coefficient_matrix must be (dim_context, dim_sparse_auxiliary_output), but found False"
                )

        if self.query_coefficient_matrix is None:
            self.query_coefficient_matrix = torch.randn(
                (self.dim_query, self.dim_sparse_auxiliary_output),
                device=self.device,
            )
        else:
            check_tensor(
                self.query_coefficient_matrix,
                name="query_coefficient_matrix",
                expected_dim=2,
            )
            if self.query_coefficient_matrix.shape != (
                self.dim_query,
                self.dim_sparse_auxiliary_output,
            ):
                raise ValueError(
                    "The shape of query_coefficient_matrix must be (dim_query, dim_sparse_auxiliary_output), but found False"
                )

        if self.normalizer is None:
            self.normalizer = (
                (self.dim_context + self.dim_query)
                * self.dim_sparse_auxiliary_output
                * 5
            )

    def calc_expected_reward(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        auxiliary_output: torch.Tensor,
    ):
        """Sample auxiliary output.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            Context.

        query: torch.Tensor, shape (n_samples, dim_query)
            Query.

        auxiliary_output: torch.Tensor, shape (n_samples, dim_auxiliary_output)
            Auxiliary output.

        Return
        -------
        expected_reward: torch.Tensor, shape (n_samples, )
            Expected reward of given auxiliary output on given context.

        """
        n_samples = len(context)
        context = context[:, None, :]
        query = query[:, None, :]
        auxiliary_output = auxiliary_output[:, : self.dim_sparse_auxiliary_output, None]

        context_coefficient_matrix = torch.tile(
            self.context_coefficient_matrix, (n_samples, 1, 1)
        )
        query_coefficient_matrix = torch.tile(
            self.query_coefficient_matrix, (n_samples, 1, 1)
        )

        coefficient = torch.bmm(context, context_coefficient_matrix) + torch.bmm(
            query, query_coefficient_matrix
        )
        expected_reward = torch.bmm(coefficient, auxiliary_output).squeeze()
        expected_reward = expected_reward / self.normalizer
        return expected_reward
