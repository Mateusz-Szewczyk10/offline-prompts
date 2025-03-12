"""Implementation of models."""
from typing import Optional, Union, Callable

import torch
from torch import nn
from torch.nn import functional as F

from sentence_transformers.util import cos_sim
from sklearn.cluster import KMeans

from ..dataset.function import AuxiliaryOutputGenerator
from .base import (
    BasePolicy,
    BaseActionPolicyModel,
    BaseClusterPolicyModel,
    BaseOutputRewardModel,
    BaseActionRewardModel,
    BaseKernelMarginalDensityModel,
    BaseClusteringModel,
)
from ..utils import torch_seed, gaussian_kernel


Policy = Union[BasePolicy, BaseActionPolicyModel, BaseClusterPolicyModel]


class NeuralActionPolicy(BaseActionPolicyModel):
    """NN based policy to choose discrete actions.

    Bases: :class:`src.policy.BaseActionPolicyModel`

    Imported as: :class:`src.policy.NeuralActionPolicy`

    Parameters
    -------
    n_actions: int
        Number of discrete actions.

    dim_context: int
        Dimension of the context.

    dim_query: int
        Dimension of the query.

    linear_hidden_dim: int, default=100.
        Dimension of hidden layer for MLP.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    def __init__(
        self,
        n_actions: int,
        dim_context: int,
        dim_query: int,
        linear_hidden_dim: int = 100,
        device: str = "cuda:0",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.device = device

        if random_state is not None:
            torch_seed(random_state, device=self.device)

        self.l1 = nn.Linear(dim_context + dim_query, linear_hidden_dim).to(self.device)
        self.l2 = nn.Linear(linear_hidden_dim, n_actions).to(self.device)

    def forward(self, inputs: torch.Tensor):
        """Produce logit values using a Transformer model."""
        x = F.relu(self.l1(inputs))
        output = self.l2(x)
        return output


class NeuralClusterPolicy(BaseClusterPolicyModel):
    """NN based policy to choose discrete clusters.

    Bases: :class:`src.policy.BaseClusterPolicyModel`

    Imported as: :class:`src.policy.NeuralClusterPolicy`

    Parameters
    -------
    n_actions: int
        Number of clusters. (referred to `n_actions` due to API consistency)

    dim_context: int
        Dimension of the context.

    dim_query: int
        Dimension of the query.

    dim_feature_emb: int
        Dimension of feature vector of cluster centers.

    linear_hidden_dim: int, default=100.
        Dimension of hidden layer for MLP.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    def __init__(
        self,
        n_actions: int,
        dim_context: int,
        dim_query: int,
        dim_feature_emb: int,
        linear_hidden_dim: int = 100,
        device: str = "cuda:0",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.device = device

        if random_state is not None:
            torch_seed(random_state, device=self.device)

        self.l1 = nn.Linear(
            dim_context + dim_query + dim_feature_emb,
            linear_hidden_dim,
        ).to(self.device)
        self.l2 = nn.Linear(linear_hidden_dim, 1).to(self.device)

    def forward(self, inputs: torch.Tensor, cluster_centers: torch.Tensor):
        """Produce logit values using a Transformer model."""
        n_samples = len(inputs)
        output = torch.zeros((n_samples, self.n_actions), device=self.device)

        for k in range(self.n_actions):
            x = torch.cat((inputs, cluster_centers[:, k]), dim=1)
            x = F.relu(self.l1(x))
            output[:, k] = self.l2(x).squeeze()

        return output


class NeuralOutputRewardPredictor(BaseOutputRewardModel):
    """NN based reward predictor based on auxiliary output.

    Bases: :class:`src.policy.BaseOutputRewardPredictor`

    Imported as: :class:`src.policy.NeuralOutputRewardPredictor`

    Parameters
    -------
    dim_context: int
        Dimension of the context.

    dim_query: int
        Dimension of the query.

    dim_auxiliary_output: int
        Dimension of the auxiliary output.

    linear_hidden_dim: int, default=100.
        Dimension of hidden layer for MLP.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    def __init__(
        self,
        dim_context: int,
        dim_query: int,
        dim_auxiliary_output: int,
        linear_hidden_dim: int = 100,
        device: str = "cuda:0",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.device = device

        if random_state is not None:
            torch_seed(random_state, device=self.device)

        self.l1 = nn.Linear(
            dim_context + dim_query + dim_auxiliary_output, linear_hidden_dim
        ).to(self.device)
        self.l2 = nn.Linear(linear_hidden_dim, 1).to(self.device)

    def forward(self, inputs: torch.Tensor):
        """Produce logit values using a Transformer model."""
        x = F.relu(self.l1(inputs))
        output = self.l2(x)
        return output.squeeze()


class NeuralActionRewardPredictor(BaseActionRewardModel):
    """NN based reward predictor based on action.

    Bases: :class:`src.policy.BaseActionRewardModel`

    Imported as: :class:`src.policy.NeuralActionRewardPredictor`

    Parameters
    -------
    action_list: torch.Tensor
        Mapping from action id to its embeddings.

    dim_context: int
        Dimension of the context.

    dim_query: int
        Dimension of the query.

    linear_hidden_dim: int, default=100.
        Dimension of hidden layer for MLP.

    device: str, default="cuda:0"
        Device.

    random_state: Optional[int] = None
        Random state.

    """

    def __init__(
        self,
        action_list: torch.Tensor,
        dim_context: int,
        dim_query: int,
        linear_hidden_dim: int = 100,
        device: str = "cuda:0",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.device = device
        self.action_list = action_list
        self.n_actions = len(action_list)
        dim_action = action_list.shape[1]

        if random_state is not None:
            torch_seed(random_state, device=self.device)

        self.l1 = nn.Linear(dim_context + dim_query + dim_action, linear_hidden_dim).to(
            self.device
        )
        self.l2 = nn.Linear(linear_hidden_dim, 1).to(self.device)

    def forward(self, inputs: torch.Tensor):
        """Produce logit values using a Transformer model."""
        x = F.relu(self.l1(inputs))
        output = self.l2(x)
        return output.squeeze()


class NeuralMarginalDensityEstimator(BaseKernelMarginalDensityModel):
    """L2 norm-based kernel.

    Bases: :class:`src.policy.BaseKernelMarginalDensityModel

    Imported as: :class:`src.policy.NeuralMarginalDensityEstimator

    Parameters
    -------
    dim_context: torch.Tensor
        Dimension of context.

    dim_context: int
        Dimension of the context.

    action_list: torch.Tensor
        Mapping from action id to its embeddinds.

    auxiliary_output_generator: AuxiliaryOutputGenerator
        Auxiliary output generator.

    linear_hidden_dim: int, default=100.
        Dimension of hidden layer for MLP.

    kernel_function: Callable, default=gaussian_kernel
        Kernel function to use.

    kernel_kwargs: dict, default={"tau": 1.0}
        Kwargs of the kernel function.

    emb_noise: float, default=0.0
        Magnitude of the explicit noise of the output embedding. (for ablation)

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    def __init__(
        self,
        dim_context: int,
        dim_query: int,
        action_list: torch.Tensor,
        auxiliary_output_generator: AuxiliaryOutputGenerator,
        linear_hidden_dim: int = 100,
        kernel_function: Callable = gaussian_kernel,
        kernel_kwargs: dict = {"tau": 1.0},
        emb_noise: float = 0.0,
        device: str = "cuda:0",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.action_list = action_list
        self.dim_auxiliary_output = auxiliary_output_generator.dim_auxiliary_output
        self.auxiliary_output_generator = auxiliary_output_generator
        self.kernel_function = kernel_function
        self.kernel_kwargs = kernel_kwargs
        self.emb_noise = emb_noise
        self.device = device

        if random_state is not None:
            torch_seed(random_state, device=self.device)

        self.l1 = nn.Linear(
            dim_context + dim_query + self.dim_auxiliary_output, linear_hidden_dim
        ).to(self.device)
        self.l2 = nn.Linear(linear_hidden_dim, 1).to(self.device)

    def forward(self, inputs: torch.Tensor):
        """Calculate some logit values.

        Parameters
        -------
        inputs: torch.Tensor, shape (n_samples, dim_context + dim_query + dim_auxiliary_output)
            Input vectors.

        Return
        -------
        marginal_density: torch.Tensor (n_samples, )
            Predicted value of the given action.

        """
        x = F.relu(self.l1(inputs))
        output = F.softplus(self.l2(x), beta=0.5) + 1e-10
        return output.squeeze()

    def calc_pairwise_distance(
        self,
        pivot_output: torch.Tensor,
        sampled_outputs: torch.Tensor,
    ):
        """Calculate pairwise distance between pivot output and sampled outputs.

        Parameters
        -------
        pivot_output: torch.Tensor, shape (n_samples, dim_auxiliary_output)
            Pivot output observed in the logged data.

        sampled_outputs: torch.Tensor, shape (n_samples, dim_auxiliary_output) or (n_samples, n_samples_to_approximate, dim_auxiliary_output)
            Sampled outputs that are used to calculate the marginalized density.

        Return
        -------
        pairwise_distance: torch.Tensor, shape (n_samples, n_samples_to_approximate)
            Pairwise distance between the pivot output and sampled outputs.

        """
        dim_output = pivot_output.shape[-1]

        # adding noise for ablation
        pivot_output = torch.normal(
            mean=pivot_output, std=torch.full_like(pivot_output, self.emb_noise)
        )
        sampled_outputs = torch.normal(
            mean=sampled_outputs, std=torch.full_like(sampled_outputs, self.emb_noise)
        )

        if sampled_outputs.ndim == 2:
            squared_distance = torch.sum((pivot_output - sampled_outputs) ** 2, dim=-1)
        else:
            squared_distance = torch.sum(
                (pivot_output.unsqueeze(1) - sampled_outputs) ** 2, dim=-1
            )

        pairwise_distance = torch.sqrt(squared_distance) / dim_output
        return pairwise_distance


class KmeansActionClustering(BaseClusteringModel):
    """Similarity-based clustering.

    Bases: :class:`src.policy.BaseClusteringModel`

    Imported as: :class:`src.policy.KmeansActionClustering`

    Parameters
    -------
    n_clusters: int
        Number of clusters.

    action_list: torch.Tensor
        Mapping from action id to its embeddings.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    def __init__(
        self,
        n_clusters: int,
        action_list: torch.Tensor,
        device: str = "cuda:0",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.device = device

        self.n_clusters = n_clusters
        self.n_actions = len(action_list)
        self.action_list = action_list

        if random_state is None:
            raise ValueError("random_state must be given")
        else:
            torch_seed(random_state, device=self.device)

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_ids = kmeans.fit_predict(self.action_list.cpu())
        self._cluster_ids = torch.LongTensor(cluster_ids).to(self.device)
        self._cluster_centers = torch.FloatTensor(kmeans.cluster_centers_).to(
            self.device
        )

    def sample_clustering(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        **kwargs,
    ):
        """Sample clustering centers and assignment of each action to clusters.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            For API consistency.

        query: torch.Tensor, shape (n_samples, dim_query)
            Fpr API consistency.

        """
        n_samples = len(context)
        self.cluster_ids = torch.tile(self._cluster_ids, (n_samples, 1))
        self.cluster_centers = torch.tile(self._cluster_centers, (n_samples, 1, 1))

    def retrieve_cluster(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        action: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        resample_clustering: bool = False,
        **kwargs,
    ):
        """Retrieve cluster id given action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            For API consistency.

        query: torch.Tensor, shape (n_samples, dim_query)
            For API consistency.

        action: torch.Tensor, shape (n_samples, )
            Action chosen by the logging policy.

        idx: torch.Tensor, shape (n_subsamples, ), default=None
            Index for subsamples.

        resample_clustering: bool, default=False
            For API consistency.

        Return
        -------
        cluster: torch.Tensor, shape (n_samples, )
            Index of the cluster the given action belongs to.

        """
        if idx is not None:
            action = action[idx]

        return self._cluster_ids[action]
