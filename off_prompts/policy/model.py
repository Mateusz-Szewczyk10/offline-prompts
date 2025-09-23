"""Implementation of models."""
from typing import Any, Union, Optional, Dict, List, Callable

import torch
from torch import nn

from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedModel,
)
from transformers import (
    AutoTokenizer,
    AutoModel,
)
from sklearn.cluster import KMeans

from .base import (
    BasePolicy,
    BasePromptPolicyModel,
    BaseClusterPolicyModel,
    BaseSentenceRewardModel,
    BasePromptRewardModel,
    BaseKernelMarginalDensityModel,
    BaseClusteringModel,
)
from .policy import UniformRandomPolicy
from ..dataset.base import BaseFrozenLLM, BaseEncoder
from ..dataset.encoder import TransformerEncoder
from ..types import Sentence, Tokens
from ..utils import torch_seed, tokenize, to_device, gaussian_kernel


Policy = Union[BasePolicy, BasePromptPolicyModel, BaseClusterPolicyModel]


class PromptPolicy(BasePromptPolicyModel):
    """Policy to choose discrete actions.

    Bases: :class:`off_prompts.policy.BasePromptPolicyModel`

    Imported as: :class:`off_prompts.policy.PromptPolicy`

    Parameters
    -------
    n_actions: int
        Number of discrete prompts.

    dim_context: int
        Dimensions of context.

    query_encoder: BaseEncoder
        Encoder of query.

    linear_hidden_dim: int, default=100
        Dimension of hidden layer for MLP.

    device: str, default="cuda"
        Device.

    random_state: int, default=None
        Random state.

    """

    def __init__(
        self,
        n_actions: int,
        dim_context: int,
        query_encoder: BaseEncoder,
        linear_hidden_dim: int = 100,
        device: str = "cuda",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.device = device

        if random_state is not None:
            torch_seed(random_state, device=self.device)

        if query_encoder.device != self.device:
            raise ValueError(
                "query_encoder.device and device must be the same, but found False."
            )
        if (
            isinstance(query_encoder, TransformerEncoder)
            and query_encoder.pca_matrix is None
        ):
            raise ValueError(
                "query_encoder must be fitted in advance, but found False. Please call `query_encoder.fit_pca` first."
            )

        self.query_encoder = query_encoder
        self.l1 = nn.Linear(dim_context + query_encoder.dim_emb, linear_hidden_dim).to(
            self.device
        )
        self.l2 = nn.Linear(linear_hidden_dim, n_actions).to(self.device)
        self.relu = nn.ReLU()

    def forward(
        self, context: torch.Tensor, query: Union[Sentence, Tokens, torch.Tensor]
    ):
        """Produce logit values using a Transformer model."""
        if not isinstance(query, torch.Tensor):
            query = self.query_encoder.encode(query)

        context = to_device(context, device=self.device)
        query = to_device(query, device=self.device)

        x = torch.cat((context, query), dim=1)
        x = self.relu(self.l1(x))
        output = self.l2(x)
        return output


class ClusterPolicy(BaseClusterPolicyModel):
    """Transformer based policy to choose discrete clusters.

    Bases: :class:`off_prompts.policy.BaseClusterPolicyModel`

    Imported as: :class:`off_prompts.policy.ClusterPolicy`

    Parameters
    -------
    n_actions: int
        Number of clusters. (referred to `n_actions` due to API consistency)

    dim_context: int
        Dimensions of context.

    query_encoder: BaseEncoder
        Encoder of query.

    cluster_center_encoder: BaseEncoder, default=None
        Encoder of cluster center (either prompt or sentence).

    linear_hidden_dim: int, default=100
        Dimension of hidden layer for MLP.

    device: str, default="cuda"
        Device.

    random_state: int, default=None
        Random state.

    """

    def __init__(
        self,
        n_actions: int,
        dim_context: int,
        query_encoder: BaseEncoder,
        cluster_center_encoder: BaseEncoder,
        linear_hidden_dim: int = 100,
        device: str = "cuda",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.device = device

        if random_state is not None:
            torch_seed(random_state, device=self.device)

        if query_encoder.device != self.device:
            raise ValueError(
                "query_encoder.device and device must be the same, but found False."
            )
        if (
            isinstance(query_encoder, TransformerEncoder)
            and query_encoder.pca_matrix is None
        ):
            raise ValueError(
                "query_encoder must be fitted in advance, but found False. Please call `query_encoder.fit_pca` first."
            )

        if cluster_center_encoder.device != self.device:
            raise ValueError(
                "cluster_center_encoder.device and device must be the same, but found False."
            )
        if (
            isinstance(cluster_center_encoder, TransformerEncoder)
            and cluster_center_encoder.pca_matrix is None
        ):
            raise ValueError(
                "cluster_center_encoder must be fitted in advance, but found False. Please call `cluster_center_encoder.fit_pca` first."
            )

        self.query_encoder = query_encoder
        self.cluster_center_encoder = cluster_center_encoder
        self.l1 = nn.Linear(
            dim_context + query_encoder.dim_emb + cluster_center_encoder.dim_emb,
            linear_hidden_dim,
        ).to(self.device)
        self.l2 = nn.Linear(linear_hidden_dim, 1).to(self.device)
        self.relu = nn.ReLU()

    def forward(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        cluster_centers: torch.Tensor,
    ):
        """Produce logit values using a Transformer model."""
        if not isinstance(query, torch.Tensor):
            query = self.query_encoder.encode(query)

        context = to_device(context, device=self.device)
        query = to_device(query, device=self.device)
        cluster_centers = to_device(cluster_centers, device=self.device)

        inputs = torch.cat((context, query), dim=1)

        n_samples = len(inputs)
        output = torch.zeros((n_samples, self.n_actions), device=self.device)

        for k in range(self.n_actions):
            x = torch.cat((inputs, cluster_centers[:, k]), dim=1)
            x = self.relu(self.l1(x))
            output[:, k] = self.l2(x).squeeze()

        return output


class SentenceRewardPredictor(BaseSentenceRewardModel):
    """Reward predictor for output sentence.

    Bases: :class:`off_prompts.policy.BaseSentenceRewardModel`

    Imported as: :class:`off_prompts.policy.SentenceRewardPredictor`

    Parameters
    -------
    dim_context: int
        Dimensions of context.

    frozen_llm: BaseFrozenLLM
        Frozen LLM.

    query_encoder: BaseEncoder
        Encoder of query.

    sentence_encoder: BaseEncoder
        Encoder of sentence.

    linear_hidden_dim: int, default=100
        Dimension of hidden layer for MLP.

    device: str, default="cuda"
        Device.

    random_state: int, default=None
        Random state.

    """

    def __init__(
        self,
        dim_context: int,
        frozen_llm: BaseFrozenLLM,
        query_encoder: BaseEncoder,
        sentence_encoder: BaseEncoder,
        linear_hidden_dim: int = 100,
        device: str = "cuda",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.frozen_llm = frozen_llm
        self.device = device

        if random_state is not None:
            torch_seed(random_state, device=self.device)

        if query_encoder.device != self.device:
            raise ValueError(
                "query_encoder.device and device must be the same, but found False."
            )
        if (
            isinstance(query_encoder, TransformerEncoder)
            and query_encoder.pca_matrix is None
        ):
            raise ValueError(
                "query_encoder must be fitted in advance, but found False. Please call `sentence_encoder.fit_pca` first."
            )

        if sentence_encoder.device != self.device:
            raise ValueError(
                "sentence_encoder.device and device must be the same, but found False."
            )
        if (
            isinstance(sentence_encoder, TransformerEncoder)
            and sentence_encoder.pca_matrix is None
        ):
            raise ValueError(
                "sentence_encoder must be fitted in advance, but found False. Please call `sentence_encoder.fit_pca` first."
            )

        self.query_encoder = query_encoder
        self.sentence_encoder = sentence_encoder
        self.l1 = nn.Linear(
            dim_context + query_encoder.dim_emb + sentence_encoder.dim_emb,
            linear_hidden_dim,
        ).to(self.device)
        self.l2 = nn.Linear(linear_hidden_dim, 1).to(self.device)
        self.relu = nn.ReLU()

    def forward(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        sentence: Union[Sentence, Tokens, torch.Tensor],
    ):
        """Produce logit values using a Transformer model."""
        if not isinstance(query, torch.Tensor):
            query = self.query_encoder.encode(query)

        if not isinstance(sentence, torch.Tensor):
            sentence = self.sentence_encoder.encode(sentence, context, query)

        context = to_device(context, device=self.device)
        query = to_device(query, device=self.device)
        sentence = to_device(sentence, device=self.device)

        x = torch.cat((context, query, sentence), dim=1)
        x = self.relu(self.l1(x))
        output = self.l2(x)
        return output.squeeze()


class PromptRewardPredictor(BasePromptRewardModel):
    """Reward predictor for prompt.

    Bases: :class:`off_prompts.policy.BasePromptRewardModel`

    Imported as: :class:`off_prompts.policy.PromptRewardPredictor`

    Parameters
    -------
    dim_context: int
        Dimensions of context.

    action_list: Sentence
        Mapping from action id to its embeddinds.

    query_encoder: BaseEncoder
        Encoder of query.

    prompt_encoder: BaseEncoder
        Encoder of prompt.

    linear_hidden_dim: int, default=100
        Dimension of hidden layer for MLP.

    device: str, default="cuda"
        Device.

    random_state: int, default=None
        Random state.

    """

    def __init__(
        self,
        dim_context: int,
        action_list: List[str],
        query_encoder: BaseEncoder,
        prompt_encoder: BaseEncoder,
        linear_hidden_dim: int = 100,
        device: str = "cuda",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.action_list = action_list
        self.n_actions = len(action_list)
        self.device = device

        if random_state is not None:
            torch_seed(random_state, device=self.device)

        if query_encoder.device != self.device:
            raise ValueError(
                "query_encoder.device and device must be the same, but found False."
            )
        if (
            isinstance(query_encoder, TransformerEncoder)
            and query_encoder.pca_matrix is None
        ):
            raise ValueError(
                "query_encoder must be fitted in advance, but found False. Please call `query_encoder.fit_pca` first."
            )

        if prompt_encoder.device != self.device:
            raise ValueError(
                "prompt_encoder.device and device must be the same, but found False."
            )
        if (
            isinstance(prompt_encoder, TransformerEncoder)
            and prompt_encoder.pca_matrix is None
        ):
            raise ValueError(
                "prompt_encoder must be fitted in advance, but found False. Please call `sentence_encoder.fit_pca` first."
            )

        self.prompt_embeddings = prompt_encoder.encode(action_list)

        self.query_encoder = query_encoder
        self.prompt_encoder = prompt_encoder
        self.l1 = nn.Linear(
            dim_context + query_encoder.dim_emb + prompt_encoder.dim_emb,
            linear_hidden_dim,
        ).to(self.device)
        self.l2 = nn.Linear(linear_hidden_dim, 1).to(self.device)
        self.relu = nn.ReLU()

    def forward(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        prompt: Union[Sentence, Tokens, torch.Tensor],
    ):
        """Produce logit values using a Transformer model."""
        if not isinstance(query, torch.Tensor):
            query = self.query_encoder.encode(query)

        if not isinstance(prompt, torch.Tensor):
            prompt = self.prompt_encoder.encode(prompt)

        context = to_device(context, device=self.device)
        query = to_device(query, device=self.device)
        prompt = to_device(prompt, device=self.device)

        x = torch.cat((context, query, prompt), dim=1)
        x = self.relu(self.l1(x))
        output = self.l2(x)
        return output.squeeze()


class KernelMarginalDensityEstimator(BaseKernelMarginalDensityModel):
    """L2 norm-based kernel.

    Bases: :class:`off_prompts.policy.BaseKernelMarginalDensityModel

    Imported as: :class:`off_prompts.policy.KernelMarginalDensityEstimator

    Parameters
    -------
    dim_context: torch.Tensor
        Dimension of context.

    action_list: Sentence
        Mapping from action id to its embeddinds.

    frozen_llm: BaseFrozenLLM
        Frozen LLM to generate sentence.

    query_encoder: BaseEncoder
        Encoder of query.

    sentence_encoder: BaseEncoder
        Encoder of sentence.

    linear_hidden_dim: int, default=100
        Dimension of hidden layer for MLP.

    kernel_function: Callable, default=gaussian_kernel
        Kernel function to use.

    kernel_kwargs: dict, default={"tau": 1.0}
        Kwargs of the kernel function.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    def __init__(
        self,
        dim_context: int,
        action_list: List[str],
        frozen_llm: BaseFrozenLLM,
        query_encoder: BaseEncoder,
        sentence_encoder: BaseEncoder,
        linear_hidden_dim: int = 100,
        kernel_function: Callable = gaussian_kernel,
        kernel_kwargs: dict = {"tau": 1.0},
        device: str = "cuda:0",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.action_list = action_list
        self.frozen_llm = frozen_llm
        self.kernel_function = kernel_function
        self.kernel_kwargs = kernel_kwargs
        self.device = device

        if random_state is not None:
            torch_seed(random_state, device=self.device)

        if query_encoder.device != self.device:
            raise ValueError(
                "query_encoder.device and device must be the same, but found False."
            )
        if (
            isinstance(query_encoder, TransformerEncoder)
            and query_encoder.pca_matrix is None
        ):
            raise ValueError(
                "query_encoder must be fitted in advance, but found False. Please call `sentence_encoder.fit_pca` first."
            )

        if sentence_encoder.device != self.device:
            raise ValueError(
                "sentence_encoder.device and device must be the same, but found False."
            )
        if (
            isinstance(sentence_encoder, TransformerEncoder)
            and sentence_encoder.pca_matrix is None
        ):
            raise ValueError(
                "sentence_encoder must be fitted in advance, but found False. Please call `sentence_encoder.fit_pca` first."
            )

        self.query_encoder = query_encoder
        self.sentence_encoder = sentence_encoder
        self.l1 = nn.Linear(
            dim_context + self.query_encoder.dim_emb + self.sentence_encoder.dim_emb,
            linear_hidden_dim,
        ).to(self.device)
        self.l2 = nn.Linear(linear_hidden_dim, 1).to(self.device)
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus(beta=0.5)

    def forward(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        sentence: Union[Sentence, Tokens, torch.Tensor],
    ):
        """Calculate logit values."""
        if not isinstance(query, torch.Tensor):
            query = self.query_encoder.encode(query)

        if not isinstance(sentence, torch.Tensor):
            sentence = self.sentence_encoder.encode(sentence, context, query)

        context = to_device(context, device=self.device)
        query = to_device(query, device=self.device)
        sentence = to_device(sentence, device=self.device)

        x = torch.cat((context, query, sentence), dim=1)
        x = self.relu(self.l1(x))
        output = self.softplus(self.l2(x)) + 1e-10
        return output.squeeze()

    def calc_pairwise_distance(
        self,
        pivot_sentence: Union[Sentence, Tokens, torch.Tensor],
        sampled_sentences: Union[Sentence, Tokens, torch.Tensor],
        context: Optional[torch.Tensor] = None,
        query: Optional[Union[Sentence, Tokens, torch.Tensor]] = None,
    ):
        """Calculate pairwise distance between pivot output and sampled outputs.

        Parameters
        -------
        pivot_sentence: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, )
            Pivot output observed in the logged data.

        sampled_sentences: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, n_samles_to_approximate)
            Sampled outputs that are used to calculate the marginalized density.

        context: torch.Tensor, shape (n_samples, dim_context), default=None
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query), default=None
            Original keywords of input sentence specified by users.

        Return
        -------
        pairwise_distance: torch.Tensor, shape (n_samples, n_samples_to_approximate)
            Pairwise distance between the pivot output and sampled outputs.

        """
        if not isinstance(pivot_sentence, torch.Tensor):
            pivot_sentence = self.sentence_encoder.encode(
                pivot_sentence, context, query
            )

        if not isinstance(sampled_sentences, torch.Tensor):
            sampled_sentences = self.sentence_encoder.encode(
                sampled_sentences, context, query
            )

        dim_sentence = pivot_sentence.shape[-1]

        if sampled_sentences.ndim == 2:
            squared_distance = torch.sum(
                (pivot_sentence - sampled_sentences) ** 2, dim=-1
            )
        else:
            squared_distance = torch.sum(
                (pivot_sentence.unsqueeze(1) - sampled_sentences) ** 2, dim=-1
            )

        pairwise_distance = torch.sqrt(squared_distance) / dim_sentence
        return pairwise_distance


class KmeansPromptClustering(BaseClusteringModel):
    """Similarity-based clustering.

    Bases: :class:`off_prompts.policy.BaseClusteringModel`

    Imported as: :class:`off_prompts.policy.KmeansPromptClustering`

    Parameters
    -------
    n_clusters: int
        Number of clusters.

    action_list: Sentence
        Mapping from action id to prompt.

    prompt_encoder: BaseEncoder
        Encoder of prompt.

    device: str, default="cuda"
        Device.

    random_state: int, default=None
        Random state.

    """

    def __init__(
        self,
        n_clusters: int,
        action_list: List[str],
        prompt_encoder: BaseEncoder,
        device: str = "cuda",
        random_state: Optional[int] = None,
    ):
        super().__init__()

        self.n_clusters = n_clusters
        self.n_actions = len(action_list)
        self.action_list = action_list
        self.device = device

        if random_state is None:
            raise ValueError("random_state must be given")
        else:
            torch_seed(random_state, device=self.device)

        if prompt_encoder.device != self.device:
            raise ValueError(
                "prompt_encoder.device and device must be the same, but found False."
            )
        if (
            isinstance(prompt_encoder, TransformerEncoder)
            and prompt_encoder.pca_matrix is None
        ):
            raise ValueError(
                "prompt_encoder must be fitted in advance, but found False. Please call `prompt_encoder.fit_pca` first."
            )

        self.action_emb = prompt_encoder.encode(self.action_list)
        action_emb_ = self.action_emb.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=random_state).fit(
            action_emb_
        )

        self._cluster_ids = torch.LongTensor(kmeans.labels_).to(self.device)
        self._cluster_centers = torch.FloatTensor(kmeans.cluster_centers_).to(
            self.device
        )

    def sample_clustering(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        logging_predicted_reward: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Sample clustering centers and assignment of each action to clusters.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, )
            For API consistency.

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Fpr API consistency.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            For API consistency.

        """
        n_samples = len(context)
        self.cluster_ids = torch.tile(self._cluster_ids, (n_samples, 1))
        self.cluster_centers = torch.tile(self._cluster_centers, (n_samples, 1, 1))

    def retrieve_cluster(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        action: torch.Tensor,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        idx: Optional[torch.Tensor] = None,
        resample_clustering: bool = False,
        **kwargs,
    ):
        """Retrieve cluster id given action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, )
            For API consistency.

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Fpr API consistency.

        action: torch.Tensor, shape (n_samples, )
            Action chosen by the logging policy.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            For API consistency.

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
