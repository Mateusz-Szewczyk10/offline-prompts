"""Implementations of behavior and evaluation policies."""
from dataclasses import dataclass
from typing import Optional, Union, List
from operator import itemgetter

import torch
from torch.nn import functional as F
from sklearn.utils import check_scalar

from .base import (
    BasePolicy,
    BasePromptPolicyModel,
    BaseClusterPolicyModel,
    BasePromptRewardModel,
    BaseClusteringModel,
)
from ..types import Sentence, Tokens
from ..utils import torch_seed


Policy = Union[BasePolicy, BasePromptPolicyModel, BaseClusterPolicyModel]


@dataclass
class TwoStagePolicy(BasePolicy):
    """Two-stage policy.

    Parameters
    -------
    first_stage_policy: Policy
        First stage policy that chooses a cluster.

    second_stage_policy: Policy
        Second stage policy that chooses an action within the given cluster.

    clustering_policy: BaseClusteringModel
        Specifies a way to form a clustering in the action space.

    is_reward_based_first_stage_policy: bool, default=False
        Whether the first stage policy adopts a model-based approach.

    device: str, default="cuda"
        Device.

    """

    first_stage_policy: Policy
    second_stage_policy: Policy
    clustering_policy: BaseClusteringModel
    is_reward_based_first_stage_policy: bool = False
    device: str = "cuda"

    def __post_init__(self):
        self.action_list = self.second_stage_policy.action_list
        self.n_actions = len(self.action_list)

    def _calc_cluster_predicted_reward(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        clustering_logging_predicted_reward: Optional[torch.Tensor] = None,
        candidate_actions_all: Optional[List[List[torch.Tensor]]] = None,
        resample_clustering: bool = False,
    ):
        """Calculate the expectation of predicted reward within each cluster.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        clustering_logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions used for the logging policy in the clustering policy.

        candidate_actions_all: list of torch.Tensor, default=None
            Candidate actions for all clusters.

        resample_clustering: bool, default=False
            Whether to resample clustering or not.

        Return
        -------
        cluster_predicted_reward: torch.Tensor, shape (n_samples, n_clusters)
            Predicted reward of each cluster.

        """
        if not self.is_reward_based_first_stage_policy:
            cluster_predicted_reward = None

        else:
            if candidate_actions_all is None:
                candidate_actions_all = self.clustering_policy.retrieve_candidate_actions_for_all_clusters(
                    context=context,
                    query=query,
                    logging_predicted_reward=clustering_logging_predicted_reward,
                    resample_clustering=resample_clustering,
                )

            n_samples = len(candidate_actions_all)
            n_clusters = len(candidate_actions_all[0])

            cluster_predicted_reward = torch.zeros(
                (n_samples, n_clusters), device=self.device
            )
            for j in range(n_clusters):
                cluster_predicted_reward[
                    :, j
                ] = self.second_stage_policy.predict_policy_value(
                    context=context,
                    query=query,
                    predicted_reward=predicted_reward,
                    candidate_actions=[
                        candidate_actions_all[i][j] for i in range(n_samples)
                    ],
                )

        return cluster_predicted_reward

    def sample_multiple_actions(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        clustering_logging_predicted_reward: Optional[torch.Tensor] = None,
        return_action_type: str = "idx",
        n_actions_for_each: int = 1,
        replacement: bool = True,
        **kwargs,
    ):
        """Sample multiple actions for each context and action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        clustering_logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions used for the logging policy in the clustering policy.

        return_action_type: {"idx", "prompt"}, default="idx"
            Type of action to return.

        n_actions_for_each: int, default=1
            Number of actions to sample.

        replacement: bool, default=True
            Whether to draw with replacement or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions_for_each)
            Action (vector) indicating which prompt is sampled for each input sentence.

        """
        n_samples = len(context)

        action_choice_prob = self.calc_action_choice_probability(
            context=context,
            query=query,
            predicted_reward=predicted_reward,
            clustering_logging_predicted_reward=clustering_logging_predicted_reward,
        )
        action = torch.multinomial(
            action_choice_prob, num_samples=n_actions_for_each, replacement=replacement,
        )

        if return_action_type == "prompt":
            prompt = []
            for i in range(n_samples):
                prompt.append(list(itemgetter(*action[i])(self.action_list)))
            action = prompt

        return action

    def sample_action(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        clustering_logging_predicted_reward: Optional[torch.Tensor] = None,
        return_action_type: str = "idx",
        **kwargs,
    ):
        """Sample action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        clustering_logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions used for the logging policy in the clustering policy.

        return_action_type: {"idx", "onehot", "prompt"}, default="idx"
            Type of action to return.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions) or (n_samples, )
            (Onehot) action (vector) indicating which prompt is sampled for each input sentence.

        """
        n_samples = len(context)

        if self.is_reward_based_first_stage_policy:
            candidate_actions_all = self.clustering_policy.retrieve_candidate_actions_for_all_clusters(
                context=context,
                query=query,
                logging_predicted_reward=clustering_logging_predicted_reward,
                resample_clustering=True,
            )

            if predicted_reward is None:
                cluster_predicted_reward = None
            else:
                cluster_predicted_reward = self._calc_cluster_predicted_reward(
                    context=context,
                    query=query,
                    predicted_reward=predicted_reward,
                    clustering_logging_predicted_reward=clustering_logging_predicted_reward,
                    candidate_actions_all=candidate_actions_all,
                )

            cluster = self.first_stage_policy.sample_action(
                context=context, query=query, predicted_reward=cluster_predicted_reward,
            )

            candidate_actions = []
            for i in range(n_samples):
                candidate_actions.append(candidate_actions_all[i][cluster[i]])

        else:
            cluster_centers = self.clustering_policy.retrieve_cluster_centers(
                context=context,
                query=query,
                logging_predicted_reward=clustering_logging_predicted_reward,
                return_type="embedding",
                resample_clustering=True,
            )
            cluster = self.first_stage_policy.sample_action(
                context=context, query=query, cluster_centers=cluster_centers,
            )
            candidate_actions = self.clustering_policy.retrieve_candidate_actions(
                context=context,
                query=query,
                logging_predicted_reward=clustering_logging_predicted_reward,
                cluster=cluster,
            )

        action = self.second_stage_policy.sample_action(
            context=context,
            query=query,
            predicted_reward=predicted_reward,
            candidate_actions=candidate_actions,
            return_action_type=return_action_type,
        )
        return action

    def sample_action_and_output_prob(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        clustering_logging_predicted_reward: Optional[torch.Tensor] = None,
        return_action_type: str = "idx",
        is_log_prob: bool = False,
        **kwargs,
    ):
        """Sample an action and return the action choice probability of sampled action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        clustering_logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions used for the logging policy in the clustering policy.

        return_action_type: {"idx", "onehot", "prompt"}, default="idx"
            Type of action to return.

        is_log_prob: bool, default=False.
            Whether to return log probability or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions) or (n_samples, )
            (Onehot) action (vector) indicating which prompt is sampled for each input sentence.

        prob: torch.Tensor (n_samples, )
            (Log) probability of sampling the above action.

        """
        n_samples = len(context)

        if self.is_reward_based_first_stage_policy:
            candidate_actions_all = self.clustering_policy.retrieve_candidate_actions_for_all_clusters(
                context=context,
                query=query,
                logging_predicted_reward=clustering_logging_predicted_reward,
                resample_clustering=True,
            )
            cluster_predicted_reward = self._calc_cluster_predicted_reward(
                context=context,
                query=query,
                predicted_reward=predicted_reward,
                clustering_logging_predicted_reward=clustering_logging_predicted_reward,
                candidate_actions_all=candidate_actions_all,
            )
            (
                cluster,
                cluster_prob,
            ) = self.first_stage_policy.sample_action_and_output_prob(
                context=context,
                query=query,
                predicted_reward=cluster_predicted_reward,
                is_log_prob=is_log_prob,
            )

            candidate_actions = []
            for i in range(n_samples):
                candidate_actions.append(candidate_actions_all[i][cluster[i]])

        else:
            cluster_centers = self.clustering_policy.retrieve_cluster_centers(
                context=context,
                query=query,
                logging_predicted_reward=clustering_logging_predicted_reward,
                return_type="embedding",
                resample_clustering=True,
            )
            (
                cluster,
                cluster_prob,
            ) = self.first_stage_policy.sample_action_and_output_prob(
                context=context,
                query=query,
                cluster_centers=cluster_centers,
                is_log_prob=is_log_prob,
            )
            candidate_actions = self.clustering_policy.retrieve_candidate_actions(
                context=context,
                query=query,
                cluster=cluster,
                logging_predicted_reward=clustering_logging_predicted_reward,
            )

        action, action_prob = self.second_stage_policy.sample_action(
            context=context,
            query=query,
            candidate_actions=candidate_actions,
            return_action_type=return_action_type,
            is_log_prob=is_log_prob,
        )

        if is_log_prob:
            prob = cluster_prob + action_prob
        else:
            prob = cluster_prob * action_prob

        return action, prob

    def calc_action_choice_probability(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        clustering_logging_predicted_reward: Optional[torch.Tensor] = None,
        is_log_prob: bool = False,
        **kwargs,
    ):
        """Calculate the action choice probabilities.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, )
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        clustering_logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions used for the logging policy in the clustering policy.

        is_log_prob: bool, default=False.
            Whether to return log probability or not.

        Return
        -------
        action_choice_probability: torch.Tensor or List[torch.Tensor], shape (n_samples, n_candidate_actions[i])
            Action choice probability of all candidate actions.

        """
        candidate_actions_all = self.clustering_policy.retrieve_candidate_actions_for_all_clusters(
            context=context,
            query=query,
            logging_predicted_reward=clustering_logging_predicted_reward,
            resample_clustering=True,
        )
        cluster_centers = self.clustering_policy.retrieve_cluster_centers(
            context=context,
            query=query,
            logging_predicted_reward=clustering_logging_predicted_reward,
            return_type="embedding",
        )
        cluster_predicted_reward = self._calc_cluster_predicted_reward(
            context=context,
            query=query,
            predicted_reward=predicted_reward,
            clustering_logging_predicted_reward=clustering_logging_predicted_reward,
            candidate_actions_all=candidate_actions_all,
        )
        cluster_prob = self.first_stage_policy.calc_action_choice_probability(
            context=context,
            query=query,
            predicted_reward=cluster_predicted_reward,
            cluster_centers=cluster_centers,
            is_log_prob=is_log_prob,
        )

        n_samples = len(context)
        n_clusters = self.clustering_policy.n_clusters

        prob = torch.zeros((n_samples, self.n_actions), device=self.device)
        for cluster_ in range(n_clusters):
            candidate_actions_ = []
            for i in range(n_samples):
                candidate_actions_.append(candidate_actions_all[i][cluster_])

            action_prob_ = self.second_stage_policy.calc_action_choice_probability(
                context=context,
                query=query,
                predicted_reward=predicted_reward,
                candidate_actions=candidate_actions_,
                is_log_prob=is_log_prob,
            )

            for i in range(n_samples):
                if is_log_prob:
                    prob[i, candidate_actions_[i]] = (
                        cluster_prob[i, cluster_]
                        + action_prob_[i, candidate_actions_[i]]
                    )
                else:
                    prob[i, candidate_actions_[i]] = (
                        cluster_prob[i, cluster_]
                        * action_prob_[i, candidate_actions_[i]]
                    )

        return prob

    def calc_prob_given_action(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        action: torch.Tensor,
        predicted_reward: Optional[torch.Tensor] = None,
        clustering_logging_predicted_reward: Optional[torch.Tensor] = None,
        is_log_prob: bool = False,
        **kwargs,
    ):
        """Calculate the action choice probability of the given action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        action: torch.Tensor, shape (n_samples, )
             prompts (i.e., action).

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        clustering_logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions used for the logging policy in the clustering policy.

        is_log_prob: bool, default=False
            Whether to return log probability or not.

        Return
        -------
        action_choice_probability: torch.Tensor, shape (n_samples, )
            Action choice probability of the given action.

        """
        cluster = self.clustering_policy.retrieve_cluster(
            context=context,
            query=query,
            action=action,
            logging_predicted_reward=clustering_logging_predicted_reward,
            resample_clustering=True,
        )
        cluster_centers = self.clustering_policy.retrieve_cluster_centers(
            context=context,
            query=query,
            logging_predicted_reward=clustering_logging_predicted_reward,
            return_type="embedding",
        )
        candidate_actions = self.clustering_policy.retrieve_candidate_actions(
            context=context,
            query=query,
            cluster=cluster,
            logging_predicted_reward=clustering_logging_predicted_reward,
        )
        cluster_predicted_reward = self._calc_cluster_predicted_reward(
            context=context,
            query=query,
            predicted_reward=predicted_reward,
            clustering_logging_predicted_reward=clustering_logging_predicted_reward,
        )
        cluster_prob = self.first_stage_policy.calc_prob_given_action(
            context=context,
            query=query,
            predicted_reward=cluster_predicted_reward,
            cluster_centers=cluster_centers,
            action=cluster,
            is_log_prob=is_log_prob,
        )
        action_prob = self.second_stage_policy.calc_prob_given_action(
            context=context,
            query=query,
            action=action,
            predicted_reward=predicted_reward,
            candidate_actions=candidate_actions,
            is_log_prob=is_log_prob,
        )

        if is_log_prob:
            prob = cluster_prob + action_prob
        else:
            prob = cluster_prob * action_prob

        return prob

    def predict_policy_value(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        clustering_logging_predicted_reward: Optional[torch.Tensor] = None,
        reward_predictor: Optional[BasePromptRewardModel] = None,
        return_per_sample: bool = False,
        **kwargs,
    ):
        """Calculate policy value.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        clustering_logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions used for the logging policy in the clustering policy.

        reward_predictor: BasePromptRewardModel, default=None
            (Pre-trained) prompt reward predictor. This must be given when using logit-based model.

        return_per_sample: bool, default=False
            Whether to return policy value per sample or not (i.e., take mean).

        Return
        -------
        policy_value: float or torch.Tensor, shape (n_samples, )
            Predicted value of the two stage policy.

        """
        if predicted_reward is None and reward_predictor is None:
            raise ValueError(
                "Either predicted_reward or reward_predictor must be given."
            )

        if predicted_reward is None:
            predicted_reward = reward_predictor.predict_values(
                context=context, query=query,
            )

        candidate_actions_all = self.clustering_policy.retrieve_candidate_actions_for_all_clusters(
            context=context,
            query=query,
            logging_predicted_reward=clustering_logging_predicted_reward,
            resample_clustering=True,
        )
        cluster_centers = self.clustering_policy.retrieve_cluster_centers(
            context=context,
            query=query,
            return_type="embedding",
            logging_predicted_reward=clustering_logging_predicted_reward,
        )
        cluster_predicted_reward = self._calc_cluster_predicted_reward(
            context=context,
            query=query,
            predicted_reward=predicted_reward,
            clustering_logging_predicted_reward=clustering_logging_predicted_reward,
            candidate_actions_all=candidate_actions_all,
        )
        cluster_choice_probability = self.first_stage_policy.calc_action_choice_probability(
            context=context, query=query, cluster_centers=cluster_centers,
        )

        policy_value = (cluster_predicted_reward * cluster_choice_probability).sum(
            dim=1
        )
        policy_value = policy_value if return_per_sample else policy_value.mean()
        return policy_value


@dataclass
class SoftmaxPolicy(BasePolicy):
    """Softmax policy.

    Bases: :class:`off_prompts.policy.BasePolicy`

    Imported as: :class:`off_prompts.policy.SoftmaxPolicy`

    Parameters
    -------
    action_list: Sentence
        Mapping from action id to  prompts. Only applicable when using  prompts.

    is_first_stage_policy: bool, default=False
        Whether the policy is the first stage policy of the two stage policy.

    n_actions: int, default=None
        Number of clusters. (referred to `n_actions` due to API consistency)

    base_model: BasePolicyModel or BaseClusterPolicyModel or BasePromptRewardModel, default=None
        Base model to predict the logit value.

    beta: float, default=0.
        Inverse temperature hyperparameter.

    device: str, default = "cuda"
        Device.

    random_state: int, default=None
        Random state.

    """

    action_list: List[str]
    is_first_stage_policy: bool = False
    n_actions: Optional[int] = None
    base_model: Optional[
        Union[BasePromptPolicyModel, BaseClusterPolicyModel, BasePromptRewardModel]
    ] = None
    beta: float = 0.0
    device: str = "cuda"
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.is_first_stage_policy:
            if self.n_actions is None:
                raise ValueError(
                    "n_clusters must be given when is_first_stage_policy is True."
                )
        else:
            self.n_actions = len(self.action_list)

        if self.base_model is not None:
            self.given_reward_model = isinstance(self.base_model, BasePromptRewardModel)
        else:
            self.given_reward_model = False

        check_scalar(self.beta, "beta", target_type=float)

        if self.random_state is not None:
            torch_seed(self.random_state, device=self.device)

    def sample_multiple_actions(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        cluster_centers: Optional[torch.Tensor] = None,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        return_action_type: str = "idx",
        n_actions_for_each: int = 1,
        replacement: bool = True,
        **kwargs,
    ):
        """Sample multiple actions for each context and action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence, shape (n_samples, )
            Original keywords of input sentence specified by users.

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_cluster_emb), default=None
            Low-dimensional embeddings of the cluster center (sentence).

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        return_action_type: {"idx", "prompt"}, default="idx"
            Type of action to return.

        n_actions_for_each: int, default=1
            Number of actions to sample.

        replacement: bool, default=True
            Whether to draw with replacement or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions_for_each)
            Action (vector) indicating which prompt is sampled for each input sentence.

        """
        n_samples = len(context)

        action_choice_prob = self.calc_action_choice_probability(
            context=context,
            query=query,
            predicted_reward=predicted_reward,
            cluster_centers=cluster_centers,
            candidate_actions=candidate_actions,
        )
        action = torch.multinomial(
            action_choice_prob, num_samples=n_actions_for_each, replacement=replacement,
        )

        if return_action_type == "prompt":
            prompt = []
            for i in range(n_samples):
                prompt.append(list(itemgetter(*action[i])(self.action_list)))
            action = prompt  # (n_samples, n_actions_for_each)

        return action

    def sample_action(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        cluster_centers: Optional[torch.Tensor] = None,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        return_action_type: str = "idx",
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Sample action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_cluster_emb), default=None
            Low-dimensional embeddings of the cluster center (sentence).

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        return_action_type: {"idx", "onehot", "prompt"}, default="idx"
            Type of action to return.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions) or (n_samples, )
            (Onehot) action (vector) indicating which prompt is sampled for each input sentence.

        """
        n_samples = len(context)

        if self.base_model is None:
            if predicted_reward is None:
                raise ValueError(
                    "when base_model is not set, predicted_reward must be given."
                )
            x = predicted_reward

        elif self.given_reward_model:
            x = self.base_model.predict_values(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                calc_gradient=calc_gradient,
            )

        else:
            x = self.base_model.calc_logits(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                cluster_centers=cluster_centers,
                calc_gradient=calc_gradient,
            )

        if candidate_actions is None:
            x = x.to(self.device)
            action = F.gumbel_softmax(x * self.beta, hard=True).argmax(dim=1)

        elif type(x) == list:
            action = torch.zeros((n_samples,), dtype=int, device=self.device)
            for i in range(n_samples):
                x_ = x[i].to(self.device)
                action_id_ = F.gumbel_softmax(
                    x_.reshape((1, -1)) * self.beta, hard=True
                ).argmax(dim=1)
                action[i] = candidate_actions[i][action_id_]

        else:
            x = x.to(self.device)
            action = torch.zeros((n_samples,), dtype=int, device=self.device)
            for i in range(n_samples):
                candidate_actions_ = candidate_actions[i].cpu()

                action_id_ = F.gumbel_softmax(
                    x[i][candidate_actions_].reshape((1, -1)) * self.beta, hard=True
                ).argmax(dim=1)

                action[i] = candidate_actions[i][action_id_]

        if return_action_type == "onehot":
            action = F.one_hot(action, num_classes=self.n_actions)
        elif return_action_type == "prompt":
            action = list(itemgetter(*action)(self.action_list))

        return action

    def sample_action_and_output_prob(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        cluster_centers: Optional[torch.Tensor] = None,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        return_action_type: str = "idx",
        is_log_prob: bool = False,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Sample an action and return the action choice probability of sampled action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence, shape (n_samples, )
            Original keywords of input sentence specified by users.

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_cluster_emb), default=None
            Low-dimensional embeddings of the cluster center (sentence).

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        return_action_type: {"idx", "onehot", "prompt"}, default="idx"
            Type of action to return.

        is_log_prob: bool, default=False.
            Whether to return log probability or not.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions) or (n_samples, )
            (Onehot) action (vector) indicating which prompt is sampled for each input sentence.

        prob: torch.Tensor (n_samples, )
            (Log) probability of sampling the above action.

        """
        n_samples = len(context)

        if self.base_model is None:
            if predicted_reward is None:
                raise ValueError(
                    "when base_model is not set, predicted_reward must be given."
                )
            x = predicted_reward

        elif self.given_reward_model:
            x = self.base_model.predict_values(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                calc_gradient=calc_gradient,
            )

        else:
            x = self.base_model.calc_logits(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                cluster_centers=cluster_centers,
                calc_gradient=calc_gradient,
            )

        if candidate_actions is None:
            x = x.to(self.device)
            action_onehot = F.gumbel_softmax(x * self.beta, hard=True)
            log_prob = (F.log_softmax(x * self.beta) * action_onehot).sum(dim=1)
            action = action_onehot.argmax(dim=1)

        elif type(x) == list:
            action = torch.zeros((n_samples,), dtype=int, device=self.device)
            log_prob = torch.zeros((n_samples,), device=self.device)

            for i in range(n_samples):
                x_ = x[i].to(self.device)
                logit_ = x_.reshape((1, -1)) * self.beta
                action_onehot_ = F.gumbel_softmax(logit_, hard=True)
                log_prob_ = (F.log_softmax(logit_) * action_onehot).sum(dim=1)
                action_id_ = action_onehot_.argmax(dim=1)

                action[i] = candidate_actions[i][action_id_]
                log_prob[i] = log_prob_

        else:
            x = x.to(self.device)
            action = torch.zeros((n_samples,), dtype=int, device=self.device)
            log_prob = torch.zeros((n_samples,), device=self.device)

            for i in range(n_samples):
                candidate_actions_ = candidate_actions[i].cpu()
                logit_ = x[i][candidate_actions_].reshape((1, -1)) * self.beta
                action_onehot_ = F.gumbel_softmax(logit_, hard=True)
                log_prob_ = (F.log_softmax(logit_) * action_onehot).sum(dim=1)
                action_id_ = action_onehot_.argmax(dim=1)

                action[i] = candidate_actions[i][action_id_]
                log_prob[i] = log_prob_

        prob = log_prob if is_log_prob else torch.exp(log_prob)

        if return_action_type == "onehot":
            action = F.one_hot(action, num_classes=self.n_actions)
        elif return_action_type == "prompt":
            action = list(itemgetter(*action)(self.action_list))

        return action, prob

    def calc_action_choice_probability(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        cluster_centers: Optional[torch.Tensor] = None,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        is_log_prob: bool = False,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Calculate the action choice probabilities (within candidate actions).

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_cluster_emb), default=None
            Low-dimensional embeddings of the cluster center (sentence).

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        is_log_prob: bool, default=False.
            Whether to return log probability or not.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action_choice_probability: torch.Tensor, shape (n_samples, n_actions)
            Action choice probability of all candidate actions.

        """
        n_samples = len(context)

        if self.base_model is None:
            if predicted_reward is None:
                raise ValueError(
                    "when base_model is not set, predicted_reward must be given."
                )
            x = predicted_reward

        elif self.given_reward_model:
            x = self.base_model.predict_values(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                calc_gradient=calc_gradient,
            )

        else:
            x = self.base_model.calc_logits(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                calc_gradient=calc_gradient,
            )

        if candidate_actions is None:
            x = x.to(self.device)
            log_prob = F.log_softmax(x * self.beta)

        elif type(x) == list:
            log_prob = torch.full(
                (n_samples, self.n_actions), -torch.inf, device=self.device
            )
            for i in range(n_samples):
                x_ = x[i].to(self.device)
                log_prob_ = F.log_softmax(x_.reshape((1, -1)) * self.beta)[0]
                log_prob[i, candidate_actions[i]] = log_prob_

        else:
            x = x.to(self.device)
            log_prob = torch.full(
                (n_samples, self.n_actions), -torch.inf, device=self.device
            )
            for i in range(n_samples):
                candidate_actions_ = candidate_actions[i].cpu()
                log_prob_ = F.log_softmax(
                    x[i][candidate_actions_].reshape((1, -1)) * self.beta
                )[0]
                log_prob[i, candidate_actions_] = log_prob_

        prob = log_prob if is_log_prob else torch.exp(log_prob)

        return prob

    def calc_prob_given_action(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        action: torch.Tensor,
        predicted_reward: Optional[torch.Tensor] = None,
        cluster_centers: Optional[torch.Tensor] = None,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        is_log_prob: bool = False,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Calculate the action choice probability of the given action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        action: torch.Tensor, shape (n_samples, )
             prompts (i.e., action).

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_cluster_emb), default=None
            Low-dimensional embeddings of the cluster center (sentence).

        candidate_actions: list of torch.Tensor, default=None
            Candidate set of actions for each given context.

        is_log_prob: bool, default=False
            Whether to return log probability or not.

        calc_gradient: bool, default=False
            Whether to calculate the gradient or not.

        Return
        -------
        action_choice_probability: torch.Tensor, shape (n_samples, )
            Action choice probability of the given action.

        """
        n_samples = len(context)

        if self.base_model is None:
            if predicted_reward is None:
                raise ValueError(
                    "when base_model is not set, predicted_reward must be given."
                )
            x = predicted_reward

        elif self.given_reward_model:
            x = self.base_model.predict_values(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                calc_gradient=calc_gradient,
            )

        else:
            x = self.base_model.calc_logits(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                cluster_centers=cluster_centers,
                calc_gradient=calc_gradient,
            )

        if candidate_actions is None:
            x = x.to(self.device)
            log_prob_all = F.log_softmax(x * self.beta)
            log_prob = log_prob_all[torch.arange(n_samples), action]

        elif type(x) == list:
            log_prob = torch.zeros((n_samples,), device=self.device)
            for i in range(n_samples):
                x_ = x[i].to(self.device)
                action_flg_ = (candidate_actions[i] == action[i]).long()
                if action_flg_.sum() == 0:
                    log_prob[i] = -torch.inf
                else:
                    log_prob_ = F.log_softmax(x_.reshape((1, -1)) * self.beta)[0]
                    action_id_ = action_flg_.argmax()
                    log_prob[i] = log_prob_[action_id_]

        else:
            x = x.to(self.device)
            log_prob = torch.zeros((n_samples,), device=self.device)
            for i in range(n_samples):
                candidate_actions_ = candidate_actions[i].cpu()
                action_ = action[i].cpu()

                action_flg_ = (candidate_actions_ == action_).long()
                if action_flg_.sum() == 0:
                    log_prob[i] = -torch.inf
                else:
                    log_prob_ = F.log_softmax(
                        x[i][candidate_actions_].reshape((1, -1)) * self.beta
                    )[0]
                    action_id_ = action_flg_.argmax()
                    log_prob[i] = log_prob_[action_id_]

        prob = log_prob if is_log_prob else torch.exp(log_prob)
        return prob

    def predict_policy_value(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        cluster_centers: Optional[torch.Tensor] = None,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        reward_predictor: Optional[BasePromptRewardModel] = None,
        return_per_sample: bool = False,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Calculate policy value.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_cluster_emb), default=None
            Low-dimensional embeddings of the cluster center (sentence).

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        reward_predictor: BasePromptRewardModel, default=None
            (Pre-trained) prompt reward predictor. This must be given when using logit-based model.

        return_per_sample: bool, default=False
            Whether to return policy value per sample or not (i.e., take mean).

        calc_gradient: bool, default=False
            Whether to calculate the gradient or not.

        Return
        -------
        policy_value: float or torch.Tensor, shape (n_samples, )
            Predicted value of the softmax policy.

        """
        n_samples = len(context)

        if (
            not self.given_reward_model
            and reward_predictor is None
            and predicted_reward is None
        ):
            raise ValueError(
                "reward_predictor or predicted_reward must be given when using a logit-based policy."
            )

        if self.base_model is None:
            if predicted_reward is None:
                raise ValueError(
                    "when base_model is not set, predicted_reward must be given."
                )
            x = predicted_reward

        elif self.given_reward_model:
            x = self.base_model.predict_values(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                calc_gradient=calc_gradient,
            )

        else:
            if reward_predictor is None and predicted_reward is None:
                raise ValueError(
                    "when using logit-based model, either reward_predictor or predicted_reward must be given."
                )

            x = self.base_model.calc_logits(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                cluster_centers=cluster_centers,
                calc_gradient=calc_gradient,
            )

        if self.base_model is None:
            reward = x
        elif self.given_reward_model:
            reward = x
        else:
            reward = reward_predictor.predict_values(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                calc_gradient=calc_gradient,
            )

        if candidate_actions is None:
            x = x.to(self.device)
            reward = reward.to(self.device)
            log_prob = F.log_softmax(x * self.beta)
            prob = torch.exp(log_prob)
            value = (reward * prob).sum(dim=1)

        elif type(x) == type(reward) == list:
            value = torch.zeros((n_samples,), device=self.device)

            for i in range(n_samples):
                x_ = x[i].to(self.device)
                reward_ = reward[i].to(self.device)

                log_prob_ = F.log_softmax(x_.reshape((1, -1)) * self.beta)[0]
                prob_ = torch.exp(log_prob_)
                value[i] = (reward_ * prob_).sum()

        elif type(x) == list:
            reward = reward.to(self.device)
            value = torch.zeros((n_samples,), device=self.device)
            for i in range(n_samples):
                x_ = x[i].to(self.device)
                log_prob_ = F.log_softmax(x_.reshape((1, -1)) * self.beta)[0]
                prob_ = torch.exp(log_prob_)
                value[i] = (reward[i][candidate_actions[i]] * prob_).sum()

        else:
            x = x.to(self.device)
            reward = reward.to(self.device)
            value = torch.zeros((n_samples,), device=self.device)
            for i in range(n_samples):
                candidate_actions_ = candidate_actions[i].cpu()
                log_prob_ = F.log_softmax(
                    x[i][candidate_actions_].reshape((1, -1)) * self.beta
                )[0]
                prob_ = torch.exp(log_prob_)
                value[i] = (reward[i][candidate_actions_] * prob_).sum()

        value = value if return_per_sample else value.mean()
        return value


@dataclass
class EpsilonGreedyPolicy(BasePolicy):
    """Epsilon greedy policy.

    Bases: :class:`off_prompts.policy.BasePolicy`

    Imported as: :class:`off_prompts.policy.EpsilonGreedyPolicy`

    Parameters
    -------
    action_list: Sentence, default=None
        Mapping from action id to  prompts. Only applicable when using  prompts.

    n_actions: int, default=None
        Number of clusters. (referred to `n_actions` due to API consistency.

    is_first_stage_policy: bool, default=False
        Whether the policy is the first stage policy of the two stage policy.

    base_model: BasePolicyModel or BaseClusterPolicyModel or BasePromptRewardModel, default=None
        Base model to predict the logit value.

    epsilon: float, default=0.
        Exploration hyperparameter.

    device: str, default = "cuda"
        Device.

    random_state: int, default=None
        Random state.

    """

    action_list: List[str]
    n_actions: Optional[int] = None
    is_first_stage_policy: bool = False
    base_model: Optional[
        Union[BasePromptPolicyModel, BaseClusterPolicyModel, BasePromptRewardModel]
    ] = None
    epsilon: float = 0.0
    device: str = "cuda"
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.is_first_stage_policy:
            if self.n_actions is None:
                raise ValueError(
                    "n_clusters must be given when is_first_stage_policy is True."
                )
        else:
            self.n_actions = len(self.action_list)

        if self.base_model is not None:
            self.given_reward_model = isinstance(self.base_model, BasePromptRewardModel)
        else:
            self.given_reward_model = False

        check_scalar(
            self.epsilon, "epsilon", target_type=float, min_val=0.0, max_val=0.0
        )

        if self.random_state is not None:
            torch_seed(self.random_state, device=self.device)

    def sample_multiple_actions(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        cluster_centers: Optional[torch.Tensor] = None,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        return_action_type: str = "idx",
        n_actions_for_each: int = 1,
        replacement: bool = True,
        **kwargs,
    ):
        """Sample multiple actions for each context and action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_cluster_emb), default=None
            Low-dimensional embeddings of the cluster center (sentence).

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        return_action_type: {"idx", "prompt"}, default="idx"
            Type of action to return.

        n_actions_for_each: int, default=1
            Number of actions to sample.

        replacement: bool, default=True
            Whether to draw with replacement or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions_for_each)
            Action (vector) indicating which prompt is sampled for each input sentence.

        """
        n_samples = len(context)

        action_choice_prob = self.calc_action_choice_probability(
            context=context,
            query=query,
            predicted_reward=predicted_reward,
            cluster_centers=cluster_centers,
            candidate_actions=candidate_actions,
        )
        action = torch.multinomial(
            action_choice_prob, num_samples=n_actions_for_each, replacement=replacement,
        )

        if return_action_type == "prompt":
            prompt = []
            for i in range(n_samples):
                prompt.append(list(itemgetter(*action[i])(self.action_list)))
            action = prompt  # (n_samples, n_actions_for_each)

        return action

    def sample_action(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        cluster_centers: Optional[torch.Tensor] = None,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        return_action_type: str = "idx",
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Sample action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_cluster_emb), default=None
            Low-dimensional embeddings of the cluster center (sentence).

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        return_action_type: {"idx", "onehot", "prompt"}, default="idx"
            Type of action to return.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions) or (n_samples, )
            (Onehot) action (vector) indicating which prompt is sampled for each input sentence.

        """
        n_samples = len(context)
        flg = torch.multinomial(
            torch.FloatTensor([self.epsilon, 1 - self.epsilon]).to(self.device),
            num_samples=n_samples,
            replacement=True,
        )

        if self.base_model is None:
            if predicted_reward is None:
                raise ValueError(
                    "when base_model is not set, predicted_reward must be given."
                )
            x = predicted_reward

        elif self.given_reward_model:
            x = self.base_model.predict_values(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                calc_gradient=calc_gradient,
            )

        else:
            x = self.base_model.calc_logits(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                cluster_centers=cluster_centers,
                calc_gradient=calc_gradient,
            )

        if candidate_actions is None:
            x = x.to(self.device)
            greedy_action = x.argmax(dim=1)
            random_action = torch.multinomial(
                torch.ones((self.n_actions,), device=self.device),
                num_samples=n_samples,
                replacement=True,
            )
            action = flg * greedy_action + (1 - flg) * random_action

        elif type(x) == list:
            action = torch.zeros((n_samples,), dtype=int, device=self.device)
            for i in range(n_samples):
                n_candidate_actions_ = len(candidate_actions[i])
                x_ = x[i].to(self.device)

                greedy_action_ = x_.argmax()
                random_action_ = torch.multinomial(
                    torch.ones((n_candidate_actions_,), device=self.device),
                    num_samples=1,
                )

                action_id_ = flg[i] * greedy_action_ + (1 - flg[i]) * random_action_
                action[i] = candidate_actions[i][action_id_]

        else:
            x = x.to(self.device)
            action = torch.zeros((n_samples,), dtype=int, device=self.device)
            for i in range(n_samples):
                n_candidate_actions_ = len(candidate_actions[i])
                candidate_actions_ = candidate_actions[i].cpu()

                greedy_action_ = x[i][candidate_actions_].argmax()
                random_action_ = torch.multinomial(
                    torch.ones((n_candidate_actions_,), device=self.device),
                    num_samples=1,
                )

                action_id_ = flg[i] * greedy_action_ + (1 - flg[i]) * random_action_
                action[i] = candidate_actions[i][action_id_]

        if return_action_type == "onehot":
            action = F.one_hot(action, num_classes=self.n_actions)
        elif return_action_type == "prompt":
            action = list(itemgetter(*action)(self.action_list))

        return action

    def sample_action_and_output_prob(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        cluster_centers: Optional[torch.Tensor] = None,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        return_action_type: str = "idx",
        is_log_prob: bool = False,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Sample an action and return the action choice probability of sampled action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_cluster_emb), default=None
            Low-dimensional embeddings of the cluster center (sentence).

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        return_action_type: {"idx", "onehot", "prompt"}, default="idx"
            Type of action to return.

        is_log_prob: bool, default=False.
            Whether to return log probability or not.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions) or (n_samples, )
            (Onehot) action (vector) indicating which prompt is sampled for each input sentence.

        prob: torch.Tensor (n_samples, )
            (Log) probability of sampling the above action.

        """
        n_samples = len(context)
        flg = torch.multinomial(
            torch.FloatTensor([self.epsilon, 1 - self.epsilon]).to(self.device),
            num_samples=n_samples,
            replacement=True,
        )

        if self.base_model is None:
            if predicted_reward is None:
                raise ValueError(
                    "when base_model is not set, predicted_reward must be given."
                )
            x = predicted_reward

        elif self.given_reward_model:
            x = self.base_model.predict_values(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                calc_gradient=calc_gradient,
            )

        else:
            x = self.base_model.calc_logits(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                cluster_centers=cluster_centers,
                calc_gradient=calc_gradient,
            )

        if candidate_actions is None:
            x = x.to(self.device)
            greedy_action = x.argmax(dim=1)
            random_action = torch.multinomial(
                torch.ones((self.n_actions,), device=self.device),
                num_samples=n_samples,
                replacement=True,
            )
            action = flg * greedy_action + (1 - flg) * random_action
            prob = (
                flg * (1 - self.epsilon + self.epsilon / self.n_actions)
                + (1 - flg) * self.epsilon / self.n_actions
            )

        elif type(x) == list:
            action = torch.zeros((n_samples,), dtype=int, device=self.device)
            prob = torch.zeros((n_samples,), device=self.device)
            for i in range(n_samples):
                n_candidate_actions_ = len(candidate_actions[i])
                x_ = x[i].to(self.device)

                greedy_action_ = x_.argmax()
                random_action_ = torch.multinomial(
                    torch.ones((n_candidate_actions_,), device=self.device),
                    num_samples=1,
                )

                action_id_ = flg[i] * greedy_action_ + (1 - flg[i]) * random_action_
                prob_ = (
                    flg[i] * (1 - self.epsilon + self.epsilon / n_candidate_actions_)
                    + (1 - flg) * self.epsilon / n_candidate_actions_
                )

                action[i] = candidate_actions[i][action_id_]
                prob[i] = prob_

        else:
            x = x.to(self.device)
            action = torch.zeros((n_samples,), dtype=int, device=self.device)
            prob = torch.zeros((n_samples,), device=self.device)
            for i in range(n_samples):
                n_candidate_actions_ = len(candidate_actions[i])
                candidate_actions_ = candidate_actions[i].cpu()

                greedy_action_ = x[i][candidate_actions_].argmax()
                random_action_ = torch.multinomial(
                    torch.ones((n_candidate_actions_,)), num_samples=1,
                )

                action_id_ = flg[i] * greedy_action_ + (1 - flg[i]) * random_action_
                prob_ = (
                    flg[i] * (1 - self.epsilon + self.epsilon / n_candidate_actions_)
                    + (1 - flg) * self.epsilon / n_candidate_actions_
                )

                action[i] = candidate_actions[i][action_id_]
                prob[i] = prob_

        prob = torch.log(prob) if is_log_prob else prob

        if return_action_type == "onehot":
            action = F.one_hot(action, num_classes=self.n_actions)
        elif return_action_type == "prompt":
            action = list(itemgetter(*action)(self.action_list))

        return action, prob

    def calc_action_choice_probability(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        cluster_centers: Optional[torch.Tensor] = None,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        is_log_prob: bool = False,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Calculate the action choice probabilities (within candidate actions).

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_cluster_emb), default=None
            Low-dimensional embeddings of the cluster center (sentence).

        candidate_actions: list of torch.Tensor, default=None
            Candidate set of actions for each given context.

        is_log_prob: bool, default=False
            Whether to return log probability or not.

        calc_gradient: bool, default=False
            Whether to calculate the gradient or not.

        Return
        -------
        action_choice_probability: torch.Tensor or List[torch.Tensor], shape (n_samples, n_candidate_actions[i])
            Action choice probability of all candidate actions.

        """
        n_samples = len(context)

        if self.base_model is None:
            if predicted_reward is None:
                raise ValueError(
                    "when base_model is not set, predicted_reward must be given."
                )
            x = predicted_reward

        elif self.given_reward_model:
            x = self.base_model.predict_values(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                calc_gradient=calc_gradient,
            )

        else:
            x = self.base_model.calc_logits(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                cluster_centers=cluster_centers,
                calc_gradient=calc_gradient,
            )

        if candidate_actions is None:
            x = x.to(self.device)
            greedy_flg = F.one_hot(x.argmax(dim=1), num_classes=self.n_actions)
            prob = (
                greedy_flg * (1 - self.epsilon + self.epsilon / self.n_actions)
                + (1 - greedy_flg) * self.epsilon / self.n_actions
            )

        elif type(x) == list:
            prob = torch.zeros((n_samples, self.n_actions), device=self.device)
            for i in range(n_samples):
                n_candidate_actions_ = len(candidate_actions[i])
                x_ = x[i].to(self.device)

                greedy_flg_ = F.one_hot(
                    x_.reshape((1, -1)).argmax(dim=1), num_classes=n_candidate_actions_,
                )[0]
                prob[i, candidate_actions[i]] = (
                    greedy_flg_
                    * (1 - self.epsilon + self.epsilon / n_candidate_actions_)
                    + (1 - greedy_flg_) * self.epsilon / n_candidate_actions_
                )

        else:
            x = x.to(self.device)
            prob = torch.zeros((n_samples, self.n_actions), device=self.device)
            for i in range(n_samples):
                n_candidate_actions_ = len(candidate_actions[i])
                candidate_actions_ = candidate_actions[i].cpu()

                greedy_flg_ = F.one_hot(
                    x[i][candidate_actions_].reshape((1, -1)).argmax(dim=1),
                    num_classes=n_candidate_actions_,
                )[0]
                prob[i, candidate_actions[i]] = (
                    greedy_flg_
                    * (1 - self.epsilon + self.epsilon / n_candidate_actions_)
                    + (1 - greedy_flg_) * self.epsilon / n_candidate_actions_
                )

        prob = torch.log(prob) if is_log_prob else prob
        return prob

    def calc_prob_given_action(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        action: torch.Tensor,
        predicted_reward: Optional[torch.Tensor] = None,
        cluster_centers: Optional[torch.Tensor] = None,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        is_log_prob: bool = False,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Calculate the action choice probability of the given action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        action: torch.Tensor, shape (n_samples, )
             prompts (i.e., action).

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_cluster_emb), default=None
            Low-dimensional embeddings of the cluster center (sentence).

        candidate_actions: list of torch.Tensor, default=None
            Candidate set of actions for each given context.

        is_log_prob: bool, default=False
            Whether to return log probability or not.

        calc_gradient: bool, default=False
            Whether to calculate the gradient or not.

        Return
        -------
        action_choice_probability: torch.Tensor, shape (n_samples, )
            Action choice probability of the given action.

        """
        n_samples = len(context)

        if self.base_model is None:
            if predicted_reward is None:
                raise ValueError(
                    "when base_model is not set, predicted_reward must be given."
                )
            x = predicted_reward

        elif self.given_reward_model:
            x = self.base_model.predict_values(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                calc_gradient=calc_gradient,
            )

        else:
            x = self.base_model.calc_logits(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                cluster_centers=cluster_centers,
                calc_gradient=calc_gradient,
            )

        if candidate_actions is None:
            x = x.to(self.device)
            greedy_flg = F.one_hot(x.argmax(dim=1), num_classes=self.n_actions)
            prob_all = (
                greedy_flg * (1 - self.epsilon + self.epsilon / self.n_actions)
                + (1 - greedy_flg) * self.epsilon / self.n_actions
            )
            prob = prob_all[torch.arange(n_samples), action]

        elif type(x) == list:
            prob = torch.zeros((n_samples,), device=self.device)
            for i in range(n_samples):
                n_candidate_actions_ = len(candidate_actions)
                action_flg_ = (candidate_actions[i] == action[i]).long()
                x_ = x[i].to(self.device)

                if action_flg_.sum() == 0:
                    prob[i] = 0
                else:
                    greedy_flg_ = F.one_hot(
                        x_.reshape((1, -1)).argmax(dim=1), num_classes=self.n_actions
                    )[0]
                    prob_all_ = (
                        greedy_flg_
                        * (1 - self.epsilon + self.epsilon / n_candidate_actions_)
                        + (1 - greedy_flg_) * self.epsilon / n_candidate_actions_
                    )

                    action_id_ = action_flg_.argmax()
                    prob[i] = prob_all_[action_id_]

        else:
            prob = torch.zeros((n_samples,), device=self.device)
            for i in range(n_samples):
                n_candidate_actions_ = len(candidate_actions)
                candidate_actions_ = candidate_actions[i].cpu()
                action_ = action[i].cpu()
                action_flg_ = (candidate_actions_ == action_).long()

                if action_flg_.sum() == 0:
                    prob[i] = 0
                else:
                    greedy_flg_ = F.one_hot(
                        x[i][candidate_actions_].reshape((1, -1)).argmax(dim=1),
                        num_classes=self.n_actions,
                    )[0]
                    prob_all_ = (
                        greedy_flg_
                        * (1 - self.epsilon + self.epsilon / n_candidate_actions_)
                        + (1 - greedy_flg_) * self.epsilon / n_candidate_actions_
                    )

                    action_id_ = action_flg_.argmax()
                    prob[i] = prob_all_[action_id_]

        prob = torch.log(prob) if is_log_prob else prob
        return prob

    def predict_policy_value(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        cluster_centers: Optional[torch.Tensor] = None,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        reward_predictor: Optional[BasePromptRewardModel] = None,
        return_per_sample: bool = False,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Calculate policy value.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_cluster_emb), default=None
            Low-dimensional embeddings of the cluster center (sentence).

        candidate_actions: list of torch.Tensor, default=None
            Candidate set of actions for each given context.

        reward_predictor: BasePromptRewardModel, default=None
            (Pre-trained) prompt reward predictor. This must be given when using logit-based model.

        return_per_sample: bool, default=False
            Whether to return policy value per sample or not (i.e., take mean).

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        policy_value: float or torch.Tensor, shape (n_samples, )
            Predicted value of the epsilon-greedy policy.

        """
        n_samples = len(context)

        if (
            not self.given_reward_model
            and reward_predictor is None
            and predicted_reward is None
        ):
            raise ValueError(
                "reward_predictor must be given when using a logit-based policy."
            )

        if self.base_model is None:
            if predicted_reward is None:
                raise ValueError(
                    "when base_model is not set, predicted_reward must be given."
                )
            x = predicted_reward

        elif self.given_reward_model:
            x = self.base_model.predict_values(
                context=context, query=query, calc_gradient=calc_gradient,
            )

        else:
            if reward_predictor is None and predicted_reward is None:
                raise ValueError(
                    "when using logit-based model, either reward_predictor or predicted_reward must be given."
                )

            x = self.base_model.calc_logits(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                cluster_centers=cluster_centers,
                calc_gradient=calc_gradient,
            )

        if self.base_model is None:
            reward = x
        elif self.given_reward_model:
            reward = x
        else:
            reward = reward_predictor.predict_values(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                calc_gradient=calc_gradient,
            )

        if candidate_actions is None:
            x = x.to(self.device)
            reward = reward.to(self.device)
            greedy_flg = F.one_hot(x.argmax(dim=1), num_classes=self.n_actions)
            prob = (
                greedy_flg * (1 - self.epsilon + self.epsilon / self.n_actions)
                + (1 - greedy_flg) * self.epsilon / self.n_actions
            )
            value = (reward * prob).sum(dim=1)

        elif type(x) == type(reward) == list:
            value = torch.zeros((n_samples,), device=self.device)
            for i in range(n_samples):
                n_candidate_actions_ = len(candidate_actions[i])
                x_ = x[i].to(self.device)
                reward_ = reward[i].to(self.device)

                greedy_flg_ = F.one_hot(
                    x_.reshape((1, -1)).argmax(dim=1), num_classes=n_candidate_actions_,
                )[0]

                prob_ = (
                    greedy_flg_
                    * (1 - self.epsilon + self.epsilon / n_candidate_actions_)
                    + (1 - greedy_flg_) * self.epsilon / n_candidate_actions_
                )
                value[i] = (reward_ * prob_).sum()

        elif type(x) == list:
            reward = reward.to(self.device)
            value = torch.zeros((n_samples,), device=self.device)
            for i in range(n_samples):
                n_candidate_actions_ = len(candidate_actions[i])
                candidate_actions_ = candidate_actions[i].cpu()
                x_ = x[i].to(self.device)

                greedy_flg_ = F.one_hot(
                    x_.reshape((1, -1)).argmax(dim=1), num_classes=n_candidate_actions_,
                )[0]

                prob_ = (
                    greedy_flg_
                    * (1 - self.epsilon + self.epsilon / n_candidate_actions_)
                    + (1 - greedy_flg_) * self.epsilon / n_candidate_actions_
                )
                value[i] = (reward[i][candidate_actions_] * prob_).sum()

        else:
            x = x.to(self.device)
            reward = reward.to(self.device)
            value = torch.zeros((n_samples,), device=self.device)
            for i in range(n_samples):
                n_candidate_actions_ = len(candidate_actions[i])
                candidate_actions_ = candidate_actions[i].cpu()

                greedy_flg_ = F.one_hot(
                    x[i][candidate_actions_].reshape((1, -1)).argmax(dim=1),
                    num_classes=n_candidate_actions_,
                )[0]

                prob_ = (
                    greedy_flg_
                    * (1 - self.epsilon + self.epsilon / n_candidate_actions_)
                    + (1 - greedy_flg_) * self.epsilon / n_candidate_actions_
                )
                value[i] = (reward[i][candidate_actions_] * prob_).sum()

        value = value if return_per_sample else value.mean()
        return value

    def greedy_action(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        cluster_centers: Optional[torch.Tensor] = None,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        return_action_type: str = "idx",
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Retrieve greedy action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_cluster_emb), default=None
            Low-dimensional embeddings of the cluster center (sentence).

        candidate_actions: list of torch.Tensor, default=None
            Candidate set of actions for each given context.

        return_action_type: {"idx", "onehot", "prompt"}, default="idx"
            Type of action to return.

        calc_gradient: bool, default=False
            Whether to calculate the gradient or not.

        Return
        -------
        greedy_action: torch.Tensor, shape (n_samples, )
            Greedily chosen  prompts for each given user context and query.

        """
        n_samples = len(context)

        if self.base_model is None:
            if predicted_reward is None:
                raise ValueError(
                    "when base_model is not set, predicted_reward must be given."
                )
            x = predicted_reward

        elif self.given_reward_model:
            x = self.base_model.predict_values(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                calc_gradient=calc_gradient,
            )

        else:
            x = self.base_model.calc_logits(
                context=context,
                query=query,
                candidate_actions=candidate_actions,
                cluster_centers=cluster_centers,
                calc_gradient=calc_gradient,
            )

        if candidate_actions is None:
            x = x.to(self.device)
            greedy_action = x.argmax(dim=1)

        elif type(x) == list:
            greedy_action = torch.zeros((n_samples,), dtype=int, device=self.device)
            for i in range(n_samples):
                x_ = x[i].to(self.device)
                action_id_ = x_.argmax()
                greedy_action[i] = candidate_actions[i][action_id_]

        else:
            x = x.to(self.device)
            greedy_action = torch.zeros((n_samples,), dtype=int, device=self.device)
            for i in range(n_samples):
                candidate_actions_ = candidate_actions[i].cpu()
                action_id_ = x[i][candidate_actions_].argmax()
                greedy_action[i] = candidate_actions_[action_id_]

        if return_action_type == "onehot":
            greedy_action = F.one_hot(greedy_action, num_classes=self.n_actions)
        elif return_action_type == "prompt":
            greedy_action = list(itemgetter(*greedy_action)(self.action_list))

        return greedy_action


@dataclass
class UniformRandomPolicy(BasePolicy):
    """Uniform random policy for  action.

    Bases: :class:`off_prompts.policy.BasePolicy`

    Imported as: :class:`off_prompts.policy.UniformRandomPolicy`

    Parameters
    -------
    action_list: Sentence
        Mapping from action id to  prompts. Only applicable when using  prompts.

    n_actions: int, default=None
        Number of clusters. (referred to `n_actions` due to API consistency)

    is_first_stage_policy: bool, default=False
        Whether the policy is the first stage policy of the two stage policy.

    device: str, default = "cuda"
        Device.

    random_state: int, default=None
        Random state.

    """

    action_list: List[str]
    n_actions: Optional[int] = None
    is_first_stage_policy: bool = False
    device: str = "cuda"
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.is_first_stage_policy:
            if self.n_actions is None:
                raise ValueError(
                    "n_clusters must be given when is_first_stage_policy is True."
                )
        else:
            self.n_actions = len(self.action_list)

        if self.random_state is not None:
            torch_seed(self.random_state)

    def sample_multiple_actions(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        cluster_centers: Optional[torch.Tensor] = None,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        return_action_type: str = "idx",
        n_actions_for_each: int = 1,
        replacement: bool = True,
        **kwargs,
    ):
        """Sample multiple actions for each context and action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        predicted_reward: torch.Tensor, default=None
            For API consistency.

        cluster_centers: torch.Tensor, default=None
            For API consistency.

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        return_action_type: {"idx", "prompt"}, default="idx"
            Type of action to return.

        n_actions_for_each: int, default=1
            Number of actions to sample.

        replacement: bool, default=True
            Whether to draw with replacement or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions_for_each)
            Action (vector) indicating which prompt is sampled for each input sentence.

        """
        n_samples = len(context)

        action_choice_prob = self.calc_action_choice_probability(
            context=context,
            query=query,
            predicted_reward=predicted_reward,
            cluster_centers=cluster_centers,
            candidate_actions=candidate_actions,
        )
        action = torch.multinomial(
            action_choice_prob, num_samples=n_actions_for_each, replacement=replacement,
        )

        if return_action_type == "prompt":
            prompt = []
            for i in range(n_samples):
                prompt.append(list(itemgetter(*action[i])(self.action_list)))
            action = prompt  # (n_samples, n_actions_for_each)

        return action

    def sample_action(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        cluster_centers: Optional[torch.Tensor] = None,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        return_action_type: str = "idx",
        **kwargs,
    ):
        """Sample action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        predicted_reward: torch.Tensor, default=None
            For API consistency.

        cluster_centers: torch.Tensor, default=None
            For API consistency.

        candidate_actions: list of torch.Tensor, default=None
            Candidate set of actions for each given context.

        return_action_type: {"idx", "onehot", "prompt"}, default="idx"
            Type of action to return.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions) or (n_samples, )
            (Onehot) action (vector) indicating which prompt is sampled for each input sentence.

        """
        n_samples = len(context)

        if candidate_actions is None:
            action = torch.multinomial(
                torch.ones((self.n_actions,), device=self.device),
                num_samples=n_samples,
                replacement=True,
            )

        else:
            action = torch.zeros((n_samples,), dtype=int, device=self.device)
            for i in range(n_samples):
                n_candidate_actions_ = len(candidate_actions[i])
                action_id_ = torch.multinomial(
                    torch.ones((n_candidate_actions_,)), num_samples=1,
                )
                action[i] = candidate_actions[i][action_id_]

        if return_action_type == "onehot":
            action = F.one_hot(action, num_classes=self.n_actions)
        elif return_action_type == "prompt":
            action = list(itemgetter(*action)(self.action_list))

        return action

    def sample_action_and_output_prob(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        cluster_centers: Optional[torch.Tensor] = None,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        return_action_type: str = "idx",
        is_log_prob: bool = False,
        **kwargs,
    ):
        """Sample an action and return the action choice probability of sampled action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        predicted_reward: torch.Tensor, default=None
            For API consistency.

        cluster_centers: torch.Tensor, default=None
            For API consistency.

        candidate_actions: list of torch.Tensor, default=None
            Candidate set of actions for each given context.

        return_action_type: {"idx", "onehot", "prompt"}, default="idx"
            Type of action to return.

        is_log_prob: bool, default=False
            Whether to return log probability or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions) or (n_samples, )
            (Onehot) action (vector) indicating which prompt is sampled for each input sentence.

        prob: torch.Tensor (n_samples, )
            (Log) probability of sampling the above action.

        """
        n_samples = len(context)

        if candidate_actions is None:
            action = torch.multinomial(
                torch.ones((self.n_actions,), device=self.device),
                num_samples=n_samples,
                replacement=True,
            )
            prob = torch.full((n_samples,), 1 / self.n_actions, device=self.device)

        else:
            action = torch.zeros((n_samples,), dtype=int, device=self.device)
            prob = torch.zeros((n_samples,), device=self.device)
            for i in range(n_samples):
                n_candidate_actions_ = len(candidate_actions[i])
                action_id_ = torch.multinomial(
                    torch.ones((n_candidate_actions_,), device=self.device),
                    num_samples=1,
                )
                action[i] = candidate_actions[i][action_id_]
                prob[i] = 1 / n_candidate_actions_

        if return_action_type == "onehot":
            action = F.one_hot(action, num_classes=self.n_actions)
        elif return_action_type == "prompt":
            action = list(itemgetter(*action)(self.action_list))

        prob = torch.log(prob) if is_log_prob else prob
        return action, prob

    def calc_action_choice_probability(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        cluster_centers: Optional[torch.Tensor] = None,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        is_log_prob: bool = False,
        **kwargs,
    ):
        """Calculate the action choice probabilities (within candidate actions).

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        predicted_reward: torch.Tensor, default=None
            For API consistency.

        cluster_centers: torch.Tensor, default=None
            For API consistency.

        candidate_actions: list of torch.Tensor, default=None
            Candidate set of actions for each given context.

        is_log_prob: bool, default=False
            Whether to return log probability or not.

        Return
        -------
        action_choice_probability: torch.Tensor or List[torch.Tensor], shape (n_samples, n_candidate_actions[i])
            Action choice probability of all candidate actions.

        """
        n_samples = len(context)

        if candidate_actions is None:
            prob = torch.full(
                (n_samples, self.n_actions), 1 / self.n_actions, device=self.device
            )
        else:
            prob = torch.zeros((n_samples, self.n_actions), device=self.device)
            for i in range(n_samples):
                n_candidate_actions_ = len(candidate_actions[i])
                prob[i, candidate_actions[i]] = torch.full(
                    (n_candidate_actions_,),
                    1 / n_candidate_actions_,
                    device=self.device,
                )

        prob = torch.log(prob) if is_log_prob else prob
        return prob

    def calc_prob_given_action(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        action: torch.Tensor,
        predicted_reward: Optional[torch.Tensor] = None,
        cluster_centers: Optional[torch.Tensor] = None,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        is_log_prob: bool = False,
        **kwargs,
    ):
        """Calculate the action choice probability of the given action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        action: torch.Tensor, shape (n_samples, )
             prompts (i.e., action).

        predicted_reward: torch.Tensor, default=None
            For API consistency.

        cluster_centers: torch.Tensor, default=None
            For API consistency.

        candidate_actions: list of torch.Tensor, default=None
            Candidate set of actions for each given context.

        is_log_prob: bool, default=False
            Whether to return log probability or not.

        Return
        -------
        action_choice_probability: torch.Tensor, shape (n_samples, )
            Action choice probability of the given action.

        """
        n_samples = len(context)

        if candidate_actions is None:
            prob = torch.full((n_samples,), self.n_actions, device=self.device)

        else:
            prob = torch.zeros((n_samples,), device=self.device)
            for i in range(n_samples):
                n_candidate_actions_ = len(candidate_actions)
                action_flg_ = (candidate_actions[i] == action[i]).long()

                if action_flg_.sum() == 0:
                    prob[i] = 0
                else:
                    prob[i] = 1 / n_candidate_actions_

        prob = torch.log(prob) if is_log_prob else prob
        return prob

    def predict_policy_value(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        cluster_centers: Optional[torch.Tensor] = None,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        reward_predictor: Optional[BasePromptRewardModel] = None,
        return_per_sample: bool = False,
        **kwargs,
    ):
        """Calculate policy value.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        cluster_centers: torch.Tensor, default=None
            For API consistency.

        candidate_actions: list of torch.Tensor, default=None
            Candidate set of actions for each given context.

        reward_predictor: BasePromptRewardModel, default=None
            (Pre-trained) prompt reward predictor. This must be given when using logit-based model.

        return_per_sample: bool, default=False
            Whether to return policy value per sample or not (i.e., take mean).

        Return
        -------
        policy_value: float or torch.Tensor, shape (n_samples, )
            Predicted value of the uniform random policy.

        """
        n_samples = len(context)

        if predicted_reward is None:
            if reward_predictor is None:
                raise ValueError(
                    "Either reward_predictor or predicted_reward must be given."
                )

            predicted_reward = reward_predictor.predict_values(
                context=context, query=query, candidate_actions=candidate_actions,
            )

        if candidate_actions is None:
            prob = torch.full(
                (n_samples, self.n_actions), 1 / self.n_actions, device=self.device
            )
            value = (predicted_reward * prob).sum(dim=1)

        elif type(predicted_reward) == list:
            value = torch.zeros((n_samples,), device=self.device)
            for i in range(n_samples):
                n_candidate_actions_ = len(candidate_actions[i])
                prob_ = torch.full(
                    (n_candidate_actions_,),
                    1 / n_candidate_actions_,
                    device=self.device,
                )
                value[i] = (predicted_reward[i] * prob_).sum()

        else:
            value = torch.zeros((n_samples,), device=self.device)
            for i in range(n_samples):
                n_candidate_actions_ = len(candidate_actions[i])
                prob_ = torch.full(
                    (n_candidate_actions_,),
                    1 / n_candidate_actions_,
                    device=self.device,
                )
                value[i] = (predicted_reward[i][candidate_actions[i]] * prob_).sum()

        value = value if return_per_sample else value.mean()
        return value
