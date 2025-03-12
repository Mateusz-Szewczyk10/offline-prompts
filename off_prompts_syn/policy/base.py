"""Abstract base class for policies and models used to define policies."""
from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
from tqdm.auto import tqdm
from typing import Optional, List, Any, Union

import torch
from torch import nn
from torch.nn import functional as F

from ..dataset.function import AuxiliaryOutputGenerator


@dataclass
class BasePolicy(metaclass=ABCMeta):
    """Base class for policy.

    Imported as: :class:`src.policy.BasePolicy`

    Note
    -------
    1. The following parameter should be specified in init.

    n_actions: int
        Number of discrete actions.

    """

    @abstractmethod
    def sample_multiple_actions(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        return_action_type: str = "idx",
        n_actions_for_each: int = 1,
        replacement: bool = True,
        **kwargs,
    ):
        """Sample multiple actions for each context and action.

        Parameters
        -------
        ontext: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        return_action_type: {"idx", "embedding"}, default="idx"
            Type of action to return.

        n_actions_for_each: int, default=1
            Number of actions to sample.

        replacement: bool, default=True
            Whether to draw with replacement or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions_for_each) or (n_samples, n_actions_for_each, dim_action)
            Action (vector) indicating which prompt is sampled for each input sentence.

        """
        raise NotImplementedError()

    @abstractmethod
    def sample_action(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
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

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        return_action_type: {"idx", "onehot", "embedding"}, default="idx"
            Type of action to return.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions) or (n_samples, ) or (n_samples, dim_action)
            (Onehot) action (vector) sampled for each input.

        """
        raise NotImplementedError()

    @abstractmethod
    def sample_action_and_output_prob(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
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

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        return_action_type: {"idx", "onehot", "embedding"}, default="idx"
            Type of action to return.

        is_log_prob: bool, default=False.
            Whether to return log probability or not.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions) or (n_samples, ) or (n_samples, dim_action)
            (Onehot) action (vector) sampled for each input.

        prob: torch.Tensor (n_samples, )
            (Log) probability of sampling the above action.

        """
        raise NotImplementedError()

    @abstractmethod
    def calc_action_choice_probability(
        context: torch.Tensor,
        query: torch.Tensor,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        is_log_prob: bool = False,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Calculate the action choice probabilities for all candidate actions (only for discrete prompts).

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        is_log_prob: bool, default=False.
            Whether to return log probability or not.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        prob: torch.Tensor (n_samples, )
            (Log) probability of sampling actions.

        """
        raise NotImplementedError()

    @abstractmethod
    def calc_prob_given_action(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        action: torch.Tensor,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Calculate the action choice probability of the given action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        action: torch.Tensor, shape (n_samples, ) or (n_samples, dim_action)
            Discrete action.

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action_choice_probability: torch.Tensor, shape (n_samples, )
            Action choice probability of the given action.

        """
        raise NotImplementedError()

    @abstractmethod
    def predict_policy_value(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        predicted_reward: Optional[torch.Tensor] = None,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        reward_predictor: Optional[Any] = None,
        return_per_sample: bool = False,
        **kwargs,
    ):
        """Calculate policy value.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        reward_predictor: BaseActionRewardModel, default=None
            (Pre-trained) action reward predictor. This must be given when using logit-based model.

        return_per_sample: bool, default=False
            Whether to return policy value per sample or not (i.e., take mean).

        Return
        -------
        policy_value: float or torch.Tensor, shape (n_samples, )
            Predicted value of the policy.

        """
        raise NotImplementedError()


class BaseActionPolicyModel(nn.Module):
    """Base multi-head model for discrete action policy.

    Imported as: :class:`src.policy.BaseActionPolicyModel`

    Note
    -------
    The following parameter should be specified in init.

    n_actions: int
        Number of discrete prompts.

    """

    def __init__(
        self,
    ):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
    ):
        """Calculate some logit values.

        Parameters
        -------
        inputs: torch.Tensor, shape (n_samples, dim_context + dim_query)
            Input vectors.

        Return
        -------
        action_logits: torch.Tensor (n_samples, n_actions)
            Some logit value of action on given inputs.

        """
        raise NotImplementedError()

    def calc_logits(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Calculate value of all discrete actions.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action_logits: torch.Tensor (n_samples, n_actions)
            Some logit value of action on given inputs.

        """
        inputs = torch.cat((context, query), dim=1)

        if calc_gradient:
            logits = self(inputs)
        else:
            with torch.no_grad():
                logits = self(inputs)

        return logits

    def sample_multiple_actions(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
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

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        return_action_type: {"idx", "embedding"}, default="idx"
            Type of action to return.

        n_actions_for_each: int, default=1
            Number of actions to sample.

        replacement: bool, default=True
            Whether to draw with replacement or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions_for_each) or (n_samples, n_actions_for_each, dim_action)
            Action (vector) indicating which prompt is sampled for each input sentence.

        """
        n_samples = len(context)

        action_choice_prob = self.calc_action_choice_probability(
            context=context,
            query=query,
        )
        action = torch.multinomial(
            action_choice_prob,
            num_samples=n_actions_for_each,
            replacement=replacement,
        )

        if return_action_type == "embedding":
            embedding = self.action_list[
                torch.arange(n_samples * n_actions_for_each), action.flatten()
            ]
            action = embedding.reshape((n_samples, n_actions_for_each, -1))

        return action

    def sample_action(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        return_action_type: str = "idx",
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Sample action using gumble softmax.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        return_action_type: {"idx", "onehot", "embedding"}, default="idx"
            Type of action to return.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions) or (n_samples, ) or (n_samples, dim_action)
            (Onehot) action (vector) sampled for each input.

        """
        x = self.calc_logits(
            context=context,
            query=query,
            calc_gradient=calc_gradient,
        )
        action = F.gumbel_softmax(x, hard=True)

        if return_action_type == "idx":
            action = action.argmax(dim=1)
        elif return_action_type == "embedding":
            action = action.argmax(dim=1)
            action = self.action_list[action]

        return action

    def sample_action_and_output_prob(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
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

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        return_action_type: {"idx", "onehot", "embedding"}, default="idx"
            Type of action to return.

        is_log_prob: bool, default=False.
            Whether to return log probability or not.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions) or (n_samples, ) or (n_samples, dim_action)
            (Onehot) action (vector) sampled for each input.

        prob: torch.Tensor (n_samples, )
            (Log) probability of sampling the above action.

        """
        x = self.calc_logits(
            context=context,
            query=query,
            calc_gradient=calc_gradient,
        )
        action = F.gumbel_softmax(x, hard=True)
        log_prob = (F.log_softmax(x) * action).sum(dim=1)

        prob = log_prob if is_log_prob else torch.exp(log_prob)

        if return_action_type == "idx":
            action = action.argmax(dim=1)
        elif return_action_type == "embedding":
            action = action.argmax(dim=1)
            action = self.action_list[action]

        return action, prob

    def calc_action_choice_probability(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        is_log_prob: bool = False,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Calculate the action choice probabilities for all candidate actions.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        is_log_prob: bool, default=False.
            Whether to return log probability or not.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        prob: torch.Tensor (n_samples, )
            (Log) probability of sampling actions.

        """
        x = self.calc_logits(
            context=context,
            query=query,
            calc_gradient=calc_gradient,
        )
        log_prob = F.log_softmax(x)
        prob = log_prob if is_log_prob else torch.exp(log_prob)
        return prob

    def calc_prob_given_action(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        action: torch.Tensor,
        is_log_prob: bool = False,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Calculate the action choice probability of the given action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        action: torch.Tensor (n_samples, n_actions) or (n_samples, )
            (Onehot) action (vector) sampled for each input.

        is_log_prob: bool, default=False.
            Whether to return log probability or not.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        prob: torch.Tensor (n_samples, )
            (Log) probability of sampling actions.

        """
        is_onehot = len(action.shape) == 2
        action = action if is_onehot else F.one_hot(action, num_classes=self.n_actions)

        x = self.calc_logits(
            context=context,
            query=query,
            calc_gradient=calc_gradient,
        )
        log_prob = (F.log_softmax(x) * action).sum(dim=1)
        prob = log_prob if is_log_prob else torch.exp(log_prob)
        return prob

    def predict_policy_value(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        predicted_reward: Optional[torch.Tensor] = None,
        reward_predictor: Optional[Any] = None,
        return_per_sample: bool = False,
        **kwargs,
    ):
        """Calculate policy value.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        reward_predictor: BaseActionRewardModel, default=None
            (Pre-trained) action reward predictor. This must be given when using logit-based model.

        return_per_sample: bool, default=False
            Whether to return policy value per sample or not (i.e., take mean).

        Return
        -------
        policy_value: float or torch.Tensor, shape (n_samples, )
            Predicted value of the policy.

        """
        if predicted_reward is None and reward_predictor is None:
            raise ValueError(
                "Either predicted_reward or reward_predictor must be given."
            )

        if predicted_reward is None:
            predicted_reward = reward_predictor.predict_values(
                context=context,
                query=query,
            )

        prob = self.calc_action_choice_probability(
            context=context,
            query=query,
        )

        policy_value = (predicted_reward * prob).sum(dim=1)
        policy_value = policy_value if return_per_sample else policy_value.mean()
        return policy_value


class BaseClusterPolicyModel(nn.Module):
    """Base multi-head model for discrete cluster policy.

    Imported as: :class:`src.policy.BaseClusterPolicyModel`

    Note
    -------
    The following parameter should be specified in init.

    n_actions: int
        Number of clusters. (refer to n_actions due to API consistency)

    """

    def __init__(
        self,
    ):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
        cluster_centers: torch.Tensor,
    ):
        """Calculate some logit values.

        Parameters
        -------
        inputs: torch.Tensor, shape (n_samples, dim_context + dim_query)
            Input vectors.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_auxiliary_output)
            Low-dimensional embeddings of the cluster center (auxiliary output).

        Return
        -------
        action_logits: torch.Tensor (n_samples, n_actions)
            Some logit value of action on given inputs.

        """
        raise NotImplementedError()

    def calc_logits(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        cluster_centers: torch.Tensor,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Calculate value of all discrete actions.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_auxiliary_output)
            Low-dimensional embeddings of the cluster center (auxiliary output).

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action_logits: torch.Tensor (n_samples, )
            Some logit value of action on given inputs.

        """
        inputs = torch.cat((context, query), dim=1)

        if calc_gradient:
            logits = self(inputs, cluster_centers)
        else:
            with torch.no_grad():
                logits = self(inputs, cluster_centers)

        return logits

    def sample_multiple_actions(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        cluster_centers: torch.Tensor,
        n_actions_for_each: int = 1,
        replacement: bool = True,
        **kwargs,
    ):
        """Sample multiple clusters (actions) for each context and action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_auxiliary_output)
            Low-dimensional embeddings of the cluster center (auxiliary output).

        return_action_type: {"idx", "embedding"}, default="idx"
            Type of action to return.

        n_actions_for_each: int, default=1
            Number of actions to sample.

        replacement: bool, default=True
            Whether to draw with replacement or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions_for_each) or (n_samples, n_actions_for_each, dim_action)
            Cluster (vector) indicating which prompt is sampled for each input sentence.

        """
        action_choice_prob = self.calc_action_choice_probability(
            context=context,
            query=query,
            cluster_centers=cluster_centers,
        )
        action = torch.multinomial(
            action_choice_prob,
            num_samples=n_actions_for_each,
            replacement=replacement,
        )
        return action

    def sample_action(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        cluster_centers: torch.Tensor,
        return_action_type: str = "idx",
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Sample cluster (action) using gumble softmax.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_auxiliary_output)
            Low-dimensional embeddings of the cluster center (auxiliary output).

        return_action_type: {"idx", "onehot"}, default="idx"
            Type of action to return.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions) or (n_samples, ) or (n_samples, dim_action)
            (Onehot) cluster (vector) sampled for each input.

        """
        x = self.calc_logits(
            context=context,
            query=query,
            cluster_centers=cluster_centers,
            calc_gradient=calc_gradient,
        )
        action = F.gumbel_softmax(x, hard=True)

        if return_action_type == "idx":
            action = action.argmax(dim=1)

        return action

    def sample_action_and_output_prob(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        cluster_centers: torch.Tensor,
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

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_auxiliary_output)
            Low-dimensional embeddings of the cluster center (auxiliary output).

        return_action_type: {"idx", "onehot"}, default="idx"
            Type of action to return.

        is_log_prob: bool, default=False.
            Whether to return log probability or not.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions) or (n_samples, ) or (n_samples, dim_action)
            (Onehot) action (vector) sampled for each input.

        prob: torch.Tensor (n_samples, )
            (Log) probability of sampling the above action.

        """
        x = self.calc_logits(
            context=context,
            query=query,
            cluster_centers=cluster_centers,
            calc_gradient=calc_gradient,
        )
        action = F.gumbel_softmax(x, hard=True)
        log_prob = (F.log_softmax(x) * action).sum(dim=1)

        prob = log_prob if is_log_prob else torch.exp(log_prob)

        if return_action_type == "idx":
            action = action.argmax(dim=1)

        return action, prob

    def calc_action_choice_probability(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        cluster_centers: torch.Tensor,
        is_log_prob: bool = False,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Calculate the action choice probabilities for all candidate actions.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_auxiliary_output)
            Low-dimensional embeddings of the cluster center (auxiliary output).

        is_log_prob: bool, default=False.
            Whether to return log probability or not.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        prob: torch.Tensor (n_samples, )
            (Log) probability of sampling actions.

        """
        x = self.calc_logits(
            context=context,
            query=query,
            cluster_centers=cluster_centers,
            calc_gradient=calc_gradient,
        )
        log_prob = F.log_softmax(x)
        prob = log_prob if is_log_prob else torch.exp(log_prob)
        return prob

    def calc_prob_given_action(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        action: torch.Tensor,
        cluster_centers: torch.Tensor,
        is_log_prob: bool = False,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Calculate the action choice probability of the given action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        action: torch.Tensor (n_samples, n_actions) or (n_samples, )
            (Onehot) cluster (vector) sampled for each input. (referred to `action` due to API consistency)

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_auxiliary_output)
            Low-dimensional embeddings of the cluster center (auxiliary output).

        is_log_prob: bool, default=False.
            Whether to return log probability or not.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        prob: torch.Tensor (n_samples, )
            (Log) probability of sampling actions.

        """
        is_onehot = len(action.shape) == 2
        action = action if is_onehot else F.one_hot(action, num_classes=self.n_actions)

        x = self.calc_logits(
            context=context,
            query=query,
            cluster_centers=cluster_centers,
            calc_gradient=calc_gradient,
        )
        log_prob = (F.log_softmax(x) * action).sum(dim=1)
        prob = log_prob if is_log_prob else torch.exp(log_prob)
        return prob


class BaseOutputRewardModel(nn.Module):
    """Base model for reward prediction based on auxiliary output.

    Imported as: :class:`src.policy.BaseOutputRewardModel`

    """

    def __init__(
        self,
    ):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
    ):
        """Calculate some logit values.

        Parameters
        -------
        inputs: torch.Tensor, shape (n_samples, dim_context + dim_query + dim_auxiliary_output)
            Input vectors.

        Return
        -------
        output_value: torch.Tensor (n_samples, )
            Predicted value of the given output.

        """
        raise NotImplementedError()

    def predict_value(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        auxiliary_output: torch.Tensor,
        calc_gradient: bool = False,
    ):
        """Predict value of given generated sentence.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        auxiliary_output: torch.Tensor, shape (n_samples, dim_auxiliary_output)
            Auxiliary output generated by the query and prompt.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        output_value: torch.Tensor (n_samples, )
            Predicted value of the given output.

        """
        inputs = torch.cat((context, query, auxiliary_output), dim=1)

        if calc_gradient:
            sentence_value = self(inputs)
        else:
            with torch.no_grad():
                sentence_value = self(inputs)

        return sentence_value

    def predict_values(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        action_list: torch.Tensor,
        auxiliary_output_generator: AuxiliaryOutputGenerator,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        n_sentences_to_approximate: int = 1,
        calc_gradient: bool = False,
    ):
        """Predict value of given generated sentence.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        action_list: torch.Tensor, shape (n_samples, dim_action_embedding)
            Mapping from action id to its embedding.

        auxiliary_output_generator: AuxiliaryOutputGenerator
            Generator of auxiliary output.

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        frozen_llms: BaseFrozenLLMs
            Frozen LLM to generate output sentence.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action_values: torch.Tensor or List[torch.Tensor] (n_samples, n_candidate_actions[i])
            Predicted values of the candidate actions.

        """
        n_samples = len(context)
        n_actions = len(action_list)

        if candidate_actions is None:
            predicted_values = torch.zeros(
                (n_samples, n_actions, n_sentences_to_approximate),
                device=self.device,
            )

            for j in range(n_sentences_to_approximate):
                for a in range(n_actions):
                    auxiliary_output_ = (
                        auxiliary_output_generator.sample_auxiliary_output(
                            query=query,
                            action_embedding=torch.tile(action_list[a], (n_samples, 1)),
                        )
                    )
                    predicted_values[:, a, j] = self.predict_value(
                        context=context,
                        query=query,
                        auxiliary_output=auxiliary_output_,
                        calc_gradient=calc_gradient,
                    )

            predicted_values = predicted_values.sum(dim=2)

        else:
            predicted_values = []
            for i in range(n_samples):
                predicted_values.append(
                    torch.zeros((len(candidate_actions[i]),), device=self.device)
                )

            for i in range(n_samples):
                context_ = torch.tile(context[i], (n_sentences_to_approximate, 1))
                query_ = torch.tile(query[i], (n_sentences_to_approximate, 1))

                for k, a in enumerate(candidate_actions[i]):
                    prompt_ = torch.tile(
                        action_list[a], (n_sentences_to_approximate, 1)
                    )
                    auxiliary_output_ = (
                        auxiliary_output_generator.sample_auxiliary_output(
                            query=query_,
                            prompt=prompt_,
                        )
                    )
                    predicted_values[i][a] = self.predict_value(
                        context=context_,
                        query=query_,
                        auxiliary_output=auxiliary_output_,
                        calc_gradient=calc_gradient,
                    ).mean()

        raise predicted_values


class BaseActionRewardModel(nn.Module):
    """Base model for reward prediction based on action.

    Imported as: :class:`src.policy.BaseActionRewardModel`

    Note
    -------
    The following parameter should be specified in init.

    n_actions: int
        Number of all available actions.

    action_list: torch.Tensor, default=None
        Mapping from action id to its embeddinds. Only applicable when using discrete prompts.

    """

    def __init__(
        self,
    ):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
    ):
        """Calculate some logit values.

        Parameters
        -------
        inputs: torch.Tensor, shape (n_samples, dim_context + dim_query + dim_action)
            Input vectors.

        Return
        -------
        action_value: torch.Tensor (n_samples, )
            Predicted value of the given action.

        """
        raise NotImplementedError()

    def predict_value(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        action: torch.Tensor,
        calc_gradient: bool = False,
    ):
        """Predict value of given prompt.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        action: torch.Tensor, shape (n_samples, )
             action.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action_value: torch.Tensor (n_samples, )
            Predicted value of the given action.

        """
        action = self.action_list[action]
        inputs = torch.cat((context, query, action), dim=1)

        if calc_gradient:
            action_value = self(inputs)
        else:
            with torch.no_grad():
                action_value = self(inputs)

        return action_value

    def predict_values(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        calc_gradient: bool = False,
    ):
        """Predict values of all candidate prompts.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        candidate_actions: torch.Tensor, shape (n_candidate_actions, )
            Candidate actions.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action_values: torch.Tensor or List[torch.Tensor] (n_samples, n_candidate_actions[i])
            Predicted values of the candidate actions.

        """
        n_samples = len(context)

        if candidate_actions is None:
            predicted_values = torch.zeros(
                (n_samples, self.n_actions), device=self.device
            )

            for a_ in range(self.n_actions):
                action_ = torch.full((n_samples,), a_, device=self.device)
                predicted_values[:, a_] = self.predict_value(
                    context=context,
                    query=query,
                    action=action_,
                    calc_gradient=calc_gradient,
                ).squeeze()

        else:
            predicted_values = []
            for i in range(n_samples):
                n_candidate_actions_ = len(candidate_actions[i])
                context_ = torch.tile(context[i], (n_candidate_actions_, 1))
                query_ = torch.tile(query[i], (n_candidate_actions_, 1))

                predicted_values.append(
                    self.predict_value(
                        context=context_,
                        query=query_,
                        action=candidate_actions[i],
                        calc_gradient=calc_gradient,
                    )
                )

        return predicted_values


class BaseKernelMarginalDensityModel(nn.Module):
    """Base kernel function.

    Imported as: :class:`src.policy.BaseKernelMarginalDensityModel`

    Note
    -------
    The following parameter should be specified in init.

    action_list: torch.Tensor, default=None
        Mapping from action id to its embeddinds. Only applicable when using discrete prompts.

    auxiliary_output_generator: AuxiliaryOutputGenerator, default=None
        Auxiliary output generator.

    kernel_function: Callable, default=gaussian_kernel
        Kernel function to use.

    kernel_kwargs: dict, default={"tau": 1.0}
        Kwargs of the kernel function.

    random_state: int, default=None
        Random state.

    """

    def __init__(
        self,
    ):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        inputs: torch.Tensor,
    ):
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
        raise NotImplementedError()

    @abstractmethod
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
        raise NotImplementedError()

    def calc_pairwise_weight(
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
        pairwise_weight: torch.Tensor, shape (n_samples, n_samples_to_approximate)
            Pairwise weight between the pivot output and sampled outputs.

        """
        pairwise_distance = self.calc_pairwise_distance(
            pivot_output=pivot_output,
            sampled_outputs=sampled_outputs,
        )
        pairwise_weight = self.kernel_function(
            distance=pairwise_distance,
            **self.kernel_kwargs,
        )
        return pairwise_weight

    def estimate_marginal_density(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        auxiliary_output: torch.Tensor,
        policy: Optional[Union[BasePolicy, BaseActionPolicyModel]] = None,
        n_samples_to_approximate: int = 100,
        calc_gradient: bool = False,
        use_monte_carlo: bool = False,
    ):
        """Calculate marginal density of the given policy via simulating the output generation.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        auxiliary_output: torch.Tensor, shape (n_samples, dim_auxiliary_output)
            Pivot outputs to calculate the marginal density.

        policy: Policy, default=None
            (Logging) policy. Required when using monte-carlo sampling.

        n_samples_to_approximate: int, default=100
            Number of samples used to approximate the marginal density.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        use_monte_carlo: bool, default=False
            Whether to use monte-carlo sampling.

        Return
        -------
        marginal_density: torch.Tensor, shape (n_samples, )
            Marginal density of the pivot output.

        """
        if use_monte_carlo:
            n_samples, dim_auxiliary_output = auxiliary_output.shape
            sampled_outputs = torch.zeros(
                (n_samples, n_samples_to_approximate, dim_auxiliary_output)
            )
            with tqdm(torch.arange(n_samples_to_approximate)) as pbar:
                for i, ch in enumerate(pbar):
                    pbar.set_description(
                        f"[simulating sentence generation to calculate marginal density: Epoch {i}]"
                    )

                    action_ = policy.sample_action(
                        context=context,
                        query=query,
                    )
                    sampled_outputs[
                        :, i
                    ] = self.auxiliary_output_generator.sample_auxiliary_output(
                        query=query,
                        action_embedding=self.action_list[action_],
                    )

            marginal_density = self.calc_pairwise_weight(
                pivot_output=auxiliary_output,
                sampled_outputs=sampled_outputs,
            ).mean(dim=1)

        else:  # function approximation
            inputs = torch.cat([context, query, auxiliary_output], dim=1)

            if calc_gradient:
                marginal_density = self(inputs)
            else:
                with torch.no_grad():
                    marginal_density = self(inputs)

        return marginal_density


@dataclass
class BaseClusteringModel(nn.Module):
    """Base model for clustering actions based on outputs.

    Imported as: :class:`src.policy.BaseClusteringModel`

    Note
    -------
    The following parameter should be specified in init.

    n_actions: int
        Number of all available actions.

    n_clusters: int
        Number of clusters.

    action_list: torch.Tensor, default=None
        Mapping from action id to its embeddinds. Only applicable when using discrete prompts.

    random_state: int, default=None
        Random state.

    """

    def __init__(
        self,
    ):
        super().__init__()

    @abstractmethod
    def sample_clustering(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Sample clustering centers.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        """
        raise NotImplementedError()

    @abstractmethod
    def retrieve_cluster(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        auxiliary_output: Optional[torch.Tensor] = None,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        idx: Optional[torch.Tensor] = None,
        resample_clustering: bool = False,
        **kwargs,
    ):
        """Retrieve cluster id given action or auxiliary_output.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        action: torch.Tensor, shape (n_samples, ), default=None
            Action chosen by the logging policy.

        auxiliary_output: torch.Tensor, shape (n_samples, dim_auxiliary_output), default=None
            Auxiliary output observed by the logging policy.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        idx: torch.Tensor, shape (n_subsamples, ), default=None
            Index for subsamples.

        resample_clustering: bool, default=False
            Whether to resample the clustering.

        Return
        -------
        cluster: torch.Tensor, shape (n_samples, )
            Index of the cluster the given action belongs to.

        """
        raise NotImplementedError()

    def retrieve_cluster_centers(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        logging_predicted_reward: torch.Tensor,
        idx: Optional[torch.Tensor] = None,
        resample_clustering: bool = False,
        **kwargs,
    ):
        """Retrieve all cluster centers.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        idx: torch.Tensor, shape (n_subsamples, ), default=None
            Index for subsamples.

        resample_clustering: bool, default=False
            Whether to resample the clustering.

        Return
        -------
        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_auxiliary_output)
            Low-dimensional embeddings of the cluster center (auxiliary output).

        """
        if resample_clustering:
            self.sample_clustering(
                context=context,
                query=query,
                logging_predicted_reward=logging_predicted_reward,
            )

        if idx is None:
            cluster_centers = self.cluster_centers
        else:
            cluster_centers = self.cluster_centers[idx]

        return cluster_centers

    def retrieve_candidate_actions_for_all_clusters(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        idx: Optional[torch.Tensor] = None,
        resample_clustering: bool = False,
        **kwargs,
    ):
        """Retrieve candidate actions of all clusters.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        idx: torch.Tensor, shape (n_subsamples, ), default=None
            Index for subsamples.

        resample_clustering: bool, default=False
            Whether to resample the clustering.

        Return
        -------
        candidate_actions: (nested) list of str
            Set of candidate actions of all clusters for each sample.

        """
        n_samples = len(context)

        if resample_clustering:
            self.sample_clustering(
                context=context,
                query=query,
                logging_predicted_reward=logging_predicted_reward,
            )

        if idx is None:
            candidate_actions = []
            for i in range(n_samples):
                candidate_actions_ = []
                for k in range(self.n_clusters):
                    candidate_actions_.append(torch.where(self.cluster_ids[i] == k)[0])
                candidate_actions.append(candidate_actions_)
        else:
            candidate_actions = []
            for i in idx:
                candidate_actions_ = []
                for k in range(self.n_clusters):
                    candidate_actions_.append(torch.where(self.cluster_ids[i] == k)[0])
                candidate_actions.append(candidate_actions_)

        return candidate_actions

    def retrieve_cluster_center(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        auxiliary_output: Optional[torch.Tensor] = None,
        cluster: Optional[torch.Tensor] = None,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        idx: Optional[torch.Tensor] = None,
        resample_clustering: bool = False,
        **kwargs,
    ):
        """Retrieve cluster center given action or cluster id.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        action: torch.Tensor, shape (n_samples, ), default=None
            Action chosen by the logging policy.

        auxiliary_output: torch.Tensor, shape (n_samples, dim_auxiliary_output), default=None
            Auxiliary output observed by the logging policy.

        cluster: torch.Tensor, shape (n_samples, ), default=None
            Index of the cluster the given action belongs to.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        idx: torch.Tensor, shape (n_subsamples, ), default=None
            Index for subsamples.

        resample_clustering: bool, default=False
            Whether to resample the clustering.

        Return
        -------
        cluster_centers: torch.Tensor, shape (n_samples, dim_auxiliary_output)
            Low-dimensional embeddings of the cluster center (auxiliary output) of the cluster the action belongs to.

        """
        cluster_centers = self.retrieve_cluster_centers(
            context=context,
            query=query,
            logging_predicted_reward=logging_predicted_reward,
            resample_clustering=resample_clustering,
            idx=idx,
        )

        if auxiliary_output is not None:
            cluster = self.retrieve_cluster(
                context=context,
                query=query,
                action=action,
                auxiliary_output=auxiliary_output,
                idx=idx,
            )

        n_samples = len(cluster_centers)
        return cluster_centers[torch.arange(n_samples), cluster]

    def retrieve_candidate_actions(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        auxiliary_output: Optional[torch.Tensor] = None,
        cluster: Optional[torch.Tensor] = None,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        idx: Optional[torch.Tensor] = None,
        resample_clustering: bool = False,
        **kwargs,
    ):
        """Retrieve candidate actions given action or cluster id.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        action: torch.Tensor, shape (n_samples, ), default=None
            Action chosen by the logging policy.

        auxiliary_output: torch.Tensor, shape (n_samples, dim_auxiliary_output), default=None
            Auxiliary output observed by the logging policy.

        cluster: torch.Tensor, shape (n_samples, ), default=None
            Index of the cluster the given action belongs to.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        idx: torch.Tensor, shape (n_subsamples, ), default=None
            Index for subsamples.

        resample_clustering: bool, default=False
            Whether to resample the clustering.

        Return
        -------
        candidate_actions: list of str
            Set of candidate actions of the cluster that the action belongs to.

        """
        n_samples = len(context)

        if resample_clustering:
            self.sample_clustering(
                context=context,
                query=query,
                logging_predicted_reward=logging_predicted_reward,
            )

        if action is not None or auxiliary_output is not None:
            cluster = self.retrieve_cluster(
                context=context,
                query=query,
                action=action,
                auxiliary_output=auxiliary_output,
                idx=idx,
            )

        if idx is None:
            candidate_actions = []
            for i in range(n_samples):
                candidate_actions.append(
                    torch.where(self.cluster_ids[i] == cluster[i])[0]
                )
        else:
            candidate_actions = []
            for i, id_ in enumerate(idx):
                candidate_actions.append(
                    torch.where(self.cluster_ids[id_] == cluster[i])[0]
                )

        return candidate_actions

    def calc_cluster_choice_prob(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        action_choice_prob: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        auxiliary_output: Optional[torch.Tensor] = None,
        cluster: Optional[torch.Tensor] = None,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        idx: Optional[torch.Tensor] = None,
        resample_clustering: bool = False,
        **kwargs,
    ):
        """Calculate cluster choice probability given action or cluster id.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        action_choice_prob: shape (n_samples, n_actions)
            Action choice probability of all actions.

        action: torch.Tensor, shape (n_samples, ), default=None
            Action chosen by the logging policy.

        auxiliary_output: torch.Tensor, shape (n_samples, dim_auxiliary_output), default=None
            Auxiliary output observed by the logging policy.

        cluster: torch.Tensor, shape (n_samples, ), default=None
            Index of the cluster the given action belongs to.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        idx: torch.Tensor, shape (n_subsamples, ), default=None
            Index for subsamples.

        resample_clustering: bool, default=False
            Whether to resample the clustering.

        Return
        -------
        cluster_choice_prob: torch.Tensor, shape (n_samples, ) or shape (n_samples, n_clusters)
            Cluster choice probability of all clusters.

        """
        n_samples = len(context)

        if resample_clustering:
            self.sample_clustering(
                context=context,
                query=query,
                logging_predicted_reward=logging_predicted_reward,
            )

        if action is None and cluster is None:
            candidate_actions = self.retrieve_candidate_actions_for_all_clusters(
                context=context,
                query=query,
                logging_predicted_reward=logging_predicted_reward,
                idx=idx,
            )

            if idx is None:
                cluster_choice_prob = torch.zeros(
                    (n_samples, self.n_clusters), device=self.device
                )
                for i in range(n_samples):
                    for j in range(self.n_clusters):
                        cluster_choice_prob[i, j] = action_choice_prob[i][
                            candidate_actions[i][j]
                        ].sum()
            else:
                n_subsamples = len(idx)
                cluster_choice_prob = torch.zeros(
                    (n_subsamples, self.n_clusters), device=self.device
                )
                for i, id_ in enumerate(idx):
                    for j in range(self.n_clusters):
                        cluster_choice_prob[i, j] = action_choice_prob[id_][
                            candidate_actions[i][j]
                        ].sum()

        else:
            candidate_actions = self.retrieve_candidate_actions(
                context=context,
                query=query,
                action=action,
                auxiliary_output=auxiliary_output,
                cluster=cluster,
                idx=idx,
            )

            if idx is None:
                cluster_choice_prob = torch.zeros((n_samples,), device=self.device)
                for i in range(n_samples):
                    cluster_choice_prob[i] = action_choice_prob[i][
                        candidate_actions[i]
                    ].sum()
            else:
                n_subsamples = len(idx)
                cluster_choice_prob = torch.zeros((n_subsamples,), device=self.device)
                for i, id_ in enumerate(idx):
                    cluster_choice_prob[i] = action_choice_prob[id_][
                        candidate_actions[i]
                    ].sum()

        return cluster_choice_prob

    def calc_cluster_variance(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        action_choice_prob: torch.Tensor,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        idx: Optional[torch.Tensor] = None,
        resample_clustering: bool = False,
        **kwargs,
    ):
        """Calculate cluster choice probability given action or cluster id.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: torch.Tensor, shape (n_samples, dim_query)
            Original query.

        action_choice_prob: shape (n_samples, n_actions)
            Action choice probability of all actions.

        action: torch.Tensor, shape (n_samples, ), default=None
            Action chosen by the logging policy.

        auxiliary_output: torch.Tensor, shape (n_samples, dim_auxiliary_output), default=None
            Auxiliary output observed by the logging policy.

        cluster: torch.Tensor, shape (n_samples, ), default=None
            Index of the cluster the given action belongs to.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        idx: torch.Tensor, shape (n_subsamples, ), default=None
            Index for subsamples.

        resample_clustering: bool, default=False
            Whether to resample the clustering.

        Return
        -------
        std: torch.Tensor, shape (n_samples, )
            Standard deviation of the inverse cluster choice probability of all clusters.

        """
        cluster_choice_prob = self.calc_cluster_choice_prob(
            context=context,
            query=query,
            action_choice_prob=action_choice_prob,
            logging_predicted_reward=logging_predicted_reward,
            resample_clustering=resample_clustering,
            idx=idx,
        )
        inv_cluster_choice_prob = 1 / cluster_choice_prob
        std = torch.std(inv_cluster_choice_prob, dim=1)
        return std
