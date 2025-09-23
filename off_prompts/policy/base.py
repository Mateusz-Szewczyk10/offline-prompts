"""Abstract base class for policies and models used to define policies."""
from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
from typing import Optional, List, Any, Union
from operator import itemgetter

import torch
from torch import nn
from torch.nn import functional as F

from ..dataset.base import BaseFrozenLLM
from ..types import Sentence, Tokens
from ..utils import tokenize, to_device


@dataclass
class BasePolicy(metaclass=ABCMeta):
    """Base class for policy.

    Imported as: :class:`off_prompts.policy.BasePolicy`

    Note
    -------
    1. The following parameter should be specified in init.

    n_actions: int
        Number of discrete prompts. (only when using discrete prompts)

    device: str, default="cuda"
        Device.

    """

    @abstractmethod
    def sample_multiple_actions(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        candidate_actions: Optional[List[torch.Tensor]] = None,
        return_action_type: str = "idx",
        n_actions_for_each: int = 1,
        replacement: bool = True,
        return_cpu_tensor: bool = True,
        **kwargs,
    ):
        """Sample multiple actions for each context and action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        return_action_type: {"idx", "prompt"}, default="idx"
            Type of action to return.

        n_actions_for_each: int, default=1
            Number of actions to sample.

        replacement: bool, default=True
            Whether to draw with replacement or not.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions_for_each)
            Action (vector) indicating which prompt is sampled for each input sentence.

        """
        raise NotImplementedError()

    @abstractmethod
    def sample_action(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        candidate_actions: Optional[List[torch.Tensor]] = None,
        return_action_type: str = "idx",
        return_cpu_tensor: bool = True,
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

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        return_action_type: {"idx", "onehot", "prompt"}, default="idx"
            Type of action to return.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions) or (n_samples, )
            (Onehot) action (vector) indicating which prompt is sampled for each input sentence.

        """
        raise NotImplementedError()

    @abstractmethod
    def sample_action_and_output_prob(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        candidate_actions: Optional[List[torch.Tensor]] = None,
        return_action_type: str = "idx",
        return_cpu_tensor: bool = True,
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

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        return_action_type: {"idx", "onehot", "prompt"}, default="idx"
            Type of action to return.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

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
        raise NotImplementedError()

    @abstractmethod
    def calc_action_choice_probability(
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        candidate_actions: Optional[List[torch.Tensor]] = None,
        return_cpu_tensor: bool = True,
        is_log_prob: bool = False,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Calculate the action choice probabilities for all candidate actions.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

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
        query: Union[Sentence, Tokens, torch.Tensor],
        action: torch.Tensor,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        return_cpu_tensor: bool = True,
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

        action: torch.Tensor, shape (n_samples, ) or (n_samples, dim_action)
            Discrete prompts (i.e., action).

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

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
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        reward_predictor: Optional[Any] = None,
        return_cpu_tensor: bool = True,
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

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        reward_predictor: BasePromptRewardModel, default=None
            (Pre-trained) action reward predictor. This must be given when using logit-based model.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

        return_per_sample: bool, default=False
            Whether to return policy value per sample or not (i.e., take mean).

        Return
        -------
        policy_value: float or torch.Tensor, shape (n_samples, )
            Predicted value of the policy.

        """
        raise NotImplementedError()


class BasePromptPolicyModel(nn.Module):
    """Base multi-head model for discrete action policy.

    Imported as: :class:`off_prompts.policy.BasePromptPolicyModel`

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
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
    ):
        """Calculate some logit values.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        Return
        -------
        action_logits: torch.Tensor (n_samples, n_actions)
            Some logit value of action on given input sentences.

        """
        raise NotImplementedError()

    def calc_logits(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Calculate value of all discrete actions.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action_logits: torch.Tensor (n_samples, )
            Some logit value of action on given input sentences.

        """
        if calc_gradient:
            logits = self(context, query)
        else:
            with torch.no_grad():
                logits = self(context, query)

        return logits

    def sample_multiple_actions(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        return_action_type: str = "idx",
        return_cpu_tensor: bool = True,
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

        return_action_type: {"idx", "prompt"}, default="idx"
            Type of action to return.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

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
            return_cpu_tensor=return_cpu_tensor,
        )
        action = torch.multinomial(
            action_choice_prob,
            num_samples=n_actions_for_each,
            replacement=replacement,
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
        return_action_type: str = "idx",
        return_cpu_tensor: bool = True,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Sample action using gumble softmax.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        return_action_type: {"idx", "onehot", "prompt"}, default="idx"
            Type of action to return.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions) or (n_samples, )
            (Onehot) action (vector) indicating which prompt is sampled for each input sentence.

        """
        x = self.calc_logits(context=context, query=query, calc_gradient=calc_gradient)
        action = F.gumbel_softmax(x, hard=True)

        if return_cpu_tensor:
            action = action.cpu()

        if return_action_type == "idx":
            action = action.argmax(dim=1)
        elif return_action_type == "prompt":
            action = action.argmax(dim=1)
            action = list(itemgetter(*action)(self.action_list))

        return action

    def sample_action_and_output_prob(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        return_action_type: str = "idx",
        return_cpu_tensor: bool = True,
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

        return_action_type: {"idx", "onehot", "prompt"}, default="idx"
            Type of action to return.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

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
        x = self.calc_logits(context=context, query=query, calc_gradient=calc_gradient)
        action = F.gumbel_softmax(x, hard=True)
        log_prob = (F.log_softmax(x) * action).sum(dim=1)

        prob = log_prob if is_log_prob else torch.exp(log_prob)

        if return_cpu_tensor:
            action = action.cpu()
            prob = prob.cpu()

        if return_action_type == "idx":
            action = action.argmax(dim=1)
        elif return_action_type == "prompt":
            action = action.argmax(dim=1)
            action = list(itemgetter(*action)(self.action_list))

        return action, prob

    def calc_action_choice_probability(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        return_cpu_tensor: bool = True,
        is_log_prob: bool = False,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Calculate the action choice probabilities for all candidate actions.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

        is_log_prob: bool, default=False.
            Whether to return log probability or not.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        prob: torch.Tensor (n_samples, )
            (Log) probability of sampling actions.

        """
        x = self.calc_logits(context=context, query=query, calc_gradient=calc_gradient)
        log_prob = F.log_softmax(x)
        prob = log_prob if is_log_prob else torch.exp(log_prob)

        if return_cpu_tensor:
            prob = prob.cpu()

        return prob

    def calc_prob_given_action(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        action: torch.Tensor,
        return_cpu_tensor: bool = True,
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

        action: torch.Tensor (n_samples, n_actions) or (n_samples, )
            (Onehot) action (vector) indicating which prompt is sampled for each input sentence.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

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
        action = to_device(action, device=self.device)

        x = self.calc_logits(context=context, query=query, calc_gradient=calc_gradient)
        log_prob = (F.log_softmax(x) * action).sum(dim=1)
        prob = log_prob if is_log_prob else torch.exp(log_prob)

        if return_cpu_tensor:
            prob = prob.cpu()

        return prob

    def predict_policy_value(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        predicted_reward: Optional[torch.Tensor] = None,
        reward_predictor: Optional[Any] = None,
        return_cpu_tensor: bool = True,
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

        reward_predictor: BaseActionRewardModel, default=None
            (Pre-trained) action reward predictor. This must be given when using logit-based model.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

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
                return_cpu_tensor=return_cpu_tensor,
            )

        prob = self.calc_action_choice_probability(
            context=context,
            query=query,
            return_cpu_tensor=return_cpu_tensor,
        )

        policy_value = (predicted_reward * prob).sum(dim=1)
        policy_value = policy_value if return_per_sample else policy_value.mean()
        return policy_value


class BaseClusterPolicyModel(nn.Module):
    """Base multi-head model for discrete cluster policy.

    Imported as: :class:`off_prompts.policy.BaseClusterPolicyModel`

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
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        cluster_centers: torch.Tensor,
    ):
        """Calculate some logit values.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_cluster_emb)
            Low-dimensional embeddings of the cluster center (sentence or prompt).

        Return
        -------
        action_logits: torch.Tensor (n_samples, n_actions)
            Some logit value of action on given input sentences.

        """
        raise NotImplementedError()

    def calc_logits(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        cluster_centers: torch.Tensor,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Calculate value of all discrete actions.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_cluster_emb)
            Low-dimensional embeddings of the cluster center (sentence or prompt).

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action_logits: torch.Tensor (n_samples, )
            Some logit value of action on given input sentences.

        """
        n_samples = len(context)
        cluster_centers.reshape((n_samples, -1))

        if calc_gradient:
            logits = self(context, query, cluster_centers)
        else:
            with torch.no_grad():
                logits = self(context, query, cluster_centers)

        return logits

    def sample_multiple_actions(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        cluster_centers: torch.Tensor,
        return_cpu_tensor: bool = True,
        n_actions_for_each: int = 1,
        replacement: bool = True,
        **kwargs,
    ):
        """Sample multiple clusters (actions) for each context and action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_cluster_emb)
            Low-dimensional embeddings of the cluster center (sentence or prompt).

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

        n_actions_for_each: int, default=1
            Number of actions to sample.

        replacement: bool, default=True
            Whether to draw with replacement or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions_for_each)
            Action (vector) indicating which prompt is sampled for each input sentence.

        """
        action_choice_prob = self.calc_action_choice_probability(
            context=context,
            query=query,
            cluster_centers=cluster_centers,
            return_cpu_tensor=return_cpu_tensor,
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
        query: Union[Sentence, Tokens, torch.Tensor],
        cluster_centers: torch.Tensor,
        return_action_type: str = "idx",
        return_cpu_tensor: bool = True,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Sample cluster (action) using gumble softmax.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_cluster_emb)
            Low-dimensional embeddings of the cluster center (sentence or prompt).

        return_action_type: {"idx", "onehot"}, default="idx"
            Type of action to return.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action: torch.Tensor (n_samples, n_actions) or (n_samples, )
            (Onehot) cluster (vector) indicating which prompt is sampled for each input sentence.

        """
        x = self.calc_logits(
            context=context,
            query=query,
            cluster_centers=cluster_centers,
            calc_gradient=calc_gradient,
        )
        action = F.gumbel_softmax(x, hard=True)

        if return_cpu_tensor:
            action = action.cpu()

        if return_action_type == "idx":
            action = action.argmax(dim=1)

        return action

    def sample_action_and_output_prob(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        cluster_centers: torch.Tensor,
        return_action_type: str = "idx",
        return_cpu_tensor: bool = True,
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

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_cluster_emb)
            Low-dimensional embeddings of the cluster center (sentence or prompt).

        return_action_type: {"idx", "onehot"}, default="idx"
            Type of action to return.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

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

        if return_cpu_tensor:
            action = action.cpu()
            prob = prob.cpu()

        return action, prob

    def calc_action_choice_probability(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        cluster_centers: torch.Tensor,
        return_cpu_tensor: bool = True,
        is_log_prob: bool = False,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Calculate the action choice probabilities for all candidate actions.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_cluster_emb)
            Low-dimensional embeddings of the cluster center (sentence or prompt).

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

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

        if return_cpu_tensor:
            prob = prob.cpu()

        return prob

    def calc_prob_given_action(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        action: torch.Tensor,
        cluster_centers: torch.Tensor,
        return_cpu_tensor: bool = True,
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

        action: torch.Tensor (n_samples, n_actions) or (n_samples, )
            (Onehot) cluster (vector) sampled for each input. (referred to `action` due to API consistency)

        cluster_centers: torch.Tensor, shape (n_samples, n_clusters, dim_cluster_emb)
            Low-dimensional embeddings of the cluster center (sentence or prompt).

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

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
        action = to_device(action, device=self.device)

        x = self.calc_logits(
            context=context,
            query=query,
            cluster_centers=cluster_centers,
            calc_gradient=calc_gradient,
        )
        log_prob = (F.log_softmax(x) * action).sum(dim=1)
        prob = log_prob if is_log_prob else torch.exp(log_prob)

        if return_cpu_tensor:
            prob = prob.cpu()

        return prob


class BaseSentenceRewardModel(nn.Module):
    """Base model for reward prediction based on sentence.

    Imported as: :class:`off_prompts.policy.BaseSentenceRewardModel`

    """

    def __init__(
        self,
    ):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        sentence: Union[Sentence, Tokens, torch.Tensor],
    ):
        """Calculate some logit values.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        sentence: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_sentence)
            Input sentences (raw text).

        Return
        -------
        sentence_value: torch.Tensor (n_samples, )
            Predicted value of the given sentence.

        """
        raise NotImplementedError()

    def predict_value(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        sentence: Union[Sentence, Tokens, torch.Tensor],
        return_cpu_tensor: bool = True,
        calc_gradient: bool = False,
    ):
        """Predict value of given generated sentence.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        sentence: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_sentence)
            Sentence generated by the query and prompt.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        sentence_value: torch.Tensor (n_samples, )
            Predicted value of the given sentence.

        """
        if calc_gradient:
            sentence_value = self(context, query, sentence)
        else:
            with torch.no_grad():
                sentence_value = self(context, query, sentence)

        if return_cpu_tensor:
            sentence_value = sentence_value.cpu()

        return sentence_value

    def predict_values(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        query_for_frozen_llm: Union[Sentence, Tokens, torch.Tensor],
        action_list: Union[List[str], torch.Tensor],
        frozen_llm: BaseFrozenLLM,
        candidate_actions: Optional[List[torch.Tensor]] = None,
        return_cpu_tensor: bool = True,
        n_sentences_to_approximate: int = 1,
        calc_gradient: bool = False,
    ):
        """Predict value of given generated sentence.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        query_for_frozen_llm: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users (used to generate sentences with the frozen LLM).

        action_list: list of str or torch.Tensor, shape (n_actions, dim_action_emb)
            Mapping from action id to discrete prompts.

        frozen_llm: BaseFrozenLLM
            Frozen LLM to generate output sentence.

        candidate_actions: list of torch.Tensor, default=None.
            Candidate set of actions for each given context.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

        n_sentences_to_approximate: int, default=1
            Number of sentences used to approximate prompt value.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action_values: torch.Tensor or List[torch.Tensor] (n_samples, n_candidate_actions[i])
            Predicted values of the candidate actions.

        """
        n_samples = len(context)
        n_actions = len(action_list)

        if isinstance(action_list, list):
            action_list = self.prompt_encoder.encode(action_list)

        if isinstance(query, (list, dict)):
            query = self.query_encoder.encode(query)

        if isinstance(query_for_frozen_llm, list):
            query_for_frozen_llm = tokenize(
                query_for_frozen_llm,
                tokenizer=self.frozen_llm.tokenizer,
                tokenizer_kwargs=self.frozen_llm.tokenizer_kwargs,
                device=self.frozen_llm.device,
            )

        if candidate_actions is None:
            predicted_values = torch.zeros(
                (n_samples, n_actions, n_sentences_to_approximate),
                device=self.device,
            )

            for a in range(n_actions):
                prompt = torch.full((n_samples,), action_list[a], device=self.device)

                for j in range(n_sentences_to_approximate):
                    generated_sentence_ = frozen_llm.generate_output_sentence(
                        query=query_for_frozen_llm,
                        prompt=prompt,
                    )
                    predicted_values[:, a, j] = self.predict_value(
                        context=context,
                        query=query,
                        sentence=generated_sentence_,
                        return_cpu_tensor=False,
                        calc_gradient=calc_gradient,
                    )

            predicted_values = predicted_values.sum(dim=2)

            if return_cpu_tensor:
                predict_values = predict_values.cpu()

        else:
            predicted_values = []
            for i in range(n_samples):
                predicted_values.append(
                    torch.zeros((len(candidate_actions[i]),), device=self.device)
                )

            for i in range(n_samples):
                context_ = torch.tile(context[i], (n_sentences_to_approximate, 1))
                query_ = torch.tile(query[i], (n_sentences_to_approximate, 1))

                query_for_frozen_llm_ = {}
                for key in query_for_frozen_llm:
                    query_for_frozen_llm_[key] = torch.tile(
                        query_for_frozen_llm[key][i], (n_sentences_to_approximate, 1)
                    )

                for k, a in enumerate(candidate_actions[i]):
                    prompt_ = torch.full(
                        (n_samples,), action_list[a], device=self.device
                    )

                    generated_sentence_ = frozen_llm.generate_output_sentence(
                        query=query_for_frozen_llm_,
                        prompt=prompt_,
                    )
                    predicted_values[i][a] = self.predict_value(
                        context=context_,
                        query=query_,
                        generated_sentence=generated_sentence_,
                        return_cpu_tensor=False,
                        calc_gradient=calc_gradient,
                    ).mean()

                if return_cpu_tensor:
                    predicted_values[i] = predict_values[i].cpu()

        raise predicted_values


class BasePromptRewardModel(nn.Module):
    """Base model for reward prediction based on prompt.

    Imported as: :class:`off_prompts.policy.BasePromptRewardModel`

    Note
    -------
    The following parameter should be specified in init.

    n_actions: int
        Number of all available actions.

    action_list: list of str, default=None
        Mapping from action id to discrete prompts. Only applicable when using discrete prompts.

    """

    def __init__(
        self,
    ):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        prompt: Union[Sentence, Tokens, torch.Tensor],
    ):
        """Calculate some logit values.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        prompt: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Prompt specified by policies.

        Return
        -------
        action_value: torch.Tensor (n_samples, )
            Predicted value of the given action (i.e., prompt).

        """
        raise NotImplementedError()

    def predict_value(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        action: Optional[torch.Tensor] = None,
        return_cpu_tensor: bool = True,
        calc_gradient: bool = False,
    ):
        """Predict value of given prompt.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        action: torch.Tensor, shape (n_samples, ), default=None
            Index of discrete prompts (i.e., action).

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action_value: torch.Tensor (n_samples, )
            Predicted value of the given action.

        """
        prompt = self.prompt_embeddings[action]

        if calc_gradient:
            action_value = self(context, query, prompt)
        else:
            with torch.no_grad():
                action_value = self(context, query, prompt)

        if return_cpu_tensor:
            action_value = action_value.cpu()

        return action_value

    def predict_values(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        candidate_actions: Optional[List[torch.Tensor]] = None,
        return_cpu_tensor: bool = True,
        calc_gradient: bool = False,
    ):
        """Predict values of all candidate prompts.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        candidate_actions: torch.Tensor, shape (n_candidate_actions, )
            Discrete prompts (i.e., action).

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        action_values: torch.Tensor or List[torch.Tensor] (n_samples, n_candidate_actions[i])
            Predicted values of the candidate actions.

        """
        n_samples = len(context)

        if isinstance(query, (list, dict)):
            query = self.query_encoder.encode(query)

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

            if return_cpu_tensor:
                predicted_values = predicted_values.cpu()

        else:
            predicted_values = []
            for i in range(n_samples):
                n_candidate_actions_ = len(candidate_actions[i])
                context_ = torch.tile(context[i], (n_candidate_actions_, 1))
                query_ = torch.tile(query[i], (n_candidate_actions_, 1))

                predicted_value_ = self.predict_value(
                    context=context_,
                    query=query_,
                    action=candidate_actions[i],
                    calc_gradient=calc_gradient,
                )

                if return_cpu_tensor:
                    predicted_value_ = predicted_value_.cpu()

                predicted_values.append(predicted_value_)

        return predicted_values


class BaseKernelMarginalDensityModel(nn.Module):
    """Base kernel function.

    Imported as: :class:`off_prompts.policy.BaseKernelMarginalDensityModel`

    Note
    -------
    The following parameter should be specified in init.

    action_list: torch.Tensor, default=None
        Mapping from action id to its embeddinds. Only applicable when using discrete prompts.

    frozen_llm: BaseFrozenLLM, default=None
        Frozen LLM.

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
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        sentence: Union[Sentence, Tokens, torch.Tensor],
    ):
        """Calculate some logit values.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        sentence: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_sentence)
            Input sentences (raw text).

        Return
        -------
        marginal_density: torch.Tensor (n_samples, )
            Predicted value of the given action.

        """
        raise NotImplementedError()

    @abstractmethod
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
        pivot_sentence: Sentence or Tokens or torch.Tensor, shape (n_samples, )
            Pivot output observed in the logged data.

        sampled_sentences: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, n_samples_to_approximate)
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
        raise NotImplementedError()

    def calc_pairwise_weight(
        self,
        pivot_sentence: Union[Sentence, Tokens, torch.Tensor],
        sampled_sentences: Union[Sentence, Tokens, torch.Tensor],
        context: Optional[torch.Tensor] = None,
        query: Optional[Union[Sentence, Tokens, torch.Tensor]] = None,
    ):
        """Calculate pairwise distance between pivot output and sampled outputs.

        Parameters
        -------
        pivot_sentence: Sentence or Tokens or torch.Tensor, shape (n_samples, dim_auxiliary_output)
            Pivot output observed in the logged data.

        sampled_sentences: Sentence or Tokens or torch.Tensor, shape (n_samples, dim_auxiliary_output) or (n_samples, n_samples_to_approximate, dim_auxiliary_output)
            Sampled outputs that are used to calculate the marginalized density.

        context: torch.Tensor, shape (n_samples, dim_context), default=None
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query), default=None
            Original keywords of input sentence specified by users.

        Return
        -------
        pairwise_weight: torch.Tensor, shape (n_samples, n_samples_to_approximate)
            Pairwise weight between the pivot output and sampled outputs.

        """
        pairwise_distance = self.calc_pairwise_distance(
            pivot_sentence=pivot_sentence,
            sampled_sentences=sampled_sentences,
            context=context,
            query=query,
        )
        pairwise_weight = self.kernel_function(
            distance=pairwise_distance,
            **self.kernel_kwargs,
        )
        return pairwise_weight

    def estimate_marginal_density(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        sentence: Union[Sentence, Tokens, torch.Tensor],
        calc_gradient: bool = False,
    ):
        """Calculate marginal density of the given policy via simulating the output generation.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        sentence: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_sentence)
            Pivot outputs to calculate the marginal density.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        marginal_density: torch.Tensor, shape (n_samples, )
            Marginal density of the pivot output.

        """
        if calc_gradient:
            marginal_density = self(context, query, sentence)
        else:
            with torch.no_grad():
                marginal_density = self(context, query, sentence)

        return marginal_density


class BaseClusteringModel(nn.Module):
    """Base model for clustering actions based on outputs.

    Imported as: :class:`off_prompts.policy.BaseClusteringModel`

    Note
    -------
    The following parameter should be specified in init.

    n_actions: int
        Number of all available actions.

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
        query: Union[Sentence, Tokens, torch.Tensor],
        logging_predicted_reward: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Sample clustering centers and assignment of each action to clusters.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions.

        """
        raise NotImplementedError()

    @abstractmethod
    def retrieve_cluster(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        action: Optional[torch.Tensor] = None,
        sentence: Optional[Union[Sentence, Tokens, torch.Tensor]] = None,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        idx: Optional[torch.Tensor] = None,
        resample_clustering: bool = False,
        **kwargs,
    ):
        """Retrieve cluster id given action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        action: torch.Tensor, shape (n_samples, ), default=None
            Action chosen by the logging policy.

        sentence: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_sentence), default=None
            Sentence generated by the query and prompt with logging policy.

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
        query: Union[Sentence, Tokens, torch.Tensor],
        logging_predicted_reward: Optional[torch.Tensor] = None,
        idx: Optional[torch.Tensor] = None,
        resample_clustering: bool = False,
        **kwargs,
    ):
        """Retrieve all cluster centers.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            User contexts (e.g., demographic features).

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

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
        query: Union[Sentence, Tokens, torch.Tensor],
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

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

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
        query: Union[Sentence, Tokens, torch.Tensor],
        action: Optional[torch.Tensor] = None,
        sentence: Optional[Union[Sentence, Tokens, torch.Tensor]] = None,
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

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        action: torch.Tensor, shape (n_samples, ), default=None
            Action chosen by the logging policy.

        sentence: Sentence or Tokens or torch.Tensor, shape (n_samples, ), default=None
            Sentence generated by the query and prompt with logging policy.

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

        if action is not None or sentence is not None:
            cluster = self.retrieve_cluster(
                context=context,
                query=query,
                action=action,
                sentence=sentence,
                idx=idx,
            )

        n_samples = len(cluster_centers)
        cluster_center = cluster_centers[torch.arange(n_samples), cluster]
        return cluster_center

    def retrieve_candidate_actions(
        self,
        context: torch.Tensor,
        query: Union[Sentence, Tokens, torch.Tensor],
        action: Optional[torch.Tensor] = None,
        sentence: Optional[Union[Sentence, Tokens, torch.Tensor]] = None,
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

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        action: torch.Tensor, shape (n_samples, ), default=None
            Action chosen by the logging policy.

        sentence: Sentence or Tokens, shape (n_samples, ), default=None
            Sentence generated by the query and prompt with logging policy.

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

        if action is not None or sentence is not None:
            cluster = self.retrieve_cluster(
                context=context,
                query=query,
                action=action,
                sentence=sentence,
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
        query: Union[Sentence, Tokens, torch.Tensor],
        action_choice_prob: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        sentence: Optional[Union[Sentence, Tokens, torch.Tensor]] = None,
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

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        action_choice_prob: torch.Tensor, shape (n_samples, n_actions)
            Action choice probability of all actions.

        action: torch.Tensor, shape (n_samples, ), default=None
            Action chosen by the logging policy.

        sentence: Sentence or Tokens, shape (n_samples, ), default=None
            Sentence generated by the query and prompt with logging policy.

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
        cluster_choice_prob: torch.Tensor, shape (n_samples, ) or (n_samples, n_clusters)
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
                sentence=sentence,
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
        query: Union[Sentence, Tokens, torch.Tensor],
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

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, ) or (n_samples, dim_query)
            Original keywords of input sentence specified by users.

        action_choice_prob: shape (n_samples, n_actions)
            Action choice probability of all action.

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
