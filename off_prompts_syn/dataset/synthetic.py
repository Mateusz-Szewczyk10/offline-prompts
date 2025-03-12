"""Class to handle synthetic dataset generation."""
from dataclasses import dataclass
from typing import Optional, Union

import torch
from sklearn.utils import check_scalar

from .function import (
    ContextQueryGenerator,
    CandidateActionsGenerator,
    AuxiliaryOutputGenerator,
    RewardSimulator,
)
from ..policy.base import (
    BasePolicy,
    BaseActionPolicyModel,
    BaseClusterPolicyModel,
)
from ..utils import torch_seed


Policy = Union[BasePolicy, BaseActionPolicyModel, BaseClusterPolicyModel]


@dataclass
class SyntheticDataset:
    """Base class for logged dataset.

    Imported as: :class:`off_prompts_syn.dataset.SyntheticDataset`

    Note
    -------
    This class substitute the language part of the benchmark dataset with numerical features.

    context: x (d_x dimensional vector)
    query: q   (d_q dimensional vector)
    prompt (action): a  (d_a dimensional vector)
    prompt (action) embedding: e (d_e dimensional vector)

    output sentence (auxiliary output): o (d_o dimensional vector)
    o = M q + N e where M is a (d_o, d_x) matrix and N is a (d_o, d_e) matrix

    expected reward: E[r] (float)
    E[r] = x Q o + q R o where Q is a (d_x, d_o) maxtrix and R is a (d_q, d_o) matrix

    Parameters
    -------
    n_actions: int, default=1000 (> 0)
        Number of discrete actions.

    dim_context: int, default=5 (> 0)
        Dimension of context.

    dim_query: int, default=5 (> 0)
        Dimension of query.

    dim_action_embdding: int, default=5 (> 0)
        Dimension of action embedding.

    dim_auxiliary_output: int, default=5 (> 0)
        Dimension of auxiliary output.

    reward_type: {"binary", "continuous"}, default="continuous"
        Whether to sample binary reward or continuous reward.

    reward_std: float, default=1.0
        Standard deviation of the reward.

    device: str, default="cuda:0"
        Device.

    random_state: int, default=None
        Random state.

    """

    n_actions: int = 1000
    dim_context: int = 5
    dim_query: int = 5
    dim_action_embedding: int = 5
    dim_auxiliary_output: int = 5
    context_query_generator: Optional[ContextQueryGenerator] = None
    candidate_action_generator: Optional[CandidateActionsGenerator] = None
    auxiliary_output_generator: Optional[AuxiliaryOutputGenerator] = None
    reward_simulator: Optional[RewardSimulator] = None
    reward_type: str = "continuous"
    reward_std: float = 1.0
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        check_scalar(self.n_actions, "n_actions", target_type=int, min_val=1)
        check_scalar(self.dim_context, "dim_context", target_type=int, min_val=1)
        check_scalar(self.dim_query, "dim_query", target_type=int, min_val=1)
        check_scalar(
            self.dim_action_embedding,
            "dim_action_embedding",
            target_type=int,
            min_val=1,
        )
        check_scalar(
            self.dim_auxiliary_output,
            "dim_auxiliary_output",
            target_type=int,
            min_val=1,
        )
        check_scalar(self.reward_std, "reward_std", target_type=float, min_val=0.0)

        if self.random_state is not None:
            torch_seed(self.random_state, device=self.device)

        if self.context_query_generator is None:
            self.context_query_generator = ContextQueryGenerator(
                dim_context=self.dim_context,
                dim_query=self.dim_query,
                device=self.device,
                random_state=self.random_state,
            )
        else:
            if self.context_query_generator.dim_context != self.dim_context:
                raise ValueError(
                    "Expected context_query_generator.dim_context == self.dim_context, but found False"
                )
            if self.context_query_generator.dim_query != self.dim_query:
                raise ValueError(
                    "Expected context_query_generator.dim_query == self.dim_query, but found False"
                )

        if self.candidate_action_generator is None:
            self.candidate_action_generator = CandidateActionsGenerator(
                n_actions=self.n_actions,
                dim_action_embedding=self.dim_action_embedding,
                device=self.device,
                random_state=self.random_state,
            )
        else:
            if self.candidate_action_generator.n_actions != self.n_actions:
                raise ValueError(
                    "Expected candidate_action_generator.n_actions == self.n_actions, but found False"
                )
            if (
                self.candidate_action_generator.dim_action_embedding
                != self.dim_action_embedding
            ):
                raise ValueError(
                    "Expected candidate_action_generator.dim_action_embedding == self.dim_action_embedding, but found False"
                )

        if self.auxiliary_output_generator is None:
            self.auxiliary_output_generator = AuxiliaryOutputGenerator(
                dim_query=self.dim_query,
                dim_action_embedding=self.dim_action_embedding,
                dim_auxiliary_output=self.dim_auxiliary_output,
                device=self.device,
                random_state=self.random_state,
            )
        else:
            if self.auxiliary_output_generator.dim_query != self.dim_query:
                raise ValueError(
                    "Expected auxiliary_output_generator.dim_query == self.dim_query, but found False"
                )
            if (
                self.auxiliary_output_generator.dim_action_embedding
                != self.dim_action_embedding
            ):
                raise ValueError(
                    "Expected auxiliary_output_generator.dim_action_embedding == self.dim_action_embedding, but found False"
                )
            if (
                self.auxiliary_output_generator.dim_auxiliary_output
                != self.dim_auxiliary_output
            ):
                raise ValueError(
                    "Expected auxiliary_output_generator.dim_auxiliary_output == self.dim_auxiliary_output, but found False"
                )

        if self.reward_simulator is None:
            self.reward_simulator = RewardSimulator(
                dim_context=self.dim_context,
                dim_query=self.dim_query,
                dim_auxiliary_output=self.dim_auxiliary_output,
                device=self.device,
                random_state=self.random_state,
            )
        else:
            if self.reward_simulator.dim_context != self.dim_context:
                raise ValueError(
                    "Expected reward_simulator.dim_context == self.dim_context, but found False"
                )
            if self.reward_simulator.dim_query != self.dim_query:
                raise ValueError(
                    "Expected reward_dimulator.dim_query == self.dim_query, but found False"
                )
            if self.reward_simulator.dim_auxiliary_output != self.dim_auxiliary_output:
                raise ValueError(
                    "Expected reward_dimulator.dim_auxiliary_output == self.dim_auxiliary_output, but found False"
                )

        self.action_list = (
            self.candidate_action_generator.action_embedding
        )  # shape (n_actions, dim_action_embedding)

    def _sample_reward(self, expected_reward: torch.Tensor):
        """Sample reward.

        Parameters
        -------
        expected_reward: torch.Tensor, shape (n_samples, )
            Expected reward predicted by some frozen LLMs (i.e., reward simulator).

        Return
        -------
        reward: torch.Tensor, shape (n_samples, )
            Either binary or continuous reward.

        """
        if self.reward_type == "binary":
            reward = torch.bernoulli(expected_reward)
        elif self.reward_type == "continuous":
            reward_std = torch.full_like(expected_reward, self.reward_std)
            reward = torch.normal(expected_reward, reward_std)
        return reward

    def sample_dataset(
        self,
        policy: Policy,
        n_samples: int = 10000,
        is_oracle_policy: bool = False,
        is_oracle_clustering_logging_policy: bool = False,
        return_action_choice_prob: bool = False,
        return_meta_data: bool = False,
    ):
        """Sample dataset given data collection policy.

        Parameters
        -------
        policy: Policy
            Policy that chooses discrete prompt (index) as action.

        n_samples: int, default=1000 (> 0)
            Number of samples.

        is_oracle_policy: bool, default=False
            Whether to use expected reward to sample action using policy.

        is_oracle_clustering_logging_policy: bool, default=False
            Whether the logging policy uses oracle expected reward.

        return_action_choice_prob: bool, default=False
            Whether to record action choice probability of the data collection policy.

        return_meta_data: bool, default=False
            Whether to record meta data including data size, type of action, etc.

        Return
        -------
        logged_dataset: dict
            Dictionary containing dataset with the following keys:

            .. code-block:: python

                key: [
                    context,
                    query,
                    action,
                    auxiliary_output,
                    expected_reward,
                    reward,
                    logging_policy,
                ]

            context: torch.Tensor, shape (n_samples, dim_context)
                Feature vector of each user.

            query: Sentence, shape (n_samples, dim_query)
                Query vector given by users.

            action: torch.Tensor, shape (n_samples, )
                Discrete actions (index) chosen by the given policy.

            auxiliary_output: Sentence, shape (n_samples, dim_auxiliary_output)
                Auiliary output vector generated by query and action.

            expected_reward: torch.Tensor, shape (n_samples, )
                Expected reward simulated from context, query, auxiliary_output.

            reward: torch.Tensor, shape (n_samples, )
                Either binary or continuous reward.

            logging_policy: Policy
                Policy that chooses either discrete or continuous (soft) prompt as action.

        meta_data: dict
            Dictionary containing setting of simulation with the following keys:

            .. code-block:: python

                key: [
                    size,
                    reward_type,
                    reward_std,
                    action_list,
                    dim_context,
                    dim_query,
                    dim_action_embedding,
                    dim_auxiliary_output,
                ]

            size: int
                Data size (i.e., number of samples).

            reward_type: {"binary", "continuous"}
                Whether to sample binary or continuous rewards.

            reward_std: float
                Noise level of reward. Applicable only when `reward_type=="continuous"`.

            action_list: torch.Tensor
                Mapping from discrete action to embedding.

            dim_context: int
                Dimension of the context.

            dim_query: int
                Dimension of the query.

            dim_action_embedding: int
                Dimension of the action embedding.

            dim_auxiliary_output: int
                Dimension of the auxiliary output.

        """

        # self._check_policy(policy)

        context_and_query = self.context_query_generator.sample_context_and_query(
            n_samples=n_samples
        )
        context = context_and_query[:, : self.dim_context]
        query = context_and_query[:, self.dim_context :]

        predicted_reward = None
        clustering_logging_predicted_reward = None

        if is_oracle_policy or is_oracle_clustering_logging_policy:
            expected_reward_all = self.calc_expected_reward_for_all_actions(
                context=context,
                query=query,
            )
        else:
            expected_reward_all = None

        if is_oracle_policy:
            predicted_reward = expected_reward_all
        else:
            predicted_reward = None

        if is_oracle_clustering_logging_policy:
            clustering_logging_predicted_reward = expected_reward_all
        else:
            clustering_logging_predicted_reward = None

        action = policy.sample_action(
            context=context,
            query=query,
            predicted_reward=predicted_reward,
            clustering_logging_predicted_reward=clustering_logging_predicted_reward,
        )
        action_embedding = self.action_list[action]

        auxiliary_output = self.auxiliary_output_generator.sample_auxiliary_output(
            query=query,
            action_embedding=action_embedding,
        )

        expected_reward = self.reward_simulator.calc_expected_reward(
            context=context,
            query=query,
            auxiliary_output=auxiliary_output,
        )
        reward = self._sample_reward(expected_reward=expected_reward)

        if return_action_choice_prob:
            action_choice_prob = policy.calc_prob_given_action(
                context=context,
                query=query,
                action=action,
                predicted_reward=expected_reward_all,
            )
        else:
            action_choice_prob = None

        logged_dataset = {
            "context": context,
            "query": query,
            "action": action,
            "action_choice_prob": action_choice_prob,
            "auxiliary_output": auxiliary_output,
            "expected_reward": expected_reward,
            "reward": reward,
            "logging_policy": policy,
        }

        if return_meta_data:
            meta_data = {
                "size": n_samples,
                "reward_type": self.reward_type,
                "reward_std": self.reward_std,
                "action_list": self.action_list,
                "dim_context": self.dim_context,
                "dim_query": self.dim_query,
                "dim_action_embedding": self.dim_action_embedding,
                "dim_auxiliary_output": self.dim_auxiliary_output,
            }
            output = (logged_dataset, meta_data)

        else:
            output = logged_dataset

        return output

    def calc_expected_reward_given_action(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        action: torch.Tensor,
        n_outputs_to_approximate: int = 10,
    ):
        """Calculate expected reward for the given action.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            Feature vector of each user.

        query: torch.Tensor, shape (n_samples, dim_query)
            Query vector given by users.

        action: torch.Tensor, shape (n_samples, )
            Action.

        n_outputs_to_approximate: int, default=10
            Number of outputs to approzimate the expected reward.

        Return
        -------
        expected_reward: torch.Tensor, shape (n_samples, n_actions)
            Expected reward of all candidate actions.

        """
        n_samples = len(context)
        expected_reward = torch.zeros(
            (n_samples, n_outputs_to_approximate), device=self.device
        )

        for j in range(n_outputs_to_approximate):
            auxiliary_output_ = self.auxiliary_output_generator.sample_auxiliary_output(
                query=query,
                action_embedding=self.action_list[action],
            )
            expected_reward[:, j] = self.reward_simulator.calc_expected_reward(
                context=context,
                query=query,
                auxiliary_output=auxiliary_output_,
            )
        return expected_reward.mean(axis=1)

    def calc_expected_reward_for_all_actions(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        n_outputs_to_approximate: int = 10,
    ):
        """Calculate expected reward for all actions.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            Feature vector of each user.

        query: torch.Tensor, shape (n_samples, dim_query)
            Query vector given by users.

        n_outputs_to_approximate: int, default=10
            Number of outputs to approzimate the expected reward.

        Return
        -------
        expected_reward: torch.Tensor, shape (n_samples, n_actions)
            Expected reward of all candidate actions.

        """
        n_samples = len(context)
        expected_reward = torch.zeros(
            (n_samples, self.n_actions, n_outputs_to_approximate),
            device=self.device,
        )

        for j in range(n_outputs_to_approximate):
            auxiliary_output_ = self.auxiliary_output_generator.sample_auxiliary_output(
                query=query,
                action_embedding=self.action_list,
                enumerate_actions=True,
            )
            for action in range(self.n_actions):
                expected_reward[
                    :, action, j
                ] = self.reward_simulator.calc_expected_reward(
                    context=context,
                    query=query,
                    auxiliary_output=auxiliary_output_[:, action],
                )

        return expected_reward.mean(axis=2)

    def calc_expected_policy_value(
        self,
        policy: Policy,
        is_oracle_policy: bool = False,
        is_oracle_clustering_logging_policy: bool = False,
        n_samples_to_approximate: int = 10000,
    ):
        """Sample dataset given data collection policy.

        Parameters
        -------
        policy: Policy
            Policy that chooses discrete prompt (index) as action.

        is_oracle_policy: bool, default=False
            Whether to use expected reward to sample action using policy.

        is_oracle_clustering_logging_policy: bool, default=False
            Whether the logging policy in the clustering of two-stage policy uses oracle expected reward.

        n_samples_to_approximate: int
            Number of samples to approximate the policy value.

        Return
        -------
        policy value: dict
            Approximated policy value.

        """
        logged_dataset = self.sample_dataset(
            policy=policy,
            is_oracle_policy=is_oracle_policy,
            is_oracle_clustering_logging_policy=is_oracle_clustering_logging_policy,
            n_samples=n_samples_to_approximate,
        )
        policy_value = logged_dataset["expected_reward"].mean().item()
        return policy_value

    def sample_reward_given_action(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        action: torch.Tensor,
    ):
        """Sample reward given action chosen by a policy.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            Feature vector of each user.

        query: Sentence, shape (n_samples, dim_query)
            Query vector given by users.

        action: torch.Tensor, shape (n_samples, ) or (n_samples, dim_action)
            Discrete or continuous (soft) actions chosen by the given policy.

        """
        action_embedding = self.action_list[action]
        auxiliary_output = self.auxiliary_output_generator.sample_auxiliary_output(
            query=query,
            action_embedding=action_embedding,
        )
        expected_reward = self.reward_simulator.calc_expected_reward(
            context=context,
            query=query,
            auxiliary_output=auxiliary_output,
        )
        return self._sample_reward(expected_reward=expected_reward)

    def sample_reward_given_output(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        auxiliary_output: torch.Tensor,
    ):
        """Sample reward given action chosen by a policy.

        Parameters
        -------
        context: torch.Tensor, shape (n_samples, dim_context)
            Feature vector of each user.

        query: Sentence, shape (n_samples, dim_query)
            Query vector given by users.

        auxiliary_output: Sentence, shape (n_samples, dim_auxiliary_output)
                Auiliary output vector generated by query and action.

        """
        expected_reward = self.reward_simulator.calc_expected_reward(
            context=context,
            query=query,
            auxiliary_output=auxiliary_output,
        )
        return self._sample_reward(expected_reward=expected_reward)
