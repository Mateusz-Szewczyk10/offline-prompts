from dataclasses import dataclass
from typing import Union, Optional

import torch

from ..dataset.synthetic import SyntheticDataset
from ..policy.base import (
    BasePolicy,
    BaseActionPolicyModel,
    BaseActionRewardModel,
    BaseOutputRewardModel,
    BaseKernelMarginalDensityModel,
    BaseClusteringModel,
)
from ..types import LoggedDataset
from ..utils import check_logged_feedback, to_device

Policy = Union[BasePolicy, BaseActionPolicyModel]


@dataclass
class PolicyEvaluator:
    """Evaluator class for action policy."""

    env: Optional[SyntheticDataset] = None
    action_reward_predictor: Optional[BaseActionRewardModel] = None
    auxiliary_output_reward_oredictor: Optional[BaseOutputRewardModel] = None
    clustering_policy: Optional[BaseClusteringModel] = None
    action_list: Optional[torch.Tensor] = None
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.random_state is None:
            raise ValueError("random_state must be given")

    def online_policy_evaluation(
        self,
        eval_policy: Policy,
        n_samples: int = 100,
    ):
        logged_feedback = self.env.sample_dataset(
            policy=eval_policy,
            n_samples=n_samples,
        )
        policy_value = logged_feedback["reward"].mean()
        return policy_value

    def regression_based_policy_evaluation(
        self,
        logged_feedback: LoggedDataset,
        eval_policy: Policy,
        action_reward_predictor: Optional[BaseActionRewardModel] = None,
    ):
        if action_reward_predictor is None:
            action_reward_predictor = self.action_reward_predictor

        if action_reward_predictor is None:
            raise RuntimeError("action_reward_predictor must be given.")

        device = action_reward_predictor.device

        context = logged_feedback["context"].to(device)
        query = logged_feedback["query"].to(device)

        action = eval_policy.sample_action(
            context=context,
            query=query,
            return_cpu_tensor=False,
        )
        predicted_reward = action_reward_predictor.predict_value(
            context=context,
            query=query,
            action=action,
            return_cpu_tensor=False,
        )

        policy_value = predicted_reward.mean()
        return policy_value

    def importance_sampling_based_policy_evaluation(
        self,
        logged_feedback: LoggedDataset,
        eval_policy: Policy,
        logging_action_choice_prob: Optional[torch.Tensor] = None,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        clip_threshold: int = 200,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        context = logged_feedback["context"].to(device)
        context = logged_feedback["query"].to(device)
        action = logged_feedback["action"].to(device)
        reward = logged_feedback["reward"].to(device)
        query = query.to(device)

        logging_policy = logged_feedback["logging_policy"]
        if logging_action_choice_prob is None:
            logging_action_choice_prob = logging_policy.calc_action_choice_probability(
                context=context,
                query=query,
                predicted_reward=logging_predicted_reward,
            )

        n_samples = len(context)
        logging_pscore = logging_action_choice_prob[
            to_device(torch.arange(n_samples), device=device), action
        ]
        eval_pscore = eval_policy.calc_prob_given_action(
            context=context,
            query=query,
            action=action,
        )
        iw = eval_pscore / logging_pscore
        iw = torch.nan_to_num(iw)
        iw = torch.clip(iw, max=clip_threshold)

        policy_value = (iw * reward).mean()
        return policy_value

    def hybrid_policy_evaluation(
        self,
        logged_feedback: LoggedDataset,
        eval_policy: Policy,
        action_reward_predictor: Optional[BaseActionRewardModel] = None,
        logging_action_choice_prob: Optional[torch.Tensor] = None,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        clip_threshold: int = 200,
    ):
        if action_reward_predictor is None:
            action_reward_predictor = self.action_reward_predictor

        if action_reward_predictor is None:
            raise RuntimeError("action_reward_predictor must be given.")

        device = action_reward_predictor.device

        context = logged_feedback["context"].to(device)
        query = logged_feedback["query"].to(device)
        action = logged_feedback["action"].to(device)
        reward = logged_feedback["reward"].to(device)
        query = query.to(device)

        logging_policy = logged_feedback["logging_policy"]
        if logging_action_choice_prob is None:
            logging_action_choice_prob = logging_policy.calc_action_choice_probability(
                context=context,
                query=query,
                predicted_reward=logging_predicted_reward,
            )

        n_samples = len(context)
        logging_pscore = logging_action_choice_prob[
            to_device(torch.arange(n_samples), device=device), action
        ]
        eval_pscore = eval_policy.calc_prob_given_action(
            context=context,
            query=query,
            action=action,
        )
        iw = eval_pscore / logging_pscore
        iw = torch.nan_to_num(iw)
        iw = torch.clip(iw, max=clip_threshold)

        predicted_reward = action_reward_predictor.predict_value(
            context=context,
            query=query,
            action=action,
            return_cpu_tensor=False,
        )

        eval_action = eval_policy.sample_action(
            context=context,
            query=query,
            return_cpu_tensor=False,
        )
        eval_predicted_reward = action_reward_predictor.predict_value(
            context=context,
            query=query,
            action=eval_action,
            return_cpu_tensor=False,
        )

        policy_value = (iw * (reward - predicted_reward) + eval_predicted_reward).mean()
        return policy_value

    def offcem_policy_evaluation(
        self,
        logged_feedback: LoggedDataset,
        eval_policy: Policy,
        action_reward_predictor: Optional[BaseActionRewardModel] = None,
        logging_action_choice_prob: Optional[torch.Tensor] = None,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        clustering_policy: Optional[BaseClusteringModel] = None,
        clip_threshold: int = 200,
    ):
        if action_reward_predictor is None:
            action_reward_predictor = self.action_reward_predictor

        if action_reward_predictor is None:
            raise RuntimeError("action_reward_predictor must be given.")

        if clustering_policy is None:
            clustering_policy = self.clustering_policy

        if clustering_policy is None:
            raise RuntimeError("clustering_policy must be given.")

        device = action_reward_predictor.device

        context = logged_feedback["context"].to(device)
        query = logged_feedback["context"].to(device)
        action = logged_feedback["action"].to(device)
        auxiliary_output = logged_feedback["auxiliary_output"].to(device)
        reward = logged_feedback["reward"].to(device)

        logging_policy = logged_feedback["logging_policy"]
        if logging_action_choice_prob is None:
            logging_action_choice_prob = logging_policy.calc_action_choice_probability(
                context=context,
                query=query,
                predicted_reward=logging_predicted_reward,
            )
        eval_action_choice_prob = eval_policy.calc_action_choice_probability(
            context=context,
            query=query,
        )

        cluster = self.clustering_policy.retrieve_cluster(
            context=context,
            query=query,
            action=action,
            auxiliary_output=auxiliary_output,
            resample_clustering=True,
        )
        logging_cluster_prob = self.clustering_policy.calc_cluster_choice_prob(
            context=context,
            query=query,
            cluster=cluster,
            action_choice_prob=logging_action_choice_prob,
        )
        eval_cluster_prob = self.clustering_policy.calc_cluster_choice_prob(
            context=context,
            query=query,
            cluster=cluster,
            action_choice_prob=eval_action_choice_prob,
        )

        iw = eval_cluster_prob / logging_cluster_prob
        iw = torch.nan_to_num(iw)
        iw = torch.clip(iw, max=clip_threshold)

        predicted_reward = action_reward_predictor.predict_value(
            context=context,
            query=query,
            action=action,
            return_cpu_tensor=False,
        )

        eval_action = eval_policy.sample_action(
            context=context,
            query=query,
            return_cpu_tensor=False,
        )
        eval_predicted_reward = action_reward_predictor.predict_value(
            context=context,
            query=query,
            action=eval_action,
            return_cpu_tensor=False,
        )

        policy_value = (iw * (reward - predicted_reward) + eval_predicted_reward).mean()
        return policy_value

    def kernel_IS_based_policy_evaluation(
        self,
        logged_feedback: LoggedDataset,
        logging_kernel_density_model: BaseKernelMarginalDensityModel,
        eval_kernel_density_model: BaseKernelMarginalDensityModel,
        clip_threshold: int = 200,
    ):
        if logging_kernel_density_model is None:
            raise RuntimeError("logging_kernel_density_model must be given.")

        if eval_kernel_density_model is None:
            raise RuntimeError("eval_kernel_density_model must be given.")

        device = logging_kernel_density_model.device

        context = logged_feedback["context"].to(device)
        query = logged_feedback["query"].to(device)
        reward = logged_feedback["reward"].to(device)
        auxiliary_output = logged_feedback["auxiliary_output"].to(device)

        logging_marginal_density = (
            logging_kernel_density_model.estimate_marginal_density(
                context=context,
                query=query,
                auxiliary_output=auxiliary_output,
            )
        )
        eval_marginal_density = eval_kernel_density_model.estimate_marginal_density(
            context=context,
            query=query,
            auxiliary_output=auxiliary_output,
        )
        iw = eval_marginal_density / logging_marginal_density
        iw = torch.nan_to_num(iw)
        iw = torch.clip(iw, max=clip_threshold)

        policy_value = (iw * reward).mean()
        return policy_value
