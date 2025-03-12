"""Learner class for the first and second stage policies."""
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Union, Any, Dict
from pathlib import Path
from tqdm.auto import tqdm
import random

import torch
from torch.optim import Optimizer, Adam

from ..dataset.synthetic import SyntheticDataset
from ..dataset.function import AuxiliaryOutputGenerator
from ..policy.base import (
    BasePolicy,
    BaseActionPolicyModel,
    BaseClusterPolicyModel,
    BaseClusteringModel,
    BaseActionRewardModel,
    BaseOutputRewardModel,
    BaseKernelMarginalDensityModel,
)
from ..policy.policy import TwoStagePolicy
from ..types import LoggedDataset
from ..utils import check_logged_feedback, torch_seed


Policy = Union[BasePolicy, BaseActionPolicyModel, BaseClusterPolicyModel]


@dataclass
class PolicyLearner:
    model: Union[BaseActionPolicyModel, BaseClusterPolicyModel]
    action_list: torch.Tensor
    second_stage_policy: Optional[Policy] = None
    clustering_policy: Optional[BaseClusteringModel] = None
    action_reward_predictor: Optional[BaseActionRewardModel] = None
    optimizer: Optimizer = Adam
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    env: Optional[SyntheticDataset] = None
    random_state: Optional[int] = None

    """Learner class for action policy.

    Imported as: :class:`src.opl.PolicyLearner`

    Parameters
    -------
    model: BaseActionPolicyModel or BaseClusterPolicyModel
        Action or cluster policy (either first stage or single stage policy) to train.

    action_list: torch.Tensor
        Mapping from action id to its embeddings.

    second_stage_policy: Policy, default=None
        (Pre-trained) second stage policy. This must be given when training a first stage policy.

    clustering_policy: BaseClusteringModel, default=None
        (Pre-trained) clustering policy that determines the action clustering for each context. This must be given when training a first stage policy.

    action_reward_predictor: BaseActionRewardModel, default=None
        (Pre-trained) action reward predictor. This must be given when using a model-based or hybrid appraoch to train a policy.

    optimizer: torch.optim.Optimizer, default=None
        Class of optimizer (not an instance).

    optimizer_kwargs: dict, default=None
        Arguments of optimizer.

    env: SyntheticDataset, default=None
        Online environment for evaluation.

    random_state: int, default=None.
        Random state.
    
    """

    def __post_init__(self):
        self.trained_model = None
        self.n_actions = len(self.action_list)

        if self.second_stage_policy is not None:
            if self.clustering_policy is None:
                raise ValueError(
                    "clustering_policy must be given when using two-stage policy."
                )
            if self.clustering_policy.n_clusters != self.model.n_actions:
                raise RuntimeError(
                    "Expected clustering_policy.n_clusters == model.n_actions (number of output of the first stage policy), but found False."
                )
            if not (
                self.clustering_policy.n_actions
                == self.second_stage_policy.n_actions
                == self.n_actions
            ):
                raise RuntimeError(
                    "Expected clustering_policy.n_actions == second_stage_policy.n_actions == len(action_list), but found False."
                )

        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {
                "lr": 1e-4,
            }

        if self.random_state is None:
            raise ValueError("random_state must be given")

        self.device = self.model.device

    def load(self, path: Path, is_init: bool = False):
        """Load model."""
        if is_init:
            self.model.load_state_dict(torch.load(path))
            model = self.model
        else:
            self.trained_model.load_state_dict(torch.load(path))
            model = self.trained_model

        return model

    def save(self, path: Path):
        """Save Model."""
        torch.save(self.trained_model.state_dict(), path)

    def seed(self, random_state: int):
        """Fix seed."""
        random.seed(random_state)
        torch_seed(random_state)

    def online_policy_gradient(
        self,
        logging_action_choice_prob: Optional[torch.Tensor] = None,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        is_oracle_logging_policy: bool = False,
        n_epochs: int = 1000,
        n_steps_per_epoch: int = 10,
        n_epochs_per_log: int = 10,
        batch_size: int = 32,
        make_copy: bool = False,
        return_training_logs: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
    ):
        """Train policy in an online manner.

        Parameters
        -------
        logging_action_choice_prob: torch.Tensor, shape (n_samples, n_actions), default=None
            For API consistency.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            For API consistency.

        is_oracle_logging_policy: bool, default=False
            Whether the logging policy uses oracle expected reward.

        n_epochs: int, default=1000.
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=10.
            Number of gradient steps within an epoch.

        n_epochs_per_log: int, default=10.
            Number of epochs in the logging interval.

        batch_size: int, default=32.
            Batch size.

        make_copy: bool, default=False.
            Whether to create copy of the model before training.

        return_training_logs: bool, default=False.
            Whether to return (true) policy value at each training epoch.

        save_path: Path, default=None.
            Path to save the model.

        random_state: int, default=None.
            Random state.

        """
        if self.env is None:
            raise RuntimeError(
                "self.env must be given when training a policy online. Please initialize the class with env."
            )

        if random_state is None:
            random_state = self.random_state
        self.seed(random_state)

        if make_copy:
            model = deepcopy(self.model)
        else:
            model = self.model

        optimizer = self.optimizer(model.parameters(), **self.optimizer_kwargs)

        is_two_stage_policy = self.second_stage_policy is not None

        train_losses = torch.zeros((n_epochs + 1,))
        policy_values = torch.zeros((n_epochs // n_epochs_per_log + 1,))

        if is_two_stage_policy:
            eval_policy = TwoStagePolicy(
                first_stage_policy=model,
                second_stage_policy=self.second_stage_policy,
                clustering_policy=self.clustering_policy,
                device=self.device,
            )
        else:
            eval_policy = model

        if self.env is not None:
            policy_values[0] = self.env.calc_expected_policy_value(
                eval_policy,
                is_oracle_clustering_logging_policy=is_oracle_logging_policy,
            )
            policy_value = policy_values[0].item()
        else:
            policy_value = torch.nan

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train policy online: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "loss": f"{train_losses[i]:.4g}",
                        "policy_value": f"{policy_value:.4g}",
                    }
                )

                for j in range(n_steps_per_epoch):
                    context_and_query_ = (
                        self.env.context_query_generator.sample_context_and_query(
                            n_samples=batch_size,
                        )
                    )
                    context_ = context_and_query_[:, : self.env.dim_context]
                    query_ = context_and_query_[:, self.env.dim_context :]

                    if is_two_stage_policy:
                        cluster_ = model.sample_action(
                            context=context_,
                            query=query_,
                        )
                        log_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=cluster_,
                            is_log_prob=True,
                            calc_gradient=True,
                        )
                        candidate_actions = (
                            self.clustering_policy.retrieve_cluster_center(
                                context=context_,
                                query=query_,
                                cluster=cluster_,
                            )
                        )
                        action_ = self.second_stage_policy.sample_action(
                            context=context_,
                            query=query_,
                            candidate_actions=candidate_actions,
                        )
                    else:
                        action_ = model.sample_action(
                            context=context_,
                            query=query_,
                        )
                        log_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=action_,
                            is_log_prob=True,
                            calc_gradient=True,
                        )

                    reward_ = self.env.sample_reward_given_action(
                        context=context_,
                        query=query_,
                        action=action_,
                    )

                    loss_ = -(log_prob_ * reward_).sum()

                    optimizer.zero_grad()
                    loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += loss_.item() / n_steps_per_epoch

                if (i + 1) % n_epochs_per_log == 0:
                    if is_two_stage_policy:
                        eval_policy.first_stage_policy = model

                    if self.env is not None:
                        policy_values[
                            (i + 1) // n_epochs_per_log
                        ] = self.env.calc_expected_policy_value(
                            eval_policy,
                            is_oracle_clustering_logging_policy=is_oracle_logging_policy,
                        )
                        policy_value = policy_values[(i + 1) // n_epochs_per_log].item()
                        # print(policy_value)

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        if return_training_logs:
            output = (model, policy_values)
        else:
            output = model

        return output

    def model_based_policy_gradient(
        self,
        logged_feedback: LoggedDataset,
        logging_action_choice_prob: Optional[torch.Tensor] = None,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        is_oracle_logging_policy: bool = False,
        n_epochs: int = 1000,
        n_steps_per_epoch: int = 10,
        n_epochs_per_log: int = 10,
        batch_size: int = 32,
        make_copy: bool = False,
        return_training_logs: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
    ):
        """Train policy from logged data via the model-based approach.

        Parameters
        -------
        logged_feedback: LoggedDataset
            Logged data, which contains the following keys.

            .. code-block:: python

                    key: [
                        context,          # (n_samples, )
                        query,            # (n_samples, )
                        action,           #  -
                        output,           #  -
                        reward,           #  -
                        logging_policy,   #  -
                    ]

            See dataset/synthetic.py for the details of each key.

        logging_action_choice_prob: torch.Tensor, shape (n_samples, n_actions), default=None
            For API consistency.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions used for the logging policy in the clustering policy.

        is_oracle_logging_policy: bool, default=False
            Whether the logging policy uses oracle expected reward.

        n_epochs: int, default=1000.
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=10.
            Number of gradient steps within an epoch.

        n_epochs_per_log: int, default=10.
            Number of epochs in the logging interval.

        batch_size: int, default=32.
            Batch size.

        make_copy: bool, default=False.
            Whether to create copy of the model before training.

        return_training_logs: bool, default=False.
            Whether to return (true) policy value at each training epoch.

        save_path: Path, default=None.
            Path to save the model.

        random_state: int, default=None.
            Random state.

        """
        if self.action_reward_predictor is None:
            raise RuntimeError(
                "action_reward_predictor is not provided. Please initialize the class with action_reward_predictor."
            )
        if return_training_logs and self.env is None:
            raise RuntimeError(
                "return_training_logs option is feasible when self.env is given. Please initialize the class with env."
            )

        # check_logged_feedback(logged_feedback)

        if random_state is None:
            random_state = self.random_state
        self.seed(random_state)

        if make_copy:
            model = deepcopy(self.model)
        else:
            model = self.model

        optimizer = self.optimizer(model.parameters(), **self.optimizer_kwargs)

        is_two_stage_policy = self.second_stage_policy is not None

        context = logged_feedback["context"]
        query = logged_feedback["query"]

        train_size = len(context)
        train_losses = torch.zeros((n_epochs + 1,))
        policy_values = torch.zeros((n_epochs // n_epochs_per_log + 1,))

        if is_two_stage_policy:
            eval_policy = TwoStagePolicy(
                first_stage_policy=model,
                second_stage_policy=self.second_stage_policy,
                clustering_policy=self.clustering_policy,
                device=self.device,
            )
        else:
            eval_policy = model

        if self.env is not None:
            policy_values[0] = self.env.calc_expected_policy_value(
                eval_policy,
                is_oracle_clustering_logging_policy=is_oracle_logging_policy,
            )
            policy_value = policy_values[0].item()
        else:
            policy_value = torch.nan

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train policy: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "loss": f"{train_losses[i]:.4g}",
                        "policy_value": f"{policy_value:.4g}",
                    }
                )

                for j in range(n_steps_per_epoch):
                    idx_ = torch.multinomial(
                        torch.ones(train_size), num_samples=batch_size, replacement=True
                    )
                    context_ = context[idx_]
                    query_ = query[idx_]

                    if logging_predicted_reward is not None:
                        logging_predicted_reward_ = logging_predicted_reward[idx_]
                    else:
                        logging_predicted_reward_ = None

                    if is_two_stage_policy:
                        cluster_centers_ = (
                            self.clustering_policy.retrieve_cluster_centers(
                                context=context_,
                                query=query_,
                                logging_predicted_reward=logging_predicted_reward_,
                                resample_clustering=True,
                            )
                        )
                        cluster_ = model.sample_action(
                            context=context_,
                            query=query_,
                            cluster_centers=cluster_centers_,
                        )
                        log_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            cluster_centers=cluster_centers_,
                            action=cluster_,
                            is_log_prob=True,
                            calc_gradient=True,
                        )
                        candidate_actions_ = (
                            self.clustering_policy.retrieve_candidate_actions(
                                context=context_,
                                query=query_,
                                cluster=cluster_,
                                logging_predicted_reward=logging_predicted_reward_,
                            )
                        )
                        predicted_reward_ = (
                            self.second_stage_policy.predict_policy_value(
                                context=context_,
                                query=query_,
                                candidate_actions=candidate_actions_,
                                return_per_sample=True,
                            )
                        )

                    else:
                        action_ = model.sample_action(
                            context=context_,
                            query=query_,
                        )
                        log_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=action_,
                            is_log_prob=True,
                            calc_gradient=True,
                        )
                        predicted_reward_ = self.action_reward_predictor.predict_value(
                            context=context_,
                            query=query_,
                            action=action_,
                        )

                    train_loss_ = -(predicted_reward_ * log_prob_).mean()

                    optimizer.zero_grad()
                    train_loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += train_loss_.item() / n_steps_per_epoch

                if (i + 1) % n_epochs_per_log == 0:
                    if is_two_stage_policy:
                        eval_policy.first_stage_policy = model

                    if self.env is not None:
                        policy_values[
                            (i + 1) // n_epochs_per_log
                        ] = self.env.calc_expected_policy_value(
                            eval_policy,
                            is_oracle_clustering_logging_policy=is_oracle_logging_policy,
                        )
                        policy_value = policy_values[(i + 1) // n_epochs_per_log].item()
                        # print(policy_value)

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        if return_training_logs:
            output = (model, policy_values)
        else:
            output = model

        return output

    def importance_sampling_based_policy_gradient(
        self,
        logged_feedback: LoggedDataset,
        logging_action_choice_prob: Optional[torch.Tensor] = None,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        is_oracle_logging_policy: bool = False,
        clip_threshold: Union[int, float] = 200,
        n_epochs: int = 1000,
        n_steps_per_epoch: int = 10,
        n_epochs_per_log: int = 10,
        batch_size: int = 32,
        make_copy: bool = False,
        return_training_logs: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
    ):
        """Train policy from logged data via the importance sampling-based approach.

        Parameters
        -------
        logged_feedback: LoggedDataset
            Logged data, which contains the following keys.

            .. code-block:: python

                    key: [
                        context,          # (n_samples, )
                        query,            # (n_samples, )
                        action,           # (n_samples, )
                        output,           #  -
                        reward,           # (n_samples, )
                        logging_policy,   #  -
                    ]

            See dataset/synthetic.py for the details of each key.

        logging_action_choice_prob: torch.Tensor, shape (n_samples, n_actions), default=None
            Action choice probability of logging policy to all candidate actions.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions used for the logging policy in the clustering policy.

        is_oracle_logging_policy: bool, default=False
            Whether the logging policy uses oracle expected reward.

        clip_threshold: float or int, default=200
            Threshold for clipping the importance weight.

        n_epochs: int, default=1000.
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=10.
            Number of gradient steps within an epoch.

        n_epochs_per_log: int, default=10.
            Number of epochs in the logging interval.

        batch_size: int, default=32.
            Batch size.

        make_copy: bool, default=False.
            Whether to create copy of the model before training.

        return_training_logs: bool, default=False.
            Whether to return (true) policy value at each training epoch.

        save_path: Path, default=None.
            Path to save the model.

        random_state: int, default=None.
            Random state.

        """
        if return_training_logs and self.env is None:
            raise RuntimeError(
                "return_training_logs option is feasible when self.env is given. Please initialize the class with env."
            )

        # check_logged_feedback(logged_feedback)

        if random_state is None:
            random_state = self.random_state
        self.seed(random_state)

        if make_copy:
            model = deepcopy(self.model)
        else:
            model = self.model

        optimizer = self.optimizer(model.parameters(), **self.optimizer_kwargs)

        is_two_stage_policy = self.second_stage_policy is not None

        context = logged_feedback["context"]
        query = logged_feedback["query"]
        action = logged_feedback["action"]
        auxiliary_output = logged_feedback["auxiliary_output"]
        reward = logged_feedback["reward"]
        logging_policy = logged_feedback["logging_policy"]

        if logging_action_choice_prob is None:
            logging_action_choice_prob = logging_policy.calc_action_choice_probability(
                context=context,
                query=query,
                predicted_reward=logging_predicted_reward,
            )

        train_size = len(context)
        train_losses = torch.zeros((n_epochs + 1,))
        policy_values = torch.zeros((n_epochs // n_epochs_per_log + 1,))

        if is_two_stage_policy:
            eval_policy = TwoStagePolicy(
                first_stage_policy=model,
                second_stage_policy=self.second_stage_policy,
                clustering_policy=self.clustering_policy,
                device=self.device,
            )
        else:
            eval_policy = model

        if self.env is not None:
            policy_values[0] = self.env.calc_expected_policy_value(
                eval_policy,
                is_oracle_clustering_logging_policy=is_oracle_logging_policy,
            )
            policy_value = policy_values[0].item()
        else:
            policy_value = torch.nan

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train policy: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "loss": f"{train_losses[i]:.4g}",
                        "policy_value": f"{policy_value:.4g}",
                    }
                )

                train_loss_logs = torch.zeros((n_steps_per_epoch,))

                for j in range(n_steps_per_epoch):
                    idx_ = torch.multinomial(
                        torch.ones(train_size), num_samples=batch_size, replacement=True
                    )
                    context_ = context[idx_]
                    query_ = query[idx_]
                    action_ = action[idx_]
                    auxiliary_output_ = auxiliary_output[idx_]
                    reward_ = reward[idx_]
                    logging_action_prob_ = logging_action_choice_prob[idx_]
                    logging_pscore_ = logging_action_choice_prob[idx_, action_]

                    if logging_predicted_reward is not None:
                        logging_predicted_reward_ = logging_predicted_reward[idx_]
                    else:
                        logging_predicted_reward_ = None

                    if is_two_stage_policy:
                        cluster_centers_ = (
                            self.clustering_policy.retrieve_cluster_centers(
                                context=context_,
                                query=query_,
                                logging_predicted_reward=logging_predicted_reward_,
                                resample_clustering=True,
                            )
                        )
                        cluster_ = self.clustering_policy.retrieve_cluster(
                            context=context_,
                            query=query_,
                            action=action_,
                            auxiliary_output=auxiliary_output_,
                            logging_predicted_reward=logging_predicted_reward_,
                        )
                        log_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=cluster_,
                            cluster_centers=cluster_centers_,
                            is_log_prob=True,
                            calc_gradient=True,
                        )
                        evaluation_cluster_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=cluster_,
                            cluster_centers=cluster_centers_,
                        )
                        logging_cluster_prob_ = (
                            self.clustering_policy.calc_cluster_choice_prob(
                                context=context_,
                                query=query_,
                                action_choice_prob=logging_action_prob_,
                                cluster=cluster_,
                                logging_predicted_reward=logging_predicted_reward_,
                            )
                        )
                        iw_ = evaluation_cluster_prob_ / logging_cluster_prob_
                        iw_ = torch.nan_to_num(iw_)  # avoid zero division

                    else:
                        log_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=action_,
                            is_log_prob=True,
                            calc_gradient=True,
                        )
                        evaluation_pscore_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=action_,
                        )

                        iw_ = evaluation_pscore_ / logging_pscore_

                    iw_ = torch.clip(iw_, max=clip_threshold)
                    train_loss_ = -(iw_ * log_prob_ * reward_).mean()

                    optimizer.zero_grad()
                    train_loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += train_loss_.item() / n_steps_per_epoch
                    train_loss_logs[j] = train_loss_.item()

                if (i + 1) % n_epochs_per_log == 0:
                    if is_two_stage_policy:
                        eval_policy.first_stage_policy = model

                    if self.env is not None:
                        policy_values[
                            (i + 1) // n_epochs_per_log
                        ] = self.env.calc_expected_policy_value(
                            eval_policy,
                            is_oracle_clustering_logging_policy=is_oracle_logging_policy,
                        )
                        policy_value = policy_values[(i + 1) // n_epochs_per_log].item()
                        # print(policy_value)

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        if return_training_logs:
            output = (model, policy_values)
        else:
            output = model

        return output

    def hybrid_policy_gradient(
        self,
        logged_feedback: LoggedDataset,
        logging_action_choice_prob: Optional[torch.Tensor] = None,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        is_oracle_logging_policy: bool = False,
        clip_threshold: Union[float, int] = 200,
        n_epochs: int = 1000,
        n_steps_per_epoch: int = 10,
        n_epochs_per_log: int = 10,
        batch_size: int = 32,
        make_copy: bool = False,
        return_training_logs: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
    ):
        """Train policy from logged data via the hybrid approach.

        Parameters
        -------
        logged_feedback: LoggedDataset
            Logged data, which contains the following keys.

            .. code-block:: python

                    key: [
                        context,          # (n_samples, )
                        query,            # (n_samples, )
                        action,           # (n_samples, )
                        output,           # (n_samples, )
                        reward,           # (n_samples, )
                        logging_policy,   #  -
                    ]

            See dataset/synthetic.py for the details of each key.

        logging_action_choice_prob: torch.Tensor, shape (n_samples, n_actions), default=None
            Action choice probability of logging policy to all candidate actions.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions used for the logging policy in the clustering policy.

        is_oracle_logging_policy: bool, default=False
            Whether the logging policy uses oracle expected reward.

        clip_threshold: float or int, default=200
            Threshold for clipping the importance weight.

        n_epochs: int, default=1000.
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=10.
            Number of gradient steps within an epoch.

        n_epochs_per_log: int, default=10.
            Number of epochs in the logging interval.

        batch_size: int, default=32.
            Batch size.

        make_copy: bool, default=False.
            Whether to create copy of the model before training.

        return_training_logs: bool, default=False.
            Whether to return (true) policy value at each training epoch.

        save_path: Path, default=None.
            Path to save the model.

        random_state: int, default=None.
            Random state.

        """
        if self.action_reward_predictor is None:
            raise RuntimeError(
                "action_reward_predictor is not provided. Please initialize the class with action_reward_predictor."
            )
        if return_training_logs and self.env is None:
            raise RuntimeError(
                "return_training_logs option is feasible when self.env is given. Please initialize the class with env."
            )

        # check_logged_feedback(logged_feedback)

        if random_state is None:
            random_state = self.random_state
        self.seed(random_state)

        if make_copy:
            model = deepcopy(self.model)
        else:
            model = self.model

        optimizer = self.optimizer(model.parameters(), **self.optimizer_kwargs)

        is_two_stage_policy = self.second_stage_policy is not None

        context = logged_feedback["context"]
        query = logged_feedback["query"]
        action = logged_feedback["action"]
        reward = logged_feedback["reward"]
        auxiliary_output = logged_feedback["auxiliary_output"]
        logging_policy = logged_feedback["logging_policy"]

        if logging_action_choice_prob is None:
            logging_action_choice_prob = logging_policy.calc_action_choice_probability(
                context=context,
                query=query,
                predicted_reward=logging_predicted_reward,
            )

        train_size = len(context)
        train_losses = torch.zeros((n_epochs + 1,))
        policy_values = torch.zeros((n_epochs // n_epochs_per_log + 1,))

        if is_two_stage_policy:
            eval_policy = TwoStagePolicy(
                first_stage_policy=model,
                second_stage_policy=self.second_stage_policy,
                clustering_policy=self.clustering_policy,
                device=self.device,
            )
        else:
            eval_policy = model

        if self.env is not None:
            policy_values[0] = self.env.calc_expected_policy_value(
                eval_policy,
                is_oracle_clustering_logging_policy=is_oracle_logging_policy,
            )
            policy_value = policy_values[0].item()
        else:
            policy_value = torch.nan

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train policy: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "loss": f"{train_losses[i]:.4g}",
                        "policy_value": f"{policy_value:.4g}",
                    }
                )

                train_loss_logs = torch.zeros((n_steps_per_epoch,))

                for j in range(n_steps_per_epoch):
                    idx_ = torch.multinomial(
                        torch.ones(train_size), num_samples=batch_size, replacement=True
                    )
                    context_ = context[idx_]
                    query_ = query[idx_]
                    action_ = action[idx_]
                    auxiliary_output_ = auxiliary_output[idx_]
                    reward_ = reward[idx_]
                    logging_action_prob_ = logging_action_choice_prob[idx_]
                    logging_pscore_ = logging_action_choice_prob[idx_, action_]

                    if logging_predicted_reward is not None:
                        logging_predicted_reward_ = logging_predicted_reward[idx_]
                    else:
                        logging_predicted_reward_ = None

                    # importance sampling part
                    if is_two_stage_policy:
                        cluster_centers_ = (
                            self.clustering_policy.retrieve_cluster_centers(
                                context=context_,
                                query=query_,
                                logging_predicted_reward=logging_predicted_reward_,
                                resample_clustering=True,
                            )
                        )
                        cluster_ = self.clustering_policy.retrieve_cluster(
                            context=context_,
                            query=query_,
                            action=action_,
                            auxiliary_output=auxiliary_output_,
                            logging_predicted_reward=logging_predicted_reward_,
                        )
                        log_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=cluster_,
                            cluster_centers=cluster_centers_,
                            is_log_prob=True,
                            calc_gradient=True,
                        )
                        evaluation_cluster_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=cluster_,
                            cluster_centers=cluster_centers_,
                        )
                        logging_cluster_prob_ = (
                            self.clustering_policy.calc_cluster_choice_prob(
                                context=context_,
                                query=query_,
                                action_choice_prob=logging_action_prob_,
                                cluster=cluster_,
                                logging_predicted_reward=logging_predicted_reward_,
                            )
                        )
                        iw_ = evaluation_cluster_prob_ / logging_cluster_prob_
                        iw_ = torch.nan_to_num(iw_)  # avoid zero division

                    else:
                        log_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=action_,
                            is_log_prob=True,
                            calc_gradient=True,
                        )
                        evaluation_pscore_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=action_,
                        )
                        iw_ = evaluation_pscore_ / logging_pscore_

                    baseline_ = self.action_reward_predictor.predict_value(
                        context=context_,
                        query=query_,
                        action=action_,
                    )

                    iw_ = torch.clip(iw_, max=clip_threshold)
                    iw_based_loss_ = -(iw_ * log_prob_ * (reward_ - baseline_)).mean()

                    # model-based part
                    if is_two_stage_policy:
                        cluster_ = model.sample_action(
                            context=context_,
                            query=query_,
                            cluster_centers=cluster_centers_,
                        )
                        eval_log_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            cluster_centers=cluster_centers_,
                            action=cluster_,
                            is_log_prob=True,
                            calc_gradient=True,
                        )
                        candidate_actions_ = (
                            self.clustering_policy.retrieve_candidate_actions(
                                context=context_,
                                query=query_,
                                cluster=cluster_,
                                logging_predicted_reward=logging_predicted_reward_,
                            )
                        )
                        eval_predicted_reward_ = (
                            self.second_stage_policy.predict_policy_value(
                                context=context_,
                                query=query_,
                                candidate_actions=candidate_actions_,
                                return_per_sample=True,
                            )
                        )

                    else:
                        eval_action_ = model.sample_action(
                            context=context_,
                            query=query_,
                        )
                        eval_log_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=eval_action_,
                            is_log_prob=True,
                            calc_gradient=True,
                        )
                        eval_predicted_reward_ = (
                            self.action_reward_predictor.predict_value(
                                context=context_,
                                query=query_,
                                action=eval_action_,
                            )
                        )

                    model_based_loss_ = -(
                        eval_predicted_reward_ * eval_log_prob_
                    ).mean()

                    train_loss_ = iw_based_loss_ + model_based_loss_

                    optimizer.zero_grad()
                    train_loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += train_loss_.item() / n_steps_per_epoch
                    train_loss_logs[j] = train_loss_.item()

                if (i + 1) % n_epochs_per_log == 0:
                    if is_two_stage_policy:
                        eval_policy.first_stage_policy = model

                    if self.env is not None:
                        policy_values[
                            (i + 1) // n_epochs_per_log
                        ] = self.env.calc_expected_policy_value(
                            eval_policy,
                            is_oracle_clustering_logging_policy=is_oracle_logging_policy,
                        )
                        policy_value = policy_values[(i + 1) // n_epochs_per_log].item()
                        # print(policy_value)

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        if return_training_logs:
            output = (model, policy_values)
        else:
            output = model

        return output


@dataclass
class KernelPolicyLearner:
    model: BaseActionPolicyModel
    kernel_marginal_estimator: BaseKernelMarginalDensityModel
    action_list: torch.Tensor
    action_reward_predictor: Optional[BaseActionRewardModel] = None  #
    output_reward_predictor: Optional[BaseOutputRewardModel] = None
    optimizer: Optimizer = Adam
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    env: Optional[SyntheticDataset] = None
    auxiliary_output_generator: Optional[AuxiliaryOutputGenerator] = None
    random_state: Optional[int] = None

    """Learner class for action policy.

    Imported as: :class:`src.opl.PolicyLearner`

    Parameters
    -------
    model: BaseActionPolicyModel
        Single stage policy to train.

    kernel_marginal_estimator: BaseKernelMarginalDensityModel
        Kernel function.

    action_list: torch.Tensor
        Mapping from action id to its embeddings.

    output_reward_oredictor: BaseOutputRewardModel, default=None
        (Pre-trained) output reward predictor. This must be given when using a model-based or hybrid appraoch to train a policy.

    optimizer: torch.optim.Optimizer, default=None
        Class of optimizer (not an instance).

    optimizer_kwargs: dict, default=None
        Arguments of optimizer.

    env: SyntheticDataset, default=None
        Online environment for evaluation.

    auxiliary_output_generator: AuxiliaryOutputGenerator, default=None
        Auxiliary output generator.

    random_state: int, default=None.
        Random state.
    
    """

    def __post_init__(self):
        self.trained_model = None
        self.trained_action_predictor = None
        self.trained_sample_predictor = None
        self.n_actions = len(self.action_list)

        if self.env is not None:
            if self.action_list is None:
                self.action_list = self.env.action_list
            if self.auxiliary_output_generator is None:
                self.auxiliary_output_generator = self.env.auxiliary_output_generator

        if self.random_state is None:
            raise ValueError("random_state must be given")

        self.device = self.model.device

    def load(self, path: Path, is_init: bool = False):
        """Load model."""
        if is_init:
            self.model.load_state_dict(torch.load(path))
            model = self.model
        else:
            self.trained_model.load_state_dict(torch.load(path))
            model = self.trained_model

        return model

    def save(self, path):
        """Save model."""
        torch.save(self.trained_model.state_dict(), path)

    def seed(self, random_state: int):
        """Fix seed."""
        random.seed(random_state)
        torch_seed(random_state)

    def importance_sampling_based_policy_gradient(
        self,
        logged_feedback: LoggedDataset,
        clip_threshold: Union[float, int] = 200,
        n_epochs: int = 1000,
        n_steps_per_epoch: int = 10,
        n_epochs_per_log: int = 10,
        batch_size: int = 32,
        use_monte_carlo: bool = False,
        n_samples_to_approximate: int = 100,
        make_copy: bool = False,
        return_training_logs: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
    ):
        """Train policy from logged data via the importance sampling-based approach.

        Parameters
        -------
        logged_feedback: LoggedDataset
            Logged data, which contains the following keys.

            .. code-block:: python

                    key: [
                        context,          # (n_samples, )
                        query,            # (n_samples, )
                        action,           # (n_samples, )
                        output,           #  -
                        reward,           # (n_samples, )
                        logging_policy,   #  -
                    ]

            See dataset/synthetic.py for the details of each key.

        clip_threshold: float or int, default=200
            Threshold for clipping the importance weight.

        n_epochs: int, default=1000.
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=10.
            Number of gradient steps within an epoch for training policy.

        n_epochs_per_log: int, default=10.
            Number of epochs in the logging interval.

        batch_size: int, default=32.
            Batch size.

        use_monte_carlo: bool, default=False
            Whether to use monte carlo sampling (or function approximation) to estimate the marginal density.

        n_samples_to_approximate: int, default=100.
            Number of samples to approximate the marginal density.

        make_copy: bool, default=False.
            Whether to create copy of the model before training.

        return_training_logs: bool, default=False.
            Whether to return (true) policy value at each training epoch.

        save_path: Path, default=None.
            Path to save the model.

        random_state: int, default=None.
            Random state.

        """
        if return_training_logs and self.env is None:
            raise RuntimeError(
                "return_training_logs option is feasible when self.env is given. Please initialize the class with env."
            )

        # check_logged_feedback(logged_feedback)

        if random_state is None:
            random_state = self.random_state
        self.seed(random_state)

        if make_copy:
            model = deepcopy(self.model)
        else:
            model = self.model

        context = logged_feedback["context"]
        query = logged_feedback["query"]
        auxiliary_output = logged_feedback["auxiliary_output"]
        reward = logged_feedback["reward"]
        logging_policy = logged_feedback["logging_policy"]
        train_size = len(context)

        logging_marginal_density = (
            self.kernel_marginal_estimator.estimate_marginal_density(
                context=context,
                query=query,
                auxiliary_output=auxiliary_output,
                policy=logging_policy,
                n_samples_to_approximate=n_samples_to_approximate,
                use_monte_carlo=use_monte_carlo,
            )
        )

        optimizer = self.optimizer(model.parameters(), **self.optimizer_kwargs)

        train_losses = torch.zeros((n_epochs + 1,))
        policy_values = torch.zeros((n_epochs // n_epochs_per_log + 1,))

        if self.env is not None:
            policy_values[0] = self.env.calc_expected_policy_value(model)
            policy_value = policy_values[0].item()
        else:
            policy_value = torch.nan

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train policy: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "loss": f"{train_losses[i]:.4g}",
                        "policy_value": f"{policy_value:.4g}",
                    }
                )

                train_loss_logs = torch.zeros((n_steps_per_epoch,))

                for j in range(n_steps_per_epoch):
                    idx_ = torch.multinomial(
                        torch.ones(train_size), num_samples=batch_size, replacement=True
                    )
                    context_ = context[idx_]
                    query_ = query[idx_]
                    auxiliary_output_ = auxiliary_output[idx_]
                    logging_marginal_density_ = logging_marginal_density[idx_]
                    reward_ = reward[idx_]

                    sampled_action_ = model.sample_action(
                        context=context_,
                        query=query_,
                    )
                    sampled_output_ = (
                        self.auxiliary_output_generator.sample_auxiliary_output(
                            query=query_,
                            action_embedding=self.action_list[sampled_action_],
                        )
                    )

                    pairwise_weight_ = (
                        self.kernel_marginal_estimator.calc_pairwise_weight(
                            pivot_output=auxiliary_output_,
                            sampled_outputs=sampled_output_,
                        )
                    )
                    log_prob_ = model.calc_prob_given_action(
                        context=context_,
                        query=query_,
                        action=sampled_action_,
                        is_log_prob=True,
                        calc_gradient=True,
                    )

                    marginal_iw_ = pairwise_weight_ / logging_marginal_density_
                    marginal_iw_ = torch.clip(marginal_iw_, max=clip_threshold)
                    policy_loss_ = -(marginal_iw_ * log_prob_ * reward_).sum()

                    optimizer.zero_grad()
                    policy_loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += policy_loss_.item() / n_steps_per_epoch
                    train_loss_logs[j] = policy_loss_.item()

                if (i + 1) % n_epochs_per_log == 0:
                    if self.env is not None:
                        policy_values[
                            (i + 1) // n_epochs_per_log
                        ] = self.env.calc_expected_policy_value(model)
                        policy_value = policy_values[(i + 1) // n_epochs_per_log].item()
                        # print(policy_value)

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        if return_training_logs:
            output = (model, policy_values)
        else:
            output = model

        return output

    def hybrid_policy_gradient(
        self,
        logged_feedback: LoggedDataset,
        clip_threshold: Union[float, int] = 200,
        n_epochs: int = 1000,
        n_steps_per_epoch: int = 10,
        n_epochs_per_log: int = 10,
        batch_size: int = 32,
        use_monte_carlo: bool = False,
        n_samples_to_approximate: int = 100,
        make_copy: bool = False,
        return_training_logs: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
    ):
        """Train policy from logged data via the hybrid approach.

        Parameters
        -------
        logged_feedback: LoggedDataset
            Logged data, which contains the following keys.

            .. code-block:: python

                    key: [
                        context,          # (n_samples, )
                        query,            # (n_samples, )
                        action,           # (n_samples, )
                        output,           # (n_samples, )
                        reward,           # (n_samples, )
                        logging_policy,   #  -
                    ]

            See dataset/synthetic.py for the details of each key.

        clip_threshold: float or int, default=200
            Threshold for clipping the importance weight.

        n_epochs: int, default=1000.
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=10.
            Number of gradient steps within an epoch for training policy.

        n_epochs_per_log: int, default=10.
            Number of epochs in the logging interval.

        batch_size: int, default=32.
            Batch size.

        use_monte_carlo: bool, default=False
            Whether to use monte carlo sampling (or function approximation) to estimate the marginal density.

        n_samples_to_approximate: int, default=100.
            Number of samples to approximate the marginal density.

        make_copy: bool, default=False.
            Whether to create copy of the model before training.

        return_training_logs: bool, default=False.
            Whether to return (true) policy value at each training epoch.

        save_path: Path, default=None.
            Path to save the model.

        random_state: int, default=None.
            Random state.

        """
        if self.output_reward_predictor is None:
            raise RuntimeError(
                "output_reward_predictor is not provided. Please initialize the class with output_reward_predictor."
            )
        if return_training_logs and self.env is None:
            raise RuntimeError(
                "return_training_logs option is feasible when self.env is given. Please initialize the class with env."
            )

        # check_logged_feedback(logged_feedback)

        if random_state is None:
            random_state = self.random_state
        self.seed(random_state)

        if make_copy:
            model = deepcopy(self.model)
        else:
            model = self.model

        context = logged_feedback["context"]
        query = logged_feedback["query"]
        auxiliary_output = logged_feedback["auxiliary_output"]
        reward = logged_feedback["reward"]
        logging_policy = logged_feedback["logging_policy"]
        action = logged_feedback["action"]
        train_size = len(context)

        logging_marginal_density = (
            self.kernel_marginal_estimator.estimate_marginal_density(
                context=context,
                query=query,
                auxiliary_output=auxiliary_output,
                policy=logging_policy,
                n_samples_to_approximate=n_samples_to_approximate,
                use_monte_carlo=use_monte_carlo,
            )
        )

        optimizer = self.optimizer(model.parameters(), **self.optimizer_kwargs)

        train_losses = torch.zeros((n_epochs + 1,))
        policy_values = torch.zeros((n_epochs // n_epochs_per_log + 1,))

        if self.env is not None:
            policy_values[0] = self.env.calc_expected_policy_value(model)
            policy_value = policy_values[0].item()
        else:
            policy_value = torch.nan

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train policy: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "loss": f"{train_losses[i]:.4g}",
                        "policy_value": f"{policy_value:.4g}",
                    }
                )

                train_loss_logs = torch.zeros((n_steps_per_epoch,))

                for j in range(n_steps_per_epoch):
                    idx_ = torch.multinomial(
                        torch.ones(train_size), num_samples=batch_size, replacement=True
                    )
                    context_ = context[idx_]
                    query_ = query[idx_]
                    auxiliary_output_ = auxiliary_output[idx_]
                    logging_marginal_density_ = logging_marginal_density[idx_]
                    reward_ = reward[idx_]
                    action_ = action[idx_]

                    predicted_reward_ = self.output_reward_predictor.predict_value(
                        context=context_,
                        query=query_,
                        auxiliary_output=auxiliary_output_,
                    )
                    # predicted_reward_ = self.action_reward_predictor.predict_value(
                    #     context=context_,
                    #     query=query_,
                    #     action=action_,
                    # )

                    sampled_action_ = model.sample_action(
                        context=context_,
                        query=query_,
                    )
                    sampled_output_ = (
                        self.auxiliary_output_generator.sample_auxiliary_output(
                            query=query_,
                            action_embedding=self.action_list[sampled_action_],
                        )
                    )

                    pairwise_weight_ = (
                        self.kernel_marginal_estimator.calc_pairwise_weight(
                            pivot_output=auxiliary_output_,
                            sampled_outputs=sampled_output_,
                        )
                    )
                    log_prob_ = model.calc_prob_given_action(
                        context=context_,
                        query=query_,
                        action=sampled_action_,
                        is_log_prob=True,
                        calc_gradient=True,
                    )

                    eval_predicted_reward_ = self.output_reward_predictor.predict_value(
                        context=context_,
                        query=query_,
                        auxiliary_output=sampled_output_,
                    )
                    # eval_predicted_reward_ = self.action_reward_predictor.predict_value(
                    #     context=context_,
                    #     query=query_,
                    #     action=sampled_action_,
                    # )

                    marginal_iw_ = pairwise_weight_ / logging_marginal_density_
                    marginal_iw_ = torch.clip(marginal_iw_, max=clip_threshold)

                    # debugging
                    # predicted_reward_ = self.env.reward_simulator.calc_expected_reward(
                    #     context=context_,
                    #     query=query_,
                    #     auxiliary_output=auxiliary_output_,
                    # )
                    # eval_predicted_reward_ = self.env.reward_simulator.calc_expected_reward(
                    #     context=context_,
                    #     query=query_,
                    #     auxiliary_output=sampled_output_,
                    # )
                    # marginal_iw_ = 1
                    # predicted_reward_ = 1.0
                    # eval_predicted_reward_ = 1.0
                    reward_ = 0

                    policy_loss_ = (
                        -(
                            marginal_iw_ * log_prob_ * (reward_ - predicted_reward_)
                        ).sum()
                        - (log_prob_ * eval_predicted_reward_).sum()
                    )

                    optimizer.zero_grad()
                    policy_loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += policy_loss_.item() / n_steps_per_epoch
                    train_loss_logs[j] = policy_loss_.item()

                if (i + 1) % n_epochs_per_log == 0:
                    if self.env is not None:
                        policy_values[
                            (i + 1) // n_epochs_per_log
                        ] = self.env.calc_expected_policy_value(model)
                        policy_value = policy_values[(i + 1) // n_epochs_per_log].item()
                        # print(policy_value)

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        if return_training_logs:
            output = (model, policy_values)
        else:
            output = model

        return output
