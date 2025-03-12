"""Learner class for reward prediction models."""
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Any, Dict, Union
from pathlib import Path
from tqdm.auto import tqdm
import random

import torch
from torch.optim import Optimizer, Adam

from sklearn.model_selection import train_test_split

from ..dataset.synthetic import SyntheticDataset
from ..policy.model import BasePolicy, BaseActionPolicyModel
from ..policy.policy import TwoStagePolicy
from ..utils import check_logged_feedback, torch_seed
from ..types import LoggedDataset

Policy = Union[BasePolicy, BaseActionPolicyModel, TwoStagePolicy]


@dataclass
class BehaviorCloningLearner:
    model: BaseActionPolicyModel
    optimizer: Optimizer = Adam
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    env: Optional[SyntheticDataset] = None
    random_state: Optional[int] = None

    """Learner class for behavior cloning.

    Imported as: :class:`src.opl.BehaviorCloningLearner`

    Parameters
    -------
    model: BaseDiscreteActionRewardModel
        Policy.

    optimizer: torch.optim.Optimizer
        Class of optimizer (not an instance).

    optimizer_kwargs: dict
        Arguments of optimizer.

    env: SyntheticDataset, default=None
        Online environment for evaluation.

    random_state: int, default=None.
        Random state.
    
    """

    def __post_init__(self):
        self.trained_model = None

        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {
                "lr": 1e-4,
            }

        if self.random_state is None:
            raise ValueError("random_state must be given")

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
        """Save Model."""
        torch.save(self.trained_model.state_dict(), path)

    def seed(self, random_state):
        """Fix seed."""
        random.seed(random_state)
        torch_seed(random_state)

    def online_cloning(
        self,
        teacher_model: Policy,
        is_oracle_teacher_policy: bool = False,
        is_oracle_clustering_logging_policy: bool = False,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 10,
        batch_size: int = 32,
        make_copy: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
    ):
        """Train reward predictor from logged data.

        Parameters
        -------
        teacher_model: BasePolicy or BaseActionPolicyModel or TwoStagePolicy
            (Pre-trained) policy.

        is_oracle_teacher_policy: bool, default=False
                Whether the logging policy uses oracle expected reward.

        is_oracle_clustering_logging_policy: bool, default=False
            Whether the clustering logging policy uses oracle expected reward.

        n_epochs: int, default=100.
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=10.
            Number of gradient steps within an epoch.

        batch_size: int, default=32.
            Batch size.

        make_copy: bool, default = False.
            Whether to create copy of the model before training.

        save_path: Path, default=None.
            Path to save the model.

        random_state: int, default=None.
            Random state.

        """
        # check_logged_feedback(logged_feedback)

        if self.env is None:
            raise RuntimeError(
                "self.env must be given when running behavior cloning online. Please initialize the class with env."
            )

        if self.model.n_actions != teacher_model.n_actions:
            raise ValueError(
                "Expected model.n_actions and teacher_model.n_actions to be the same, but found False."
            )

        if random_state is None:
            random_state = self.random_state
        self.seed(random_state)

        if make_copy:
            model = deepcopy(self.model)
        else:
            model = self.model

        optimizer = self.optimizer(model.parameters(), **self.optimizer_kwargs)

        train_losses = torch.zeros((n_epochs + 1,))

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(
                    f"[train behavior cloning policy (online): Epoch {i}]"
                )
                pbar.set_postfix({"loss": f"{train_losses[i]:.4g}"})

                for j in range(n_steps_per_epoch):
                    context_and_query_ = (
                        self.env.context_query_generator.sample_context_and_query(
                            n_samples=batch_size,
                        )
                    )
                    context_ = context_and_query_[:, : self.env.dim_context]
                    query_ = context_and_query_[:, self.env.dim_context :]

                    if is_oracle_teacher_policy or is_oracle_clustering_logging_policy:
                        tmp_predicted_reward_ = (
                            self.env.calc_expected_reward_for_all_actions(
                                context=context_,
                                query=query_,
                            )
                        )

                    if is_oracle_teacher_policy:
                        predicted_reward_ = tmp_predicted_reward_
                    else:
                        predicted_reward_ = None

                    if is_oracle_clustering_logging_policy:
                        clustering_logging_predicted_reward_ = tmp_predicted_reward_
                    else:
                        clustering_logging_predicted_reward_ = None

                    teacher_action_choice_prob_ = teacher_model.calc_action_choice_probability(
                        context=context_,
                        query=query_,
                        predicted_reward=predicted_reward_,
                        clustering_logging_predicted_reward=clustering_logging_predicted_reward_,
                    )
                    student_action_choice_prob_ = (
                        self.model.calc_action_choice_probability(
                            context=context_,
                            query=query_,
                            calc_gradient=True,
                        )
                    )
                    mse_ = (
                        student_action_choice_prob_ - teacher_action_choice_prob_
                    ) ** 2
                    loss_ = (teacher_action_choice_prob_ * mse_).sum(dim=1).mean()

                    optimizer.zero_grad()
                    loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += loss_.item() / n_steps_per_epoch

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        return model

    def offline_cloning(
        self,
        logged_feedback: LoggedDataset,
        teacher_model: Policy,
        is_oracle_teacher_policy: bool = False,
        is_oracle_clustering_logging_policy: bool = False,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 10,
        batch_size: int = 32,
        val_ratio: float = 0.2,
        make_copy: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
    ):
        """Train reward predictor from logged data.

        Parameters
        -------
        logged_feedback: LoggedDataset
            Logged data, which contains the following keys.

            .. code-block:: python

                    key: [
                        context,          # (n_samples, )
                        query,            # (n_samples, )
                        action,           #  -
                        sentence,         #  -
                        expected_reward,  #  -
                        reward,           #  -
                        logging_policy,   #  -
                    ]

            See dataset/benchmark.py for the details of each key.

        teacher_model: BasePolicy or BaseActionPolicyModel or TwoStagePolicy
            (Pre-trained) policy.

        is_oracle_teacher_policy: bool, default=False
                Whether the logging policy uses oracle expected reward.

        is_oracle_clustering_logging_policy: bool, default=False
            Whether the clustering logging policy uses oracle expected reward.

        n_epochs: int, default=100.
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=10.
            Number of gradient steps within an epoch.

        batch_size: int, default=32.
            Batch size.

        val_ratio: float, default = 0.2.
            Proportion of validation samples.

        make_copy: bool, default = False.
            Whether to create copy of the model before training.

        save_path: Path, default=None.
            Path to save the model.

        random_state: int, default=None.
            Random state.

        """
        # check_logged_feedback(logged_feedback)

        if self.model.n_actions != teacher_model.n_actions:
            raise ValueError(
                "Expected model.n_actions and teacher_model.n_actions to be the same, but found False."
            )

        if random_state is None:
            random_state = self.random_state
        self.seed(random_state)

        if make_copy:
            model = deepcopy(self.model)
        else:
            model = self.model

        optimizer = self.optimizer(model.parameters(), **self.optimizer_kwargs)

        context = logged_feedback["context"]
        query = logged_feedback["query"]

        (
            context_train,
            context_val,
            query_train,
            query_val,
        ) = train_test_split(
            context,
            query,
            test_size=val_ratio,
            random_state=random_state,
            shuffle=True,
        )

        if is_oracle_teacher_policy or is_oracle_clustering_logging_policy:
            predicted_reward_train_ = self.env.calc_expected_reward_for_all_actions(
                context=context_train,
                query=query_train,
            )
            predicted_reward_val_ = self.env.calc_expected_reward_for_all_actions(
                context=context_val,
                query=query_val,
            )

        if is_oracle_teacher_policy:
            predicted_reward_train = predicted_reward_train_
            predicted_reward_val = predicted_reward_val_
        else:
            predicted_reward_train = None
            predicted_reward_val = None

        if is_oracle_clustering_logging_policy:
            clustering_logging_predicted_reward_train = predicted_reward_train_
            clustering_logging_predicted_reward_val = predicted_reward_val_
        else:
            clustering_logging_predicted_reward_train = None
            clustering_logging_predicted_reward_val = None

        train_size = len(context_train)

        train_losses = torch.zeros((n_epochs + 1,))
        test_losses = torch.zeros((n_epochs + 1,))

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(
                    f"[train behavior cloning policy (offline): Epoch {i}]"
                )
                pbar.set_postfix(
                    {
                        "train_loss": f"{train_losses[i]:.4g}",
                        "test_loss": f"{test_losses[i]:.4g}",
                    }
                )

                for j in range(n_steps_per_epoch):
                    idx_ = torch.multinomial(
                        torch.ones(train_size), num_samples=batch_size, replacement=True
                    )
                    context_ = context_train[idx_]
                    query_ = query_train[idx_]

                    if predicted_reward_train is not None:
                        predicted_reward_train_ = predicted_reward_train[idx_]
                    else:
                        predicted_reward_train_ = None

                    if clustering_logging_predicted_reward_train is not None:
                        clustering_logging_predicted_reward_train_ = (
                            clustering_logging_predicted_reward_train[idx_]
                        )
                    else:
                        clustering_logging_predicted_reward_train_ = None

                    teacher_action_choice_prob_ = teacher_model.calc_action_choice_probability(
                        context=context_,
                        query=query_,
                        predicted_reward=predicted_reward_train_,
                        clustering_logging_predicted_reward=clustering_logging_predicted_reward_train_,
                    )
                    student_action_choice_prob_ = (
                        self.model.calc_action_choice_probability(
                            context=context_,
                            query=query_,
                            calc_gradient=True,
                        )
                    )
                    mse_ = (
                        student_action_choice_prob_ - teacher_action_choice_prob_
                    ) ** 2
                    train_loss_ = (teacher_action_choice_prob_ * mse_).sum(dim=1).mean()

                    optimizer.zero_grad()
                    train_loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += train_loss_.item() / n_steps_per_epoch

                teacher_action_choice_prob = teacher_model.calc_action_choice_probability(
                    context=context_val,
                    query=query_val,
                    predicted_reward=predicted_reward_val,
                    clustering_logging_predicted_reward=clustering_logging_predicted_reward_val,
                )
                student_action_choice_prob = self.model.calc_action_choice_probability(
                    context=context_val,
                    query=query_val,
                    calc_gradient=True,
                )
                mse = (student_action_choice_prob - teacher_action_choice_prob) ** 2
                test_losses[i + 1] = (
                    (teacher_action_choice_prob * mse).sum(dim=1).mean()
                )

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        return model
