"""Learner class for reward prediction models."""
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Any, Dict, Union
from pathlib import Path
from tqdm.auto import tqdm
import random

import torch
from torch.nn import MSELoss, BCELoss
from torch.optim import Optimizer, Adam

from sklearn.model_selection import train_test_split

from ..dataset.synthetic import SyntheticDataset
from ..dataset.function import AuxiliaryOutputGenerator
from ..policy.model import (
    BasePolicy,
    BaseActionPolicyModel,
    BaseOutputRewardModel,
    BaseActionRewardModel,
)
from ..policy.policy import UniformRandomPolicy, TwoStagePolicy
from ..utils import check_logged_feedback, torch_seed
from ..types import LoggedDataset

Policy = Union[BasePolicy, BaseActionPolicyModel, TwoStagePolicy]


@dataclass
class OutputRewardLearner:
    model: BaseOutputRewardModel
    action_list: torch.Tensor
    optimizer: Optimizer = Adam
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    env: Optional[SyntheticDataset] = None
    auxiliary_output_generator: Optional[AuxiliaryOutputGenerator] = None
    random_state: Optional[int] = None

    """Learner class for output reward predictor.

    Imported as: :class:`src.opl.SentenceRewardLearner`

    Parameters
    -------
    model: BaseOutputRewardModel
        Reward model to predict reward based on auxiliary output.

    action_list: torch.Tensor, shape (n_actions, dim_action_emb)
        Mapping from action id to discrete prompts.

    optimizer: torch.optim.Optimizer
        Class of optimizer (not an instance).

    optimizer_kwargs: dict
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

        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {
                "lr": 1e-4,
            }

        if self.env is not None:
            if self.action_list is None:
                self.action_list = self.env.action_list
            if self.auxiliary_output_generator is None:
                self.auxiliary_output_generator = self.env.auxiliary_output_generator

        if self.random_state is None:
            raise ValueError("random_state must be given")

        self.uniform_policy = UniformRandomPolicy(
            action_list=self.action_list,
            device=self.model.device,
            random_state=self.random_state,
        )

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

    def seed(self, random_state):
        """Fix seed."""
        random.seed(random_state)
        torch_seed(random_state)

    def online_training(
        self,
        logging_policy: Optional[Policy] = None,
        is_oracle_logging_policy: bool = False,
        is_oracle_clustering_logging_policy: bool = False,
        loss_type: str = "MSE",
        n_epochs: int = 100,
        n_steps_per_epoch: int = 1000,
        batch_size: int = 32,
        val_ratio: float = 0.2,
        make_copy: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
    ):
        """Train reward predictor from logged data.

        Parameters
        -------
        logging_policy: Policy, default=None
            Logging policy to collect data online.

        is_oracle_logging_policy: bool, default=False
            Whether the logging policy uses oracle expected reward.

        is_oracle_clustering_logging_policy: bool, default=False
            Whether the clustering logging policy uses oracle expected reward.

        loss_type: {"MSE", "BCE"}
            Whether to use mean-squared error (MSE) loss or binary cross entropy (BCE) loss.

        n_epochs: int, default=100.
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=1000.
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

        if self.env is None:
            raise RuntimeError(
                "self.env must be given when training a reward predictor online. Please initialize the class with env."
            )

        if logging_policy is None:
            logging_policy = self.uniform_policy

        if random_state is None:
            random_state = self.random_state
        self.seed(random_state)

        if make_copy:
            model = deepcopy(self.model)
        else:
            model = self.model

        optimizer = self.optimizer(model.parameters(), **self.optimizer_kwargs)

        if loss_type == "MSE":
            loss_fn = MSELoss()
        elif loss_type == "BCE":
            loss_fn = BCELoss()
        else:
            raise ValueError("loss_type must be either 'MSE' or 'BCE', but found False")

        train_losses = torch.zeros((n_epochs + 1,))

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train output reward predictor: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "loss": f"{train_losses[i]:.4g}",
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

                    if is_oracle_logging_policy or is_oracle_clustering_logging_policy:
                        expected_reward_ = (
                            self.env.calc_expected_reward_for_all_actions(
                                context=context_,
                                query=query_,
                            )
                        )
                    if is_oracle_logging_policy:
                        predicted_reward_ = expected_reward_
                    else:
                        predicted_reward_ = None

                    if is_oracle_clustering_logging_policy:
                        clustering_logging_predicted_reward_ = expected_reward_
                    else:
                        clustering_logging_predicted_reward_ = None

                    action_ = logging_policy.sample_action(
                        context=context_,
                        action_=action_,
                        predicted_reward=predicted_reward_,
                        clustering_logging_predicted_reward=clustering_logging_predicted_reward_,
                    )
                    auxiliary_output_ = (
                        self.env.auxiliary_output_generator.sample_auxiliary_output(
                            query=query_,
                            action_embedding=self.env.action_list[action_],
                        )
                    )
                    reward_ = self.env.sample_reward_given_output(
                        context=context_,
                        query=query_,
                        auxiliary_output=auxiliary_output_,
                    )

                    inputs_ = torch.cat((context_, query_, auxiliary_output_), dim=1)
                    prediction_ = model(inputs_)
                    loss_ = loss_fn(prediction_, reward_)

                    optimizer.zero_grad()
                    loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += loss_.item() / n_steps_per_epoch

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        return model

    def offline_training(
        self,
        logged_feedback: LoggedDataset,
        is_pessimistic: bool = False,
        loss_type: str = "MSE",
        n_epochs: int = 100,
        n_steps_per_epoch: int = 1000,
        batch_size: int = 32,
        val_ratio: float = 0.2,
        pessimistic_weight: float = 0.1,
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
                        sentence,         # (n_samples, )
                        expected_reward,  #  -
                        reward,           # (n_samples, )
                        logging_policy,   #  -
                    ]

            See dataset/benchmark.py for the details of each key.

        is_pessimistic: bool, default=False
            Whether to train a pessimistic reward predictor or not.

        loss_type: {"MSE", "BCE"}
            Whether to use mean-squared error (MSE) loss or binary cross entropy (BCE) loss.

        n_epochs: int, default=100.
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=1000.
            Number of gradient steps within an epoch.

        batch_size: int, default=32.
            Batch size.

        val_ratio: float, default=0.2.
            Proportion of validation samples.

        pessimistic_weight: float, default=0.1.
            Weight on the pessimistic loss.

        make_copy: bool, default=False.
            Whether to create copy of the model before training.

        save_path: Path, default=None.
            Path to save the model.

        random_state: int, default=None.
            Random state.

        """
        # check_logged_feedback(logged_feedback)

        if is_pessimistic:
            if self.action_list is None:
                raise RuntimeError(
                    "self.action_list must be given when training a pessimistic reward predictor. "
                    "Please initialize the class with action_list."
                )
            if self.auxiliary_output_generator is None:
                raise RuntimeError(
                    "self.auxiliary_output_generator must be given when training a pessimistic reward predictor. "
                    "Please initialize the class with auxiliary_output_generator."
                )

            n_actions = len(self.action_list)

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
        auxiliary_output = logged_feedback["auxiliary_output"]
        inputs = torch.cat((context, query, auxiliary_output), dim=1)

        (
            inputs_train,
            inputs_val,
            reward_train,
            reward_val,
            context_train,
            context_val,
            query_train,
            query_val,
        ) = train_test_split(
            inputs,
            logged_feedback["reward"],
            context,
            query,
            test_size=val_ratio,
            random_state=random_state,
            shuffle=True,
        )
        train_size = len(inputs_train)

        if loss_type == "MSE":
            loss_fn = MSELoss()
        elif loss_type == "BCE":
            loss_fn = BCELoss()
        else:
            raise ValueError("loss_type must be either 'MSE' or 'BCE', but found False")

        train_losses = torch.zeros((n_epochs + 1,))
        test_losses = torch.zeros((n_epochs + 1,))

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train output reward predictor: Epoch {i}]")
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
                    prediction_ = model(inputs_train[idx_])
                    train_loss_ = loss_fn(prediction_, reward_train[idx_])

                    if is_pessimistic:
                        context_ = context_train[idx_]
                        query_ = query_train[idx_]

                        action_ = torch.randint(n_actions, (batch_size,))
                        action_emb_ = self.action_list[action_]

                        negative_output_ = (
                            self.auxiliary_output_generator.sample_auxiliary_output(
                                query=query_,
                                action_embedding=action_emb_,
                            )
                        )
                        negative_input_ = torch.cat(
                            (context_, query_, negative_output_), dim=1
                        )
                        negative_prediction_ = model(negative_input_)

                        penalty_ = (negative_prediction_ - prediction_).mean()
                        train_loss_ += pessimistic_weight * penalty_

                    optimizer.zero_grad()
                    train_loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += train_loss_.item() / n_steps_per_epoch

                prediction = model(inputs_val)
                test_losses[i + 1] = loss_fn(prediction, reward_val).item()

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        return model


@dataclass
class ActionRewardLearner:
    model: BaseActionRewardModel
    auxiliary_output_generator: AuxiliaryOutputGenerator
    action_list: torch.Tensor
    optimizer: Optimizer = Adam
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    env: Optional[SyntheticDataset] = None
    random_state: Optional[int] = None

    """Learner class for action reward predictor.

    Imported as: :class:`src.opl.PromptRewardLearner`

    Parameters
    -------
    model: BaseActionRewardModel
        Reward model to predict reward based on output.

    auxiliary_output_generator: AuxiliaryOutputGenerator
        Generator of auxiliary output.

    action_list: torch.Tensor, shape (n_actions, dim_action_emb)
        Mapping from action id to discrete actions.

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

        if self.model.n_actions != len(self.action_list):
            raise ValueError(
                "Expected model.n_actions and len(actin_list) to be the same, but found False."
            )

        self.uniform_policy = UniformRandomPolicy(
            action_list=self.action_list,
            device=self.model.device,
            random_state=self.random_state,
        )

        self.n_actions = len(self.action_list)

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

    def online_training(
        self,
        logging_policy: Optional[Policy] = None,
        is_oracle_logging_policy: bool = False,
        is_oracle_clustering_logging_policy: bool = False,
        loss_type: str = "MSE",
        n_epochs: int = 100,
        n_steps_per_epoch: int = 1000,
        batch_size: int = 32,
        make_copy: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
    ):
        """Train reward predictor from logged data.

        Parameters
        -------
        logging_policy: Policy, default=None
            Logging policy to collect data online.

        is_oracle_logging_policy: bool, default=False
            Whether the logging policy uses oracle expected reward.

        is_oracle_clustering_logging_policy: bool, default=False
            Whether the clustering logging policy uses oracle expected reward.

        loss_type: {"MSE", "BCE"}
            Whether to use mean-squared error (MSE) loss or binary cross entropy (BCE) loss.

        n_epochs: int, default=100.
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=1000.
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
        if self.env is None:
            raise RuntimeError(
                "self.env must be given when training a reward predictor online. Please initialize the class with env."
            )

        if logging_policy is None:
            logging_policy = self.uniform_policy

        if random_state is None:
            random_state = self.random_state
        self.seed(random_state)

        if make_copy:
            model = deepcopy(self.model)
        else:
            model = self.model

        optimizer = self.optimizer(model.parameters(), **self.optimizer_kwargs)

        if loss_type == "MSE":
            loss_fn = MSELoss()
        elif loss_type == "BCE":
            loss_fn = BCELoss()
        else:
            raise ValueError("loss_type must be either 'MSE' or 'BCE', but found False")

        train_losses = torch.zeros((n_epochs + 1,))

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train action reward predictor: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "loss": f"{train_losses[i]:.4g}",
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

                    if is_oracle_logging_policy or is_oracle_clustering_logging_policy:
                        expected_reward_ = (
                            self.env.calc_expected_reward_for_all_actions(
                                context=context_,
                                query=query_,
                            )
                        )
                    if is_oracle_logging_policy:
                        predicted_reward_ = expected_reward_
                    else:
                        predicted_reward_ = None

                    if is_oracle_clustering_logging_policy:
                        clustering_logging_predicted_reward_ = expected_reward_
                    else:
                        clustering_logging_predicted_reward_ = None

                    action_ = logging_policy.sample_action(
                        context=context_,
                        query=query_,
                        predicted_reward=predicted_reward_,
                        clustering_logging_predicted_reward=clustering_logging_predicted_reward_,
                    )
                    reward_ = self.env.sample_reward_given_action(
                        context=context_,
                        query=query_,
                        action=action_,
                    )
                    embedding_ = self.action_list[action_]

                    inputs_ = torch.cat((context_, query_, embedding_), dim=1)
                    prediction_ = model(inputs_)
                    loss_ = loss_fn(prediction_, reward_)

                    optimizer.zero_grad()
                    loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += loss_.item() / n_steps_per_epoch

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        return model

    def offline_training(
        self,
        logged_feedback: LoggedDataset,
        is_pessimistic: bool = False,
        loss_type: str = "MSE",
        n_epochs: int = 100,
        n_steps_per_epoch: int = 1000,
        batch_size: int = 32,
        val_ratio: float = 0.2,
        pessimistic_weight: float = 0.1,
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
                        sentence,         # (n_samples, )
                        expected_reward,  #  -
                        reward,           # (n_samples, )
                        logging_policy,   #  -
                    ]

            See dataset/benchmark.py for the details of each key.

        is_pessimistic: bool, default=False
            Whether to train a pessimistic reward predictor or not.

        loss_type: {"MSE", "BCE"}
            Whether to use mean-squared error (MSE) loss or binary cross entropy (BCE) loss.

        n_epochs: int, default=100.
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=1000.
            Number of gradient steps within an epoch.

        batch_size: int, default=32.
            Batch size.

        val_ratio: float, default = 0.2.
            Proportion of validation samples.

        pessimistic_weight: float, default=0.1.
            Weight on the pessimistic loss.

        make_copy: bool, default = False.
            Whether to create copy of the model before training.

        save_path: Path, default=None.
            Path to save the model.

        random_state: int, default=None.
            Random state.

        """
        # check_logged_feedback(logged_feedback)

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
        action = logged_feedback["action"]
        action = self.action_list[action]
        inputs = torch.cat((context, query, action), dim=1)

        (
            inputs_train,
            inputs_val,
            context_train,
            context_val,
            query_train,
            query_val,
            reward_train,
            reward_val,
        ) = train_test_split(
            inputs,
            context,
            query,
            logged_feedback["reward"],
            test_size=val_ratio,
            random_state=random_state,
            shuffle=True,
        )
        train_size = len(inputs_train)

        if loss_type == "MSE":
            loss_fn = MSELoss()
        elif loss_type == "BCE":
            loss_fn = BCELoss()
        else:
            raise ValueError("loss_type must be either 'MSE' or 'BCE', but found False")

        train_losses = torch.zeros((n_epochs + 1,))
        test_losses = torch.zeros((n_epochs + 1,))

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train action reward predictor: Epoch {i}]")
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
                    prediction_ = model(inputs_train[idx_])
                    train_loss_ = loss_fn(prediction_, reward_train[idx_])

                    if is_pessimistic:
                        context_ = context_train[idx_]
                        query_ = query_train[idx_]

                        action_ = torch.randint(self.n_actions, (batch_size,))
                        action_emb_ = self.action_list[action_]

                        negative_inputs_ = torch.cat(
                            (context_, query_, action_emb_), dim=1
                        )
                        negative_prediction_ = model(negative_inputs_)

                        penalty_ = (negative_prediction_ - prediction_).mean()
                        train_loss_ += pessimistic_weight * penalty_

                    optimizer.zero_grad()
                    train_loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += train_loss_.item() / n_steps_per_epoch

                prediction = model(inputs_val)
                test_losses[i + 1] = loss_fn(prediction, reward_val).item()

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        return model

    def online_cloning(
        self,
        teacher_model: BaseOutputRewardModel,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 1000,
        batch_size: int = 32,
        make_copy: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
    ):
        """Train reward predictor from logged data.

        Parameters
        -------
        teacher_model: BaseOutputRewardModel
            (Pre-trained) output reward model.

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

        if random_state is None:
            random_state = self.random_state
        self.seed(random_state)

        if make_copy:
            model = deepcopy(self.model)
        else:
            model = self.model

        optimizer = self.optimizer(model.parameters(), **self.optimizer_kwargs)

        loss_fn = MSELoss()
        train_losses = torch.zeros((n_epochs + 1,))

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train action reward predictor: Epoch {i}]")
                pbar.set_postfix({"loss": f"{train_losses[i]:.4g}"})

                for j in range(n_steps_per_epoch):
                    context_and_query_ = (
                        self.env.context_query_generator.sample_context_and_query(
                            n_samples=batch_size,
                        )
                    )
                    context_ = context_and_query_[:, : self.env.dim_context]
                    query_ = context_and_query_[:, self.env.dim_context :]
                    action_ = self.uniform_policy.sample_action(context_, query_)

                    action_ = self.action_list[action_]
                    inputs_ = torch.cat((context_, query_, action_), dim=1)
                    student_reward_ = model(inputs_)

                    generated_auxiliary_output_ = (
                        self.auxiliary_output_generator.sample_auxiliary_output(
                            query_, action_
                        )
                    )
                    teacher_reward_ = teacher_model.predict_value(
                        context_, query_, generated_auxiliary_output_
                    )

                    train_loss_ = loss_fn(student_reward_, teacher_reward_)

                    optimizer.zero_grad()
                    train_loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += train_loss_.item() / n_steps_per_epoch

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        return model

    def offline_cloning(
        self,
        logged_feedback: LoggedDataset,
        teacher_model: BaseOutputRewardModel,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 1000,
        batch_size: int = 32,
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

        teacher_model: BaseOutputRewardModel
            (Pre-trained) output reward model.

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

        train_size = len(context)

        loss_fn = MSELoss()
        train_losses = torch.zeros((n_epochs + 1,))

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train action reward predictor: Epoch {i}]")
                pbar.set_postfix({"loss": f"{train_losses[i]:.4g}"})

                for j in range(n_steps_per_epoch):
                    idx_ = torch.multinomial(
                        torch.ones(train_size), num_samples=batch_size, replacement=True
                    )
                    context_ = context[idx_]
                    query_ = query[idx_]
                    action_ = self.uniform_policy.sample_action(context_, query_)

                    action_ = self.action_list[action_]
                    inputs_ = torch.cat((context_, query_, action_), dim=1)
                    student_reward_ = model(inputs_)

                    generated_auxiliary_output_ = (
                        self.output_generator.sample_auxiliary_output(query_, action_)
                    )
                    teacher_reward_ = teacher_model.predict_value(
                        context_, query_, generated_auxiliary_output_
                    )

                    train_loss_ = loss_fn(student_reward_, teacher_reward_)

                    optimizer.zero_grad()
                    train_loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += train_loss_.item() / n_steps_per_epoch

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        return model
