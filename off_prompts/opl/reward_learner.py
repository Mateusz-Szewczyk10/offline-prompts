"""Learner class for reward prediction models."""
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Union
from pathlib import Path
from tqdm.auto import tqdm
from operator import itemgetter
import random
import wandb

import torch
from torch.nn import MSELoss, BCELoss
from torch.optim import Optimizer, Adam
from torch.utils.data import RandomSampler, DataLoader

from accelerate import Accelerator
from sklearn.model_selection import train_test_split

from ..dataset.benchmark import SemiSyntheticDataset
from ..dataset.base import BaseFrozenLLM, BaseEncoder
from ..policy.base import (
    BasePolicy,
    BasePromptPolicyModel,
    BaseSentenceRewardModel,
    BasePromptRewardModel,
)
from ..policy.policy import UniformRandomPolicy, TwoStagePolicy
from ..utils import check_logged_feedback, torch_seed, tokenize, to_device
from ..types import LoggedDataset, Sentence

from .dataset import TorchLoggedDataset

Policy = Union[BasePolicy, BasePromptPolicyModel, TwoStagePolicy]


@dataclass
class SentenceRewardLearner:
    model: BaseSentenceRewardModel
    action_list: Sentence
    query_embeddings: Optional[torch.Tensor] = None
    prompt_embeddings: Optional[torch.Tensor] = None
    optimizer: Optimizer = Adam
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    env: Optional[SemiSyntheticDataset] = None
    frozen_llm: Optional[BaseFrozenLLM] = None
    query_encoder: Optional[BaseEncoder] = None
    sentence_encoder: Optional[BaseEncoder] = None
    random_state: Optional[int] = None

    """Learner class for sentence reward predictor.

    Imported as: :class:`off_prompts.opl.SentenceRewardLearner`

    Parameters
    -------
    model: BaseSentenceRewardModel
        Sentence reward model to train.

    action_list: Sentence
        Mapping from action id to discrete prompts.

    query_list: torch.Tensor, shape (n_items, dim_query_emb), default=None
        Mapping from item id to its (query) embeddings.

    prompt_embeddings: torch.Tensor, shape (n_items, dim_prompt_emb), default=None
        Mapping from item id to its (prompt) embeddings.

    optimizer: torch.optim.Optimizer, default=Adam
        Class of optimizer (not an instance).

    optimizer_kwargs: dict, default=None
        Arguments of optimizer.

    env: SemiSyntheticDataset, default=None
        Online environment for evaluation.

    frozen_llm: BaseFrozenLLMs, default=None
        Frozen LLMs to generate output sentence.

    query_encoder: BaseEncoder, default=None
        Encoder of query.

    sentence_encoder: BaseEncoder, default=None
        Encoder of sentence.

    random_state: int, default=None.
        Random state.
    
    """

    def __post_init__(self):
        self.trained_model = None
        self.device = self.model.device

        if self.random_state is None:
            raise ValueError("random_state must be given")

        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {
                "lr": 1e-4,
            }

        if self.env is not None:
            if self.action_list is None:
                self.action_list = self.env.action_list
            if self.frozen_llm is None:
                self.frozen_llm = self.env.frozen_llm

        self.prompt_for_frozen_llm = tokenize(
            self.action_list,
            tokenizer=self.frozen_llm.tokenizer,
            tokenizer_kwargs=self.frozen_llm.tokenizer_kwargs,
            device=self.frozen_llm.device,
        )

        self.uniform_policy = UniformRandomPolicy(
            action_list=self.action_list,
            device=self.device,
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

    def save(self, path: Path):
        """Save model."""
        torch.save(self.trained_model.state_dict(), path)

    def seed(self, random_state: Optional[int] = None):
        """Fix seed."""
        random.seed(random_state)
        torch_seed(random_state)

    def online_training(
        self,
        logging_policy: Optional[Policy] = None,
        loss_type: str = "MSE",
        n_epochs: int = 100,
        n_steps_per_epoch: int = 1000,
        batch_size: int = 32,
        val_ratio: float = 0.2,
        make_copy: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
        use_wandb: bool = False,
        experiment_name: str = "regression",
    ):
        """Train reward predictor from logged data.

        Parameters
        -------
        logging_policy: Policy, default=None
            Logging policy to collect data online.

        loss_type: {"MSE", "BCE"}, default="MSE"
            Whether to use mean-squared error (MSE) loss or binary cross entropy (BCE) loss.

        n_epochs: int, default=100
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=1000
            Number of gradient steps within an epoch.

        batch_size: int, default=32
            Batch size.

        val_ratio: float, default = 0.2
            Proportion of validation samples.

        make_copy: bool, default=False
            Whether to create copy of the model before training.

        save_path: Path, default=None
            Path to save the model.

        random_state: int, default=None
            Random state.

        use_wandb: bool, default=False
            Whether to use wandb to report the training statistics.

        experiment_name: str, default="regression"
            Experiment name used for wandb reports.

        Return
        -------
        model: SentenceRewardPredictor.
            Trained sentence reward predictor.

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

        if use_wandb:
            wandb.init(entity="", project=experiment_name)

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train sentence reward predictor: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "loss": f"{train_losses[i]:.4g}",
                    }
                )

                for j in range(n_steps_per_epoch):
                    (
                        user_id_,
                        item_id_,
                        context_,
                        query_,
                        query_embeddings_,
                    ) = self.env.context_query_generator.sample_context_and_query(
                        n_samples=batch_size,
                        return_query_embeddings=True,
                    )

                    action_ = logging_policy.sample_action(
                        context=context_,
                        action=action_,
                    )

                    if self.prompt_for_frozen_llm is not None:
                        prompt_for_frozen_llm_ = {}
                        for key in self.prompt_for_frozen_llm:
                            prompt_for_frozen_llm_[key] = self.prompt_for_frozen_llm[
                                key
                            ][action_]
                    else:
                        prompt_for_frozen_llm_ = list(
                            itemgetter(*action_)(self.action_list)
                        )

                    sentence_ = self.frozen_llm.generate_output_sentence(
                        query=query_,
                        prompt=prompt_for_frozen_llm_,
                    )
                    reward_ = self.env.sample_reward_given_output(
                        context=context_,
                        query=query_,
                        sentence=sentence_,
                        action=action_,
                    )

                    prediction_ = model(context_, query_embeddings_, sentence_)
                    loss_ = loss_fn(prediction_, reward_)

                    optimizer.zero_grad()
                    loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += loss_.item() / n_steps_per_epoch

                if use_wandb:
                    wandb.log(
                        {
                            "train_loss": train_losses[i + 1],
                            "avg_reward": reward_.mean(),
                            "avg_predicted_reward": prediction_.mean(),
                        }
                    )

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
        use_wandb: bool = False,
        experiment_name: str = "regression",
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
                        user_id,          #  -
                        item_id,          # (n_samples, )
                        action,           #  -
                        sentence,         # (n_samples, )
                        expected_reward,  #  -
                        reward,           # (n_samples, )
                        logging_policy,   #  -
                    ]

            See dataset/benchmark.py for the details of each key.

        is_pessimistic: bool, default=False
            Whether to train a pessimistic reward predictor or not.

        loss_type: {"MSE", "BCE"}, default="MSE"
            Whether to use mean-squared error (MSE) loss or binary cross entropy (BCE) loss.

        n_epochs: int, default=100
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=1000
            Number of gradient steps within an epoch.

        batch_size: int, default=32
            Batch size.

        val_ratio: float, default=0.2
            Proportion of validation samples.

        pessimistic_weight: float, default=0.1
            Weight on the pessimistic loss.

        make_copy: bool, default=False
            Whether to create copy of the model before training.

        save_path: Path, default=None
            Path to save the model.

        random_state: int, default=None
            Random state.

        use_wandb: bool, default=False
            Whether to use wandb to report the training statistics.

        experiment_name: str, default="regression"
            Experiment name used for wandb reports.

        Return
        -------
        model: SentenceRewardPredictor.
            Trained sentence reward predictor.

        """
        check_logged_feedback(logged_feedback)

        if is_pessimistic:
            if self.action_list is None:
                raise RuntimeError(
                    "self.action_list must be given when training a pessimistic reward predictor. "
                    "Please initialize the class with action_list."
                )
            if self.frozen_llms is None:
                raise RuntimeError(
                    "self.frozen_llms must be given when training a pessimistic reward predictor. "
                    "Please initialize the class with frozen_llms."
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

        (
            context_train,
            context_val,
            query_train,
            query_val,
            item_id_train,
            item_id_val,
            sentence_train,
            sentence_val,
            reward_train,
            reward_val,
        ) = train_test_split(
            logged_feedback["context"],
            logged_feedback["query"],
            logged_feedback["item_id"],
            logged_feedback["sentence"],
            logged_feedback["reward"],
            test_size=val_ratio,
            random_state=random_state,
            shuffle=True,
        )
        train_dataset = TorchLoggedDataset(
            context=context_train,
            query=query_train,
            item_id=item_id_train,
            sentence=sentence_train,
            reward=reward_train,
            action_list=self.action_list,
            query_embeddings=self.query_embeddings,
            prompt_embeddings=self.prompt_embeddings,
            query_encoder=self.query_encoder,
            sentence_encoder=self.sentence_encoder,
            frozen_llm=self.frozen_llm,
        )
        val_dataset = TorchLoggedDataset(
            context=context_val,
            query=query_val,
            item_id=item_id_val,
            sentence=sentence_val,
            reward=reward_val,
            action_list=self.action_list,
            query_embeddings=self.query_embeddings,
            prompt_embeddings=self.prompt_embeddings,
            query_encoder=self.query_encoder,
            sentence_encoder=self.sentence_encoder,
            frozen_llm=self.frozen_llm,
        )
        accelerator = Accelerator()

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        model, train_dataloader, val_dataloader = accelerator.prepare(
            model, train_dataloader, val_dataloader
        )

        if loss_type == "MSE":
            loss_fn = MSELoss()
        elif loss_type == "BCE":
            loss_fn = BCELoss()
        else:
            raise ValueError("loss_type must be either 'MSE' or 'BCE', but found False")

        train_losses = torch.zeros((n_epochs + 1,))
        test_losses = torch.zeros((n_epochs + 1,))

        if use_wandb:
            wandb.init(entity="", project=experiment_name)

        train_iterator = iter(train_dataloader)

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train sentence reward predictor: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "train_loss": f"{train_losses[i]:.4g}",
                        "test_loss": f"{test_losses[i]:.4g}",
                    }
                )

                for j in range(n_steps_per_epoch):
                    try:
                        batch_ = next(train_iterator)
                    except StopIteration:
                        # Reinitialize the iterator if the dataset is exhausted before reaching n_steps_per_epoch
                        train_iterator = iter(train_dataloader)
                        batch_ = next(train_iterator)

                    context_ = to_device(batch_["context"], device=accelerator.device)
                    query_ = to_device(batch_["query"], device=accelerator.device)
                    sentence_ = to_device(batch_["sentence"], device=accelerator.device)
                    reward_ = to_device(batch_["reward"], device=accelerator.device)

                    prediction_ = model(context_, query_, sentence_)
                    train_loss_ = loss_fn(prediction_, reward_)

                    if is_pessimistic:
                        query_for_frozen_llm_ = to_device(
                            batch_["query_for_frozen_llm"], device=accelerator.device
                        )

                        negative_action_ = torch.randint(n_actions, (batch_size,))

                        if self.prompt_for_frozen_llm is not None:
                            negative_prompt_for_frozen_llm_ = {}
                            for key in self.prompt_for_frozen_llm:
                                negative_prompt_for_frozen_llm_[
                                    key
                                ] = self.prompt_for_frozen_llm[key][
                                    negative_action_
                                ].to(
                                    accelerator.device
                                )
                        else:
                            negative_prompt_for_frozen_llm_ = list(
                                itemgetter(*negative_action_)(self.action_list)
                            )

                        negative_sentence_ = self.frozen_llm.generate_output_sentence(
                            query=query_for_frozen_llm_,
                            prompt=negative_prompt_for_frozen_llm_,
                        )
                        negative_prediction_ = model(
                            context_, query_, negative_sentence_
                        )

                        penalty_ = (negative_prediction_ - prediction_).mean()
                        train_loss_ += pessimistic_weight * penalty_

                    optimizer.zero_grad()
                    accelerator.backward(train_loss_)
                    optimizer.step()

                    train_losses[i + 1] += train_loss_.item() / n_steps_per_epoch

                for val_batch_ in val_dataloader:
                    val_context_ = to_device(
                        val_batch_["context"], device=accelerator.device
                    )
                    val_query_ = to_device(
                        val_batch_["query"], device=accelerator.device
                    )
                    val_sentence_ = to_device(
                        val_batch_["sentence"], device=accelerator.device
                    )
                    val_reward_ = to_device(
                        val_batch_["reward"], device=accelerator.device
                    )

                    val_prediction_ = model.predict_value(
                        context=val_context_,
                        query=val_query_,
                        sentence=val_sentence_,
                        return_cpu_tensor=False,
                        calc_gradient=True,
                    )
                    ratio_ = len(val_reward_) / len(val_dataloader)
                    test_losses[i + 1] += (
                        loss_fn(val_prediction_, val_reward_).item() * ratio_
                    )

                if use_wandb:
                    wandb.log(
                        {
                            "train_loss": train_losses[i + 1],
                            "avg_reward": val_reward_.mean(),
                            "avg_predicted_reward": val_prediction_.mean(),
                        }
                    )

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        return model


@dataclass
class PromptRewardLearner:
    model: BasePromptRewardModel
    action_list: Sentence
    query_embeddings: Optional[torch.Tensor] = None
    prompt_embeddings: Optional[torch.Tensor] = None
    optimizer: Optimizer = Adam
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    env: Optional[SemiSyntheticDataset] = None
    frozen_llm: Optional[BaseFrozenLLM] = None
    prompt_encoder: Optional[BaseEncoder] = None
    query_encoder: Optional[BaseEncoder] = None
    random_state: Optional[int] = None

    """Learner class for prompt reward predictor.

    Imported as: :class:`off_prompts.opl.PromptRewardLearner`

    Parameters
    -------
    model: BasePromptRewardModel
        Prompt reward model to train.

    action_list: Sentence
        Mapping from action id to discrete prompts. 

    query_list: torch.Tensor, shape (n_items, dim_query_emb), default=None
        Mapping from item id to its (query) embeddings.

    prompt_embeddings: torch.Tensor, shape (n_items, dim_prompt_emb), default=None
        Mapping from item id to its (prompt) embeddings.

    optimizer: torch.optim.Optimizer, default=Adam
        Class of optimizer (not an instance).

    optimizer_kwargs: dict, default=None
        Arguments of optimizer.

    env: SemiSyntheticDataset, default=None
        Online environment for evaluation.

    frozen_llm: BaseFrozenLLM, default=None
        Frozen LLMs to generate output sentence.

    query_encoder: BaseEncoder, default=None
        Encoder of query.

    prompt_encoder: BaseEncoder, default=None
        Encoder of prompt.

    random_state: int, default=None.
        Random state.
    
    """

    def __post_init__(self):
        self.trained_model = None
        self.n_actions = len(self.action_list)
        self.device = self.model.device

        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {
                "lr": 1e-4,
            }

        if self.random_state is None:
            raise ValueError("random_state must be given")

        if self.env is not None:
            if self.action_list is None:
                self.action_list = self.env.action_list
            if self.frozen_llm is None:
                self.frozen_llm = self.env.frozen_llm

        self.prompt_for_frozen_llm = tokenize(
            self.action_list,
            tokenizer=self.frozen_llm.tokenizer,
            tokenizer_kwargs=self.frozen_llm.tokenizer_kwargs,
            device=self.frozen_llm.device,
        )

        self.uniform_policy = UniformRandomPolicy(
            action_list=self.action_list,
            device=self.device,
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

    def save(self, path: Path):
        """Save Model."""
        torch.save(self.trained_model.state_dict(), path)

    def seed(self, random_state: Optional[int] = None):
        """Fix seed."""
        random.seed(random_state)
        torch_seed(random_state, device=self.device)

    def online_training(
        self,
        logging_policy: Optional[Policy] = None,
        loss_type: str = "MSE",
        n_epochs: int = 100,
        n_steps_per_epoch: int = 1000,
        batch_size: int = 32,
        make_copy: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
        use_wandb: bool = False,
        experiment_name: str = "regression",
    ):
        """Train reward predictor from logged data.

        Parameters
        -------
        logging_policy: Policy, default=None
            Logging policy to collect data online.

        loss_type: {"MSE", "BCE"}, default="MSE"
            Whether to use mean-squared error (MSE) loss or binary cross entropy (BCE) loss.

        n_epochs: int, default=100
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=1000
            Number of gradient steps within an epoch.

        batch_size: int, default=32
            Batch size.

        make_copy: bool, default=False
            Whether to create copy of the model before training.

        save_path: Path, default=None
            Path to save the model.

        random_state: int, default=None
            Random state.

        use_wandb: bool, default=False
            Whether to use wandb to report the training statistics.

        experiment_name: str, default="regression"
            Experiment name used for wandb reports.

        Return
        -------
        model: PromptRewardPredictor.
            Trained prompt reward predictor.

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

        if use_wandb:
            wandb.init(entity="", project=experiment_name)

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train prompt reward predictor: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "loss": f"{train_losses[i]:.4g}",
                    }
                )

                for j in range(n_steps_per_epoch):
                    (
                        user_id_,
                        item_id_,
                        context_,
                        query_,
                        query_embeddings_,
                    ) = self.env.context_query_generator.sample_context_and_query(
                        n_samples=batch_size,
                        return_query_embeddings=True,
                    )

                    action_ = logging_policy.sample_action(
                        context=context_,
                        query=query_embeddings_,
                    )
                    reward_ = self.env.sample_reward_given_action(
                        context=context_,
                        query=query_,
                        action=action_,
                    )

                    prediction_ = model.predict_value(
                        context=context_,
                        query=query_embeddings_,
                        action=action_,
                        return_cpu_tensor=False,
                        calc_gradient=True,
                    )
                    loss_ = loss_fn(prediction_, reward_)

                    optimizer.zero_grad()
                    loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += loss_.item() / n_steps_per_epoch

                if use_wandb:
                    wandb.log(
                        {
                            "train_loss": train_losses[i + 1],
                            "avg_reward": reward_.mean(),
                            "avg_predicted_reward": prediction_.mean(),
                        }
                    )

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
        use_wandb: bool = False,
        experiment_name: str = "regression",
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

        loss_type: {"MSE", "BCE"}, default="MSE"
            Whether to use mean-squared error (MSE) loss or binary cross entropy (BCE) loss.

        n_epochs: int, default=100
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=1000
            Number of gradient steps within an epoch.

        batch_size: int, default=32
            Batch size.

        val_ratio: float, default = 0.2
            Proportion of validation samples.

        pessimistic_weight: float, default=0.1
            Weight on the pessimistic loss.

        make_copy: bool, default=False
            Whether to create copy of the model before training.

        save_path: Path, default=None
            Path to save the model.

        random_state: int, default=None
            Random state.

        use_wandb: bool, default=False
            Whether to use wandb to report the training statistics.

        experiment_name: str, default="regression"
            Experiment name used for wandb reports.

        Return
        -------
        model: PromptRewardPredictor.
            Trained prompt reward predictor.

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

        (
            context_train,
            context_val,
            query_train,
            query_val,
            item_id_train,
            item_id_val,
            action_train,
            action_val,
            reward_train,
            reward_val,
        ) = train_test_split(
            logged_feedback["context"],
            logged_feedback["query"],
            logged_feedback["item_id"],
            logged_feedback["action"],
            logged_feedback["reward"],
            test_size=val_ratio,
            random_state=random_state,
            shuffle=True,
        )

        train_dataset = TorchLoggedDataset(
            context=context_train,
            query=query_train,
            item_id=item_id_train,
            action=action_train,
            reward=reward_train,
            action_list=self.action_list,
            query_embeddings=self.query_embeddings,
            prompt_embeddings=self.prompt_embeddings,
            prompt_encoder=self.prompt_encoder,
            query_encoder=self.query_encoder,
            frozen_llm=self.frozen_llm,
        )
        val_dataset = TorchLoggedDataset(
            context=context_val,
            query=query_val,
            item_id=item_id_val,
            action=action_val,
            reward=reward_val,
            action_list=self.action_list,
            query_embeddings=self.query_embeddings,
            prompt_embeddings=self.prompt_embeddings,
            prompt_encoder=self.prompt_encoder,
            query_encoder=self.query_encoder,
            frozen_llm=self.frozen_llm,
        )
        accelerator = Accelerator()

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        model, train_dataloader, val_dataloader = accelerator.prepare(
            model, train_dataloader, val_dataloader
        )

        if loss_type == "MSE":
            loss_fn = MSELoss()
        elif loss_type == "BCE":
            loss_fn = BCELoss()
        else:
            raise ValueError("loss_type must be either 'MSE' or 'BCE', but found False")

        train_losses = torch.zeros((n_epochs + 1,))
        test_losses = torch.zeros((n_epochs + 1,))

        if use_wandb:
            wandb.init(entity="", project=experiment_name)

        train_iterator = iter(train_dataloader)

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train prompt reward predictor: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "train_loss": f"{train_losses[i]:.4g}",
                        "test_loss": f"{test_losses[i]:.4g}",
                    }
                )

                for j in range(n_steps_per_epoch):
                    try:
                        batch_ = next(train_iterator)
                    except StopIteration:
                        # Reinitialize the iterator if the dataset is exhausted before reaching n_steps_per_epoch
                        train_iterator = iter(train_dataloader)
                        batch_ = next(train_iterator)

                    context_ = to_device(batch_["context"], device=accelerator.device)
                    query_ = to_device(batch_["query"], device=accelerator.device)
                    action_ = to_device(batch_["action"], device=accelerator.device)
                    reward_ = to_device(batch_["reward"], device=accelerator.device)

                    prediction_ = model.predict_value(
                        context=context_,
                        query=query_,
                        action=action_,
                        return_cpu_tensor=False,
                        calc_gradient=True,
                    )
                    train_loss_ = loss_fn(prediction_, reward_)

                    if is_pessimistic:
                        negative_action_ = torch.randint(self.n_actions, (batch_size,))
                        negative_prediction_ = model.predict_value(
                            context=context_,
                            query=query_,
                            action=negative_action_,
                            return_cpu_tensor=False,
                            calc_gradient=False,
                        )

                        penalty_ = (negative_prediction_ - prediction_).mean()
                        train_loss_ += pessimistic_weight * penalty_

                    optimizer.zero_grad()
                    accelerator.backward(train_loss_)
                    optimizer.step()

                    train_losses[i + 1] += train_loss_.item() / n_steps_per_epoch

                for val_batch_ in val_dataloader:
                    val_context_ = to_device(
                        val_batch_["context"], device=accelerator.device
                    )
                    val_query_ = to_device(
                        val_batch_["query"], device=accelerator.device
                    )
                    val_action_ = to_device(
                        val_batch_["action"], device=accelerator.device
                    )
                    val_reward_ = to_device(
                        val_batch_["reward"], device=accelerator.device
                    )

                    val_prediction_ = model.predict_value(
                        context=val_context_,
                        query=val_query_,
                        action=val_action_,
                        return_cpu_tensor=False,
                        calc_gradient=True,
                    )
                    ratio_ = len(val_reward_) / len(val_dataloader)
                    test_losses[i + 1] += (
                        loss_fn(val_prediction_, val_reward_).item() * ratio_
                    )

                if use_wandb:
                    wandb.log(
                        {
                            "train_loss": train_losses[i + 1],
                            "avg_reward": val_reward_.mean(),
                            "avg_predicted_reward": val_prediction_.mean(),
                        }
                    )

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        return model

    def online_cloning(
        self,
        teacher_model: BaseSentenceRewardModel,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 1000,
        batch_size: int = 32,
        make_copy: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
        use_wandb: bool = False,
        experiment_name: str = "regression",
    ):
        """Train reward predictor from logged data.

        Parameters
        -------
        teacher_model: BaseSentenceRewardModel
            (Pre-trained) sentence reward model.

        n_epochs: int, default=100
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=10
            Number of gradient steps within an epoch.

        batch_size: int, default=32
            Batch size.

        make_copy: bool, default=False
            Whether to create copy of the model before training.

        save_path: Path, default=None
            Path to save the model.

        random_state: int, default=None
            Random state.

        use_wandb: bool, default=False
            Whether to use wandb to report the training statistics.

        experiment_name: str, default="regression"
            Experiment name used for wandb reports.

        Return
        -------
        model: PromptRewardPredictor.
            Trained prompt reward predictor.

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

        if use_wandb:
            wandb.init(entity="", project=experiment_name)

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train prompt reward predictor: Epoch {i}]")
                pbar.set_postfix({"loss": f"{train_losses[i]:.4g}"})

                for j in range(n_steps_per_epoch):
                    (
                        user_id_,
                        item_id_,
                        context_,
                        query_,
                        query_embeddings_,
                    ) = self.env.context_query_generator.sample_context_and_query(
                        n_samples=batch_size,
                        return_query_embeddings=True,
                    )

                    action_ = self.uniform_policy.sample_action(
                        context_, query_embeddings_
                    )

                    if self.prompt_for_frozen_llm is not None:
                        prompt_for_frozen_llm_ = {}
                        for key in self.prompt_for_frozen_llm:
                            prompt_for_frozen_llm_[key] = self.prompt_for_frozen_llm[
                                key
                            ][action_]
                    else:
                        prompt_for_frozen_llm_ = list(
                            itemgetter(*action_)(self.action_list)
                        )

                    student_reward_ = model.predict_value(
                        context=context_,
                        query=query_embeddings_,
                        action=action_,
                        return_cpu_tensor=False,
                        calc_gradient=True,
                    )

                    sentence_ = self.frozen_llm.generate_output_sentence(
                        query=query_,
                        prompt=prompt_for_frozen_llm_,
                    )
                    teacher_reward_ = teacher_model.predict_value(
                        context=context_,
                        query=query_embeddings_,
                        generated_sentence=sentence_,
                        return_cpu_tensor=False,
                    )

                    train_loss_ = loss_fn(student_reward_, teacher_reward_)

                    optimizer.zero_grad()
                    train_loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += train_loss_.item() / n_steps_per_epoch

                if use_wandb:
                    wandb.log(
                        {
                            "train_loss": train_losses[i + 1],
                            "avg_reward": teacher_reward_.mean(),
                            "avg_predicted_reward": student_reward_.mean(),
                        }
                    )

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        return model

    def offline_cloning(
        self,
        logged_feedback: LoggedDataset,
        teacher_model: BaseSentenceRewardModel,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 1000,
        batch_size: int = 32,
        make_copy: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
        use_wandb: bool = False,
        experiment_name: str = "regression",
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

        teacher_model: BaseSentenceRewardModel
            (Pre-trained) sentence reward model.

        n_epochs: int, default=100
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=10
            Number of gradient steps within an epoch.

        batch_size: int, default=32
            Batch size.

        make_copy: bool, default=False
            Whether to create copy of the model before training.

        save_path: Path, default=None
            Path to save the model.

        random_state: int, default=None
            Random state.

        use_wandb: bool, default=False
            Whether to use wandb to report the training statistics.

        experiment_name: str, default="regression"
            Experiment name used for wandb reports.

        Return
        -------
        model: PromptRewardPredictor.
            Trained prompt reward predictor.

        """
        check_logged_feedback(logged_feedback)

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
        item_id = logged_feedback["item_id"]

        train_dataset = TorchLoggedDataset(
            context=context,
            query=query,
            item_id=item_id,
            query_embeddings=self.query_embeddings,
            query_encoder=self.query_encoder,
            frozen_llm=self.frozen_llm,
        )
        accelerator = Accelerator()

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        model, train_dataloader = accelerator.prepare(model, train_dataloader)

        loss_fn = MSELoss()
        train_losses = torch.zeros((n_epochs + 1,))

        if use_wandb:
            wandb.init(entity="", project=experiment_name)

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train prompt reward predictor: Epoch {i}]")
                pbar.set_postfix({"loss": f"{train_losses[i]:.4g}"})

                train_iterator = iter(train_dataloader)

                for j in range(n_steps_per_epoch):
                    try:
                        batch_ = next(train_iterator)
                    except StopIteration:
                        # Reinitialize the iterator if the dataset is exhausted before reaching n_steps_per_epoch
                        train_iterator = iter(train_dataloader)
                        batch_ = next(train_iterator)

                    context_ = to_device(batch_["context"], device=accelerator.device)
                    query_ = to_device(batch_["query"], device=accelerator.device)
                    query_for_frozen_llm_ = to_device(
                        batch_["query_for_frozen_llm"], device=accelerator.device
                    )

                    action_ = self.uniform_policy.sample_action(context_, query_)

                    if self.prompt_for_frozen_llm is not None:
                        prompt_for_frozen_llm_ = {}
                        for key in self.prompt_for_frozen_llm:
                            prompt_for_frozen_llm_[key] = self.prompt_for_frozen_llm[
                                key
                            ][action_].to(accelerator.device)
                    else:
                        prompt_for_frozen_llm_ = list(
                            itemgetter(*action_)(self.action_list)
                        )

                    student_reward_ = model.predict_value(
                        context=context_,
                        query=query_,
                        action=action_,
                        return_cpu_tensor=False,
                        calc_gradient=True,
                    )

                    sentence_ = self.frozen_llm.generate_output_sentence(
                        query=query_for_frozen_llm_,
                        prompt=prompt_for_frozen_llm_,
                    )
                    teacher_reward_ = teacher_model.predict_value(
                        context=context_,
                        query=query_,
                        generated_sentence=sentence_,
                        return_cpu_tensor=False,
                    )

                    train_loss_ = loss_fn(student_reward_, teacher_reward_)

                    optimizer.zero_grad()
                    accelerator.backward(train_loss_)
                    optimizer.step()

                    train_losses[i + 1] += train_loss_.item() / n_steps_per_epoch

                if use_wandb:
                    wandb.log(
                        {
                            "train_loss": train_losses[i + 1],
                            "avg_reward": teacher_reward_.mean(),
                            "avg_predicted_reward": student_reward_.mean(),
                        }
                    )

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        return model
