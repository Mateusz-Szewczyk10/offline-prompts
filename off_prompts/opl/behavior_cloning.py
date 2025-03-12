"""Learner class for behavior cloning policies."""
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Any, Dict, Union
from pathlib import Path
from tqdm.auto import tqdm
import random
import wandb

import torch
from torch.optim import Optimizer, Adam
from torch.utils.data import RandomSampler, DataLoader

from accelerate import Accelerator
from sklearn.model_selection import train_test_split

from ..dataset.benchmark import SemiSyntheticDataset
from ..dataset.base import BaseEncoder
from ..policy.base import BasePolicy, BasePromptPolicyModel
from ..policy.policy import TwoStagePolicy
from ..utils import torch_seed, to_device
from ..types import LoggedDataset

from .dataset import TorchLoggedDataset

Policy = Union[BasePolicy, BasePromptPolicyModel, TwoStagePolicy]


@dataclass
class BehaviorCloningLearner:
    model: BasePromptPolicyModel
    query_embeddings: Optional[torch.Tensor] = None
    optimizer: Optimizer = Adam
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    query_encoder: Optional[BaseEncoder] = None
    env: Optional[SemiSyntheticDataset] = None
    random_state: Optional[int] = None

    """Learner class for behavior cloning.

    Imported as: :class:`off_prompts.opl.BehaviorCloningLearner`

    Parameters
    -------
    model: BasePromptPolicyModel
        Policy.

    query_embeddings: torch.Tensor, shape (n_items, dim_query_emb), default=None
        Mapping from item id to its (query) embeddings.

    optimizer: torch.optim.Optimizer, default=Adam
        Class of optimizer (not an instance).

    optimizer_kwargs: dict, default=None
        Arguments of optimizer.

    query_encoder: BaseEncoder, default=None
        Encoder of queries.

    env: SyntheticDataset, default=None
        Online environment for evaluation.

    random_state: int, default=None
        Random state.
    
    """

    def __post_init__(self):
        self.trained_model = None

        if self.random_state is None:
            raise ValueError("random_state must be given")

        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {
                "lr": 1e-4,
            }

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
        torch_seed(random_state)

    def online_cloning(
        self,
        teacher_model: Policy,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 10,
        batch_size: int = 32,
        make_copy: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
        use_wandb: bool = False,
        experiment_name: str = "BC",
    ):
        """Train reward predictor from logged data.

        Parameters
        -------
        teacher_model: BasePolicy or BasePromptPolicyModel or TwoStagePolicy
            (Pre-trained) policy.

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

        experiment_name: str, default="BC"
            Experiment name used for wandb reports.

        Return
        -------
        model: Policy
            Trained (i.e., behavior cloned) policy.

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

        if use_wandb:
            wandb.init(entity="", project=experiment_name)

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(
                    f"[train behavior cloning policy (online): Epoch {i}]"
                )
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

                    teacher_action_choice_prob_ = teacher_model.calc_action_choice_probability(
                        context=context_, query=query_embeddings_, return_cpu_tensor=False
                    )
                    student_action_choice_prob_ = self.model.calc_action_choice_probability(
                        context=context_,
                        query=query_embeddings_,
                        return_cpu_tensor=False,
                        calc_gradient=True,
                    )

                    mse_ = (
                        student_action_choice_prob_ - teacher_action_choice_prob_
                    ) ** 2
                    loss_ = (teacher_action_choice_prob_ * mse_).sum(dim=1).mean()

                    optimizer.zero_grad()
                    loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += loss_.item() / n_steps_per_epoch

                if use_wandb:
                    wandb.log(
                        {
                            "train_loss": train_losses[i + 1],
                        }
                    )

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        return model

    def offline_cloning(
        self,
        logged_feedback: LoggedDataset,
        teacher_model: Policy,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 10,
        batch_size: int = 32,
        val_ratio: float = 0.2,
        make_copy: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
        use_wandb: bool = False,
        experiment_name: str = "BC",
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
                        sentence,         #  -
                        expected_reward,  #  -
                        reward,           #  -
                        logging_policy,   #  -
                    ]

            See dataset/benchmark.py for the details of each key.

        teacher_model: BasePolicy or BaseActionPolicyModel or TwoStagePolicy
            (Pre-trained) policy.

        n_epochs: int, default=100
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=10
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

        experiment_name: str, default="BC"
            Experiment name used for wandb reports.

        Return
        -------
        model: Policy
            Trained (i.e., behavior cloned) policy.

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
        item_id = logged_feedback["item_id"]

        context_train, context_val, query_train, query_val = train_test_split(
            context,
            query,
            test_size=val_ratio,
            random_state=random_state,
            shuffle=True,
        )

        train_dataset = TorchLoggedDataset(
            context=context_train, 
            query=query_train, 
            item_id=item_id,
            query_embeddings=self.query_embeddings,
            query_encoder=self.query_encoder,
        )
        val_dataset = TorchLoggedDataset(
            context=context_val, 
            query=query_val, 
            item_id=item_id,
            query_embeddings=self.query_embeddings,
            query_encoder=self.query_encoder,
        )
        accelerator = Accelerator()
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,)
        model, train_dataloader, val_dataloader = accelerator.prepare(
            model, train_dataloader, val_dataloader
        )

        train_losses = torch.zeros((n_epochs + 1,))
        test_losses = torch.zeros((n_epochs + 1,))

        if use_wandb:
            wandb.init(entity="", project=experiment_name)

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

                    teacher_action_choice_prob_ = teacher_model.calc_action_choice_probability(
                        context=context_, query=query_, return_cpu_tensor=False,
                    )
                    student_action_choice_prob_ = self.model.calc_action_choice_probability(
                        context=context_,
                        query=query_,
                        return_cpu_tensor=False,
                        calc_gradient=True,
                    )
                    mse_ = (
                        student_action_choice_prob_ - teacher_action_choice_prob_
                    ) ** 2
                    train_loss_ = (teacher_action_choice_prob_ * mse_).sum(dim=1).mean()

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

                    teacher_action_choice_prob = teacher_model.calc_action_choice_probability(
                        context=val_context_, query=val_query_, return_cpu_tensor=False,
                    )
                    student_action_choice_prob = self.model.calc_action_choice_probability(
                        context=val_context_,
                        query=val_query_,
                        return_cpu_tensor=False,
                        calc_gradient=True,
                    )
                    se = (student_action_choice_prob - teacher_action_choice_prob) ** 2
                    test_losses[i + 1] += (teacher_action_choice_prob * se).sum(
                        dim=1
                    ) / len(val_dataloader)

                if use_wandb:
                    wandb.log(
                        {
                            "train_loss": train_losses[i + 1],
                        }
                    )

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        return model
