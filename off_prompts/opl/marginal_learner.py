"""Learner class for marginal density models."""
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Any, Dict, Union
from pathlib import Path
from tqdm.auto import tqdm
from operator import itemgetter
import random
import wandb

import torch
from torch.optim import Optimizer, Adam
from torch.utils.data import RandomSampler, DataLoader
from accelerate import Accelerator

from ..dataset.benchmark import SemiSyntheticDataset
from ..dataset.base import BaseFrozenLLM, BaseEncoder
from ..policy.base import (
    BasePolicy,
    BasePromptPolicyModel,
    BaseKernelMarginalDensityModel,
)
from ..policy.policy import TwoStagePolicy
from ..utils import check_logged_feedback, torch_seed, tokenize, to_device
from ..types import LoggedDataset, Sentence

from .dataset import TorchLoggedDataset

Policy = Union[BasePolicy, BasePromptPolicyModel, TwoStagePolicy]


@dataclass
class MarginalDensityLearner:
    model: BaseKernelMarginalDensityModel
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
    """Learner class for prompt reward predictor.

    Imported as: :class:`off_prompts.opl.MarginalDensityLearner`

    Parameters
    -------
    model: BaseKernelMarginalDensityModel
        Model to estimate the marginal density of a given sentence.

    action_list: Sentence
        Mapping from action id to discrete prompts.

    query_embeddings: torch.Tensor, shape (n_items, dim_query_emb), default=None
        Mapping from item id to its (query) embeddings.

    prompt_embeddings: torch.Tensor, shape (n_actions, dim_prompt_emb), default=None
        Mapping from item id to its (prompt) embeddings.

    optimizer: torch.optim.Optimizer, default=Adam
        Class of optimizer (not an instance).

    optimizer_kwargs: dict, default=None
        Arguments of optimizer.

    env: SyntheticDataset, default=None
        Online environment for evaluation.

    frozen_llm: BaseFrozenLLM, default=None
        Frozen LLM.

    query_encoder: BaseEncoder, default=None
        Encoder of query.

    sentence_encoder: BaseEncoder, default=None
        Encoder of sentence.

    random_state: int, default=None
        Random state.
    
    """

    def __post_init__(self):
        self.trained_model = None

        if self.random_state is None:
            raise ValueError("random_state must be given")

        if self.env is not None:
            if self.action_list is None:
                self.action_list = self.env.action_list
            if self.frozen_llm is None:
                self.frozen_llm = self.env.frozen_llm

        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {
                "lr": 1e-4,
            }

        self.prompt_for_frozen_llm = tokenize(
            self.action_list,
            tokenizer=self.frozen_llm.tokenizer,
            tokenizer_kwargs=self.frozen_llm.tokenizer_kwargs,
            device=self.frozen_llm.device,
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
        torch_seed(random_state)

    def simulation_training(
        self,
        logged_feedback: LoggedDataset,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 10,
        batch_size: int = 32,
        make_copy: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
        use_wandb: bool = False,
        experiment_name: str = "marginal_density",
    ):
        """Train marginal density model via simulating the sentence generation.

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

        experiment_name: str, default="marginal_density"
            Experiment name used for wandb reports.

        Return
        -------
        model: BaseKernelDensityModel
            Trained marginal density model.

        """
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
        item_id = logged_feedback["item_id"]
        logging_policy = logged_feedback["logging_policy"]

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
            train_dataset, batch_size=batch_size, shuffle=True,
        )
        model, train_dataloader = accelerator.prepare(model, train_dataloader)

        optimizer = Adam(model.parameters(), **self.optimizer_kwargs)

        losses = torch.zeros((n_epochs + 1,))

        if use_wandb:
            wandb.init(entity="", project=experiment_name)

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train marginal density estimator: Epoch {i}]")
                pbar.set_postfix(
                    {"loss": f"{losses[i]:.4g}",}
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
                    query_for_frozen_llm_ = to_device(
                        batch_["query_for_frozen_llm"], device=accelerator.device
                    )

                    action_1_ = logging_policy.sample_action(
                        context=context_, query=query_,
                    )

                    if self.prompt_for_frozen_llm is not None:
                        prompt_1_for_frozen_llm_ = {}
                        for key in self.prompt_for_frozen_llm:
                            prompt_1_for_frozen_llm_[key] = self.prompt_for_frozen_llm[
                                key
                            ][action_1_]
                    else:
                        prompt_1_for_frozen_llm_ = list(
                            itemgetter(*action_1_)(self.action_list)
                        )

                    sampled_sentence_1_ = self.frozen_llm.generate_output_sentence(
                        query=query_for_frozen_llm_, prompt=prompt_1_for_frozen_llm_,
                    )

                    action_2_ = logging_policy.sample_action(
                        context=context_, query=query_,
                    )

                    if self.prompt_for_frozen_llm is not None:
                        prompt_2_for_frozen_llm_ = {}
                        for key in self.prompt_for_frozen_llm:
                            prompt_2_for_frozen_llm_[key] = self.prompt_for_frozen_llm[
                                key
                            ][action_2_]
                    else:
                        prompt_2_for_frozen_llm_ = list(
                            itemgetter(*action_2_)(self.action_list)
                        )

                    sampled_sentence_2_ = self.frozen_llm.generate_output_sentence(
                        query=query_for_frozen_llm_, prompt=prompt_2_for_frozen_llm_,
                    )

                    predicted_marginal_density_ = model.estimate_marginal_density(
                        context=context_,
                        query=query_,
                        sentence=sampled_sentence_1_,
                        calc_gradient=True,
                    )
                    pairwise_weight_ = model.calc_pairwise_weight(
                        pivot_sentence=sampled_sentence_1_,
                        sampled_sentences=sampled_sentence_2_,
                        context=context_,
                        query=query_,
                    )

                    loss_ = torch.square(
                        predicted_marginal_density_ - pairwise_weight_
                    ).sum()

                    optimizer.zero_grad()
                    accelerator.backward(loss_)
                    optimizer.step()

                losses[i + 1] += loss_.item() / n_steps_per_epoch

                if use_wandb:
                    wandb.log(
                        {
                            "train_loss": losses[i + 1],
                            "avg_weight": pairwise_weight_.mean(),
                            "avg_predicted_weight": predicted_marginal_density_.mean(),
                        }
                    )

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        return model
