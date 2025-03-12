"""Learner class for marginal density models."""
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Any, Dict, Union
from pathlib import Path
from tqdm.auto import tqdm
import random

import torch
from torch.optim import Optimizer, Adam

from ..dataset.synthetic import SyntheticDataset
from ..dataset.function import AuxiliaryOutputGenerator
from ..policy.model import (
    BasePolicy,
    BaseActionPolicyModel,
    BaseKernelMarginalDensityModel,
)
from ..policy.policy import TwoStagePolicy
from ..utils import check_logged_feedback, torch_seed
from ..types import LoggedDataset

Policy = Union[BasePolicy, BaseActionPolicyModel, TwoStagePolicy]


@dataclass
class MarginalDensityLearner:
    model: BaseKernelMarginalDensityModel
    action_list: torch.Tensor
    optimizer: Optimizer = Adam
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    env: Optional[SyntheticDataset] = None
    auxiliary_output_generator: Optional[AuxiliaryOutputGenerator] = None
    random_state: Optional[int] = None

    """Learner class for prompt reward predictor.

    Imported as: :class:`src.opl.PromptRewardLearner`

    Parameters
    -------
    model: BaseKernelMarginalDensityModel
        Model to estimate the marginal density of a given sentence.

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

        if self.env is not None:
            if self.action_list is None:
                self.action_list = self.env.action_list
            if self.auxiliary_output_generator is None:
                self.auxiliary_output_generator = self.env.auxiliary_output_generator

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

    def simulation_training(
        self,
        logged_feedback: LoggedDataset,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 100,
        batch_size: int = 32,
        make_copy: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
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
                        action,           #  -
                        sentence,         #  -
                        expected_reward,  #  -
                        reward,           #  -
                        logging_policy,   #  -
                    ]

            See dataset/benchmark.py for the details of each key.

        n_epochs: int, default=100.
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=1000.
            Number of gradient steps within an epoch.

        batch_size: int, default=32.
            Batch size.

        make_copy: bool, default=False.
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

        context = logged_feedback["context"]
        query = logged_feedback["query"]
        logging_policy = logged_feedback["logging_policy"]
        n_samples = len(context)

        optimizer = Adam(model.parameters(), **self.optimizer_kwargs)

        losses = torch.zeros((n_epochs + 1,))

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train marginal density estimator: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "loss": f"{losses[i]:.4g}",
                    }
                )

                for j in range(n_steps_per_epoch):
                    idx_ = torch.multinomial(
                        torch.ones(n_samples), num_samples=batch_size, replacement=True
                    )
                    context_ = context[idx_]
                    query_ = query[idx_]

                    action_1_ = logging_policy.sample_action(
                        context=context_,
                        query=query_,
                    )
                    sampled_outputs_1_ = (
                        self.auxiliary_output_generator.sample_auxiliary_output(
                            query=query_,
                            action_embedding=self.action_list[action_1_],
                        )
                    )

                    action_2_ = logging_policy.sample_action(
                        context=context_,
                        query=query_,
                    )
                    sampled_outputs_2_ = (
                        self.auxiliary_output_generator.sample_auxiliary_output(
                            query=query_,
                            action_embedding=self.action_list[action_2_],
                        )
                    )

                    predicted_marginal_density_ = model.estimate_marginal_density(
                        context=context_,
                        query=query_,
                        auxiliary_output=sampled_outputs_1_,
                        calc_gradient=True,
                    )
                    pairwise_weight_ = model.calc_pairwise_weight(
                        pivot_output=sampled_outputs_1_,
                        sampled_outputs=sampled_outputs_2_,
                    )

                    loss_ = torch.square(
                        predicted_marginal_density_ - pairwise_weight_
                    ).sum()

                    optimizer.zero_grad()
                    loss_.backward()
                    optimizer.step()

                losses[i + 1] += loss_.item() / n_steps_per_epoch

                # ### debugging
                # if i == 0 or i % 10 == 0:
                #     print()
                #     print(loss_.item())
                #     print(predicted_marginal_density_[:10])
                #     print(pairwise_weight_[:10])
                #     print()

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        return model
