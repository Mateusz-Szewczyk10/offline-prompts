"""Learner class for (oracle) sentence embeddings."""
from dataclasses import dataclass
from typing import Optional, Any, Dict, Union
from pathlib import Path

import random
from copy import deepcopy
from tqdm.auto import tqdm
from operator import itemgetter

import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Optimizer, Adam

from src.dataset import SemiSyntheticDataset
from src.dataset import TransformerRewardSimulator
from src.dataset import BaseEncoder
from src.utils import torch_seed, to_device


@dataclass
class SentenceEmbeddingLearner:
    """Learner class of the sentence embeddings.
    
    Parameters
    -------
    model: BaseEncoder or nn.Module
        Sentence embedding model to train.

    env: SemiSyntheticDataset
        Online environment for data generation.

    optimizer: torch.optim.Optimizer, default=Adam
        Class of optimizer (not an instance).

    optimizer_kwargs: dict, default=None
        Arguments of optimizer.

    random_state: int, default=None
        Random state.
    
    """
    model: Union[BaseEncoder, nn.Module]
    env: SemiSyntheticDataset
    optimizer: Optimizer = Adam
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    random_state: Optional[int] = None

    def __post_init__(self):
        self.trained_model = None
        self.device = self.model.device

        if self.random_state is None:
            self.seed(random_state)

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
        """Save model."""
        torch.save(self.trained_model.state_dict(), path)

    def seed(self, random_state: Optional[int] = None):
        """Fix seed."""
        random.seed(random_state)
        torch_seed(random_state)

    def _calc_distance(
        self, 
        sentence_1: torch.Tensor, 
        sentence_2: torch.Tensor,
    ):
        """Calculate the distance between two sentences."""
        squared_distance = torch.sum(
            (sentence_1 - sentence_2) ** 2, dim=-1
        ) + 1e-10  # to avoid gradient vanishing
        
        dist = torch.sqrt(squared_distance) / self.model.dim_emb
        return dist

    def fit_online(
        self,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 10,
        return_training_process: bool = False,
        batch_size: int = 32,
        alpha: float = 10.0,
        make_copy: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
    ):
        """Fine-tune model.
        
        Parameters
        -------
        n_epochs: int, default=100
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=10
            Number of gradient steps within an epoch.

        return_training_process: bool, default=False
            Whether to return the training logs.

        alpha: float, default=10.0 (> 0)
            Constant multiplier of the sentence distance.

        make_copy: bool, default=False
            Whether to create copy of the model before training.

        save_path: Path, default=None
            Path to save the model.

        random_state: int, default=None
            Random state.
        
        Return
        -------
        model: BaseEncoder or nn.Module
            Trained sentence embedding model.
        
        """
        if random_state is None:
            random_state = self.random_state
        self.seed(random_state)

        if make_copy:
            model = deepcopy(self.model)
        else:
            model = self.model

        optimizer = self.optimizer(model.parameters(), **self.optimizer_kwargs)

        train_losses = torch.zeros((n_epochs + 1,))
        loss_fn = MSELoss()

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[fit sentence embeddings online: Epoch {i}]")
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
                    ) = self.env.context_query_loader.sample_context_and_query(
                        n_samples=batch_size,
                        return_query_embeddings=True,
                    )

                    action_ = torch.multinomial(
                        torch.ones((self.env.n_actions, ), device=self.device), 
                        num_samples=batch_size * 2, 
                        replacement=True,
                    )
                    action_1_, action_2_ = action_[:batch_size], action_[batch_size:]

                    prompt_1_ = list(itemgetter(*action_1_)(self.env.action_list))
                    prompt_2_ = list(itemgetter(*action_2_)(self.env.action_list))

                    sentence_1_ = self.env.frozen_llm.generate_output_sentence(
                        query=query_, prompt=prompt_1_,
                    )
                    sentence_2_ = self.env.frozen_llm.generate_output_sentence(
                        query=query_, prompt=prompt_2_,
                    )

                    reward_diff_ = self.env.reward_simulator.calc_expected_reward(
                        user_id=user_id_,
                        item_id=item_id_,
                        context=context_,
                        query=query_embeddings_,
                        action=action_,
                        sentence=sentence_1_,
                        baseline_sentence=sentence_2_,
                        return_cpu_tensor=False,
                    )
                    reward_dist_ = alpha * torch.abs(reward_diff_)

                    sentence_emb_1_ = model.encode(
                        sentence_1_, 
                        context=context_,
                        query=query_embeddings_,
                        calc_gradient=True,
                    )
                    sentence_emb_2_ = model.encode(
                        sentence_2_, 
                        context=context_,
                        query=query_embeddings_,
                        calc_gradient=True,
                    )
                    sentence_dist_ = self._calc_distance(sentence_emb_1_, sentence_emb_2_)

                    loss_ = loss_fn(sentence_dist_, reward_dist_)

                    optimizer.zero_grad()
                    torch.autograd.set_detect_anomaly(True)
                    loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += loss_.item() / n_steps_per_epoch

            print(train_losses[:i+1])

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        return model


    