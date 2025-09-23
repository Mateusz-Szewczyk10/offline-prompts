"""Learner class for reward simulator."""
from dataclasses import dataclass
from typing import Optional, Any, Dict, Union
from pathlib import Path

import random
from copy import deepcopy
from tqdm.auto import tqdm

import torch
from torch.nn import MSELoss, BCELoss
from torch.optim import Optimizer, Adam
from torch.utils.data import RandomSampler, DataLoader, Dataset
from accelerate import Accelerator

from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    DistilBertTokenizerFast,
)

import pandas as pd
from sklearn.model_selection import train_test_split

from off_prompts.dataset import BaseRewardSimulator
from off_prompts.dataset import TransformerRewardSimulator
from off_prompts.utils import torch_seed, tokenize, to_device
from off_prompts.types import Tokens


class MovielensDataset(Dataset):
    """Torch dataset class for MovieLens.

    Bases: :class:`torch.utils.data.Dataset`

    Imported as: :class:`off_prompts.dataset.assets.reward_finetuner.MovieLensDataset`

    Parameters
    -------
    df: DataFrame
        Pandas dataframe of the preprocessed movielens dataset.

    user_id_column: str, default="user_id"
        Column name if the user id column.

    item_id_column: str, default="item_id"
        Column name if the item id column.

    sentence_column: str, default="description"
        Column name if the sentence description of the items.

    reward_column: str, default="reward"
        Column name if the reward column.

    tokenizer: PreTrainedTokenizer or PreTrainedTokenizerFast
        Tokenizer of the (frozen) LLM.

    tokenizer_kwargs: dict
        Tokenizer kwargs.

    device: str, default="cuda"
        Device.

    """

    def __init__(
        self,
        df: pd.DataFrame,
        user_id_column: str = "user_id",
        item_id_column: str = "item_id",
        sentence_column: str = "description",
        reward_column: str = "reward",
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
    ):
        if tokenizer is None:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(
                "distilbert-base-uncased",
                truncation=True,
                do_lower_case=True,
            )

        if tokenizer_kwargs is None:
            self.tokenizer_kwargs = {
                "add_special_tokens": True,
                "max_length": 20,
                "return_tensors": "pt",
            }
        user_id = df[user_id_column].values
        item_id = df[item_id_column].values
        reward = df[reward_column].values

        self.user_id = torch.LongTensor(user_id).to(device)
        self.item_id = torch.LongTensor(item_id).to(device)
        self.reward = torch.FloatTensor(reward).to(device)

        sentence = list(df[sentence_column].values)
        self.tokens = tokenize(
            sentence,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            device=device,
        )

    def __len__(self):
        return len(self.user_id)

    def __getitem__(self, idx):
        tokens = {}
        for key in self.tokens:
            tokens[key] = self.tokens[key][idx]

        batch = {
            "user_id": self.user_id[idx],
            "item_id": self.item_id[idx],
            "reward": self.reward[idx],
            "sentence": tokens,
        }
        return batch


@dataclass
class RewardFinetuner:
    """Finetuner class for the reward simulator.

    Imported as: :class:`off_prompts.dataset.assets.reward_finetuner.RewardFinetuner`

    Parameters
    -------
    model: BaseRewardSimulator
        Reward simulation model to finetune.

    optimizer: torch.optim.Optimizer, default=Adam
        Class of optimizer (not an instance).

    optimizer_kwargs: dict, default=None
        Arguments of optimizer.

    tokenizer: PreTrainedTokenizer or PreTrainedTokenizerFast
        Tokenizer of the (frozen) LLM.

    tokenizer_kwargs: dict
        Tokenizer kwargs.

    random_state: int, default=None.
        Random state.

    """

    model: BaseRewardSimulator
    optimizer: Optimizer = Adam
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = (None,)
    tokenizer_kwargs: Optional[Dict[str, Any]] = (None,)
    random_state: Optional[int] = None

    def __post_init__(self):
        self.trained_model = None
        self.device = self.model.device

        if self.random_state is None:
            self.seed(self.random_state)

        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {
                "lr": 1e-4,
            }

        if self.tokenizer is None:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(
                "distilbert-base-uncased",
                truncation=True,
                do_lower_case=True,
            )

        if self.tokenizer_kwargs is None:
            self.tokenizer_kwargs = {
                "add_special_tokens": True,
                "padding": True,
                "truncation": True,
                "max_length": 20,
                "return_tensors": "pt",
            }

    def _random_masking(
        self,
        tokens: Tokens,
        mask_ratio: float = 0.1,
        random_token_ratio: float = 0.1,
    ):
        """Apply random mask and replacement to input tokens.

        Parameters
        -------
        tokens: Tokens
            Original tokens.

        mask_ratio: float, default=0.1 (>= 0)
            Proportion of the input tokens to mask.

        random_token_ratio: float, default=0.1 (>= 0)
            Proportion of the input tokens to replace with a random token.

        Return
        -------
        masked_token: Tokens
            Masked and randomly replaced tokens.

        """
        if mask_ratio + random_token_ratio > 1:
            raise ValueError(
                "the sum of mask_ratio and random_token_ratio must be less than 1, but found False."
            )

        input_ids = tokens["input_ids"]
        n_samples, max_length = input_ids.shape
        total_samples = n_samples * max_length
        device = input_ids.device

        original_ratio = 1 - mask_ratio - random_token_ratio
        mask_weight = torch.tensor([original_ratio, random_token_ratio, mask_ratio])

        mask = (
            torch.multinomial(
                mask_weight,
                num_samples=total_samples,
                replacement=True,
            )
            .reshape((n_samples, max_length))
            .to(device)
        )
        mask[:, 0] = 0
        mask[:, -1] = 0

        random_words = torch.randint(
            len(self.tokenizer), input_ids.shape, dtype=torch.long
        ).to(device)
        input_ids[mask == 1] = random_words[mask == 1]
        input_ids[mask == 2] = self.tokenizer.mask_token_id
        return tokens

    def init_with_reference_model(self, parent_model: BaseRewardSimulator):
        """Initialize user and item id embeddings and biases from a parent model.

        Parameters
        -------
        parent_model: BaseRewardSimulator
            Parent model to inherit the weight.

        """
        parent_params = parent_model.state_dict()

        if isinstance(self.model, TransformerRewardSimulator):
            self.model.user_embedding.weight.data.copy_(
                parent_params["user_embedding.weight"]
            )
        else:
            self.model.user_embedding.weight.data.copy_(
                parent_params["user_embedding.weight"]
            )
            self.model.item_embedding.weight.data.copy_(
                parent_params["item_embedding.weight"]
            )

        self.model.user_bias.weight.data.copy_(parent_params["user_bias.weight"])
        self.model.item_bias.weight.data.copy_(parent_params["item_bias.weight"])
        self.model.to(self.device)

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

    def finetune_model(
        self,
        df: pd.DataFrame,
        user_id_column: str = "user_id",
        item_id_column: str = "item_id",
        sentence_column: str = "description",
        reward_column: str = "reward",
        loss_type: str = "MSE",
        n_epochs: int = 100,
        n_steps_per_epoch: int = 100,
        return_training_process: bool = False,
        batch_size: int = 16,
        mask_ratio: float = 0.1,
        random_token_ratio: float = 0.1,
        val_ratio: float = 0.2,
        shuffle: bool = False,
        make_copy: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
    ):
        """Fine-tune model.

        Parameters
        -------
        df: DataFrame
            Pandas dataframe of the preprocessed movielens dataset.

        user_id_column: str, default="user_id"
            Column name if the user id column.

        item_id_column: str, default="item_id"
            Column name if the item id column.

        sentence_column: str, default="description"
            Column name if the sentence description of the items.

        reward_column: str, default="reward"
            Column name if the reward column.

        loss_type: {"MSE", "BCE"}, default="MSE"
            Whether to use mean-squared error (MSE) loss or binary cross entropy (BCE) loss.

        n_epochs: int, default=100 (> 0)
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=100 (> 0)
            Number of gradient steps within an epoch.

        return_training_process: bool, default=False
            Whether to return the training logs.

        batch_size: int, default=32 (> 0)
            Batch size.

        mask_ratio: float, default=0.1 (>= 0)
            Proportion of the input tokens to mask.

        random_token_ratio: float, default=0.1 (>= 0)
            Proportion of the input tokens to replace with a random token.

        val_ratio: float, default = 0.2 (> 0)
            Proportion of validation samples.

        shuffle: bool, default=False
            Whether to shuffle the input data.

        make_copy: bool, default=False
            Whether to create copy of the model before training.

        save_path: Path, default=None
            Path to save the model.

        random_state: int, default=None
            Random state.

        Return
        -------
        model: BaseRewardSimulator
            Finetuned reward simulator.

        """
        if random_state is None:
            random_state = self.random_state
        self.seed(random_state)

        if make_copy:
            model = deepcopy(self.model)
        else:
            model = self.model

        optimizer = self.optimizer(model.parameters(), **self.optimizer_kwargs)

        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            shuffle=shuffle,
            random_state=random_state,
        )

        train_dataset = MovielensDataset(
            train_df,
            user_id_column=user_id_column,
            item_id_column=item_id_column,
            sentence_column=sentence_column,
            reward_column=reward_column,
            tokenizer=self.tokenizer,
            tokenizer_kwargs=self.tokenizer_kwargs,
            device=self.device,
        )
        val_dataset = MovielensDataset(
            val_df,
            user_id_column=user_id_column,
            item_id_column=item_id_column,
            sentence_column=sentence_column,
            reward_column=reward_column,
            tokenizer=self.tokenizer,
            tokenizer_kwargs=self.tokenizer_kwargs,
            device=self.device,
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
        val_losses = torch.zeros((n_epochs + 1,))

        early_stopping_cnt = 0

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[finetune reward simulator: Epoch {i + 1}]")
                pbar.set_postfix(
                    {
                        "train_loss": f"{train_losses[i]:.4g}",
                        "val_loss": f"{val_losses[i]:.4g}",
                    }
                )

                if early_stopping_cnt >= 5:
                    print(f"apply early stopping at epoch {i + 1}")
                    break

                train_iterator = iter(train_dataloader)

                for j in range(n_steps_per_epoch):
                    batch_ = next(train_iterator)
                    user_id_ = to_device(batch_["user_id"], device=accelerator.device)
                    item_id_ = to_device(batch_["item_id"], device=accelerator.device)
                    reward_ = to_device(batch_["reward"], device=accelerator.device)
                    sentence_ = to_device(batch_["sentence"], device=accelerator.device)

                    masked_sentence_ = self._random_masking(
                        sentence_,
                        mask_ratio=mask_ratio,
                        random_token_ratio=random_token_ratio,
                    )
                    prediction_ = model(
                        user_id=user_id_,
                        item_id=item_id_,
                        item_tokens=masked_sentence_,
                    )
                    train_loss_ = loss_fn(prediction_, reward_)

                    optimizer.zero_grad()
                    accelerator.backward(train_loss_)
                    optimizer.step()

                    train_losses[i + 1] += train_loss_.item() / n_steps_per_epoch

                for val_batch_ in val_dataloader:
                    val_user_id_ = to_device(
                        val_batch_["user_id"], device=accelerator.device
                    )
                    val_item_id_ = to_device(
                        val_batch_["item_id"], device=accelerator.device
                    )
                    val_reward_ = to_device(
                        val_batch_["reward"], device=accelerator.device
                    )
                    val_sentence_ = to_device(
                        val_batch_["sentence"], device=accelerator.device
                    )

                    val_prediction_ = model(
                        user_id=val_user_id_,
                        item_id=val_item_id_,
                        item_tokens=val_sentence_,
                    )

                    ratio_ = len(val_user_id_) / len(val_dataloader.dataset)
                    val_losses[i + 1] += (
                        loss_fn(val_prediction_, val_reward_).item()
                    ) * ratio_

                if val_losses[i + 1] > val_losses[i]:
                    early_stopping_cnt += 1
                else:
                    early_stopping_cnt = 0

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        if return_training_process:
            return model, train_losses, val_losses
        else:
            return model
