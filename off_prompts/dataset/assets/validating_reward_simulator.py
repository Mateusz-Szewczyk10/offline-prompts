"""Scripts for finetuning the reward simulator."""
from typing import Optional, Union, Dict, Any
from pathlib import Path
import random
import time
import gc

import hydra
from omegaconf import DictConfig

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

import torch
from torch.optim import Adam

from transformers import (
    AutoModel,
    AutoTokenizer,
)
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    DistilBertTokenizerFast,
)
from transformers import set_seed

from off_prompts.dataset import (
    CollaborativeFilteringRewardSimulator as CFRewardSimulator,
)
from off_prompts.dataset import TransformerRewardSimulator
from off_prompts.utils import torch_seed, tokenize, to_device

from reward_finetuner import RewardFinetuner

from utils import format_runtime


class MovielensDataset(Dataset):
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


def process(
    setting: str,
    model_name: str,
    dataset_path: str,
    base_model_id: str,
    tokenizer_id: str,
    use_tokenizer_fast: bool = False,
    dim_emb: int = 20,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    if random_state is not None:
        random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed(random_state)
        set_seed(random_state)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # load dataset
    print("loading dataset..")
    df = pd.read_csv(dataset_path)
    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()

    # transformer setting
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=use_tokenizer_fast)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    base_model = AutoModel.from_pretrained(base_model_id)
    base_model.resize_token_embeddings(len(tokenizer))

    tokenizer_kwargs = {
        "add_special_tokens": True,
        "padding": True,
        "truncation": True,
        "max_length": 20,
        "return_tensors": "pt",
    }

    transformer_reward_simulator = TransformerRewardSimulator(
        n_users=n_users,
        n_items=n_items,
        dim_emb=dim_emb,
        base_model=base_model,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
    )
    reward_finetuner = RewardFinetuner(
        model=transformer_reward_simulator,
        optimizer=Adam,
        optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.0},
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        random_state=random_state,
    )

    # validation visualization
    # data split is the same as the fine-tuning
    transformer_reward_simulator = reward_finetuner.load(
        path=f"{setting}_{model_name}_reward_simulator.pt",
        is_init=True,
    )
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        shuffle=False,
        random_state=random_state,
    )
    val_df = val_df[:2000]
    val_dataset = MovielensDataset(
        val_df,
        user_id_column="user_id",
        item_id_column="item_id",
        sentence_column="description",
        reward_column="reward",
        tokenizer=reward_finetuner.tokenizer,
        tokenizer_kwargs=reward_finetuner.tokenizer_kwargs,
        device=reward_finetuner.device,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=2000,
        shuffle=False,
    )
    for val_batch_ in val_dataloader:
        val_user_id_ = to_device(val_batch_["user_id"], device=reward_finetuner.device)
        val_item_id_ = to_device(val_batch_["item_id"], device=reward_finetuner.device)
        val_reward_ = to_device(val_batch_["reward"], device=reward_finetuner.device)
        val_sentence_ = to_device(
            val_batch_["sentence"], device=reward_finetuner.device
        )

        val_prediction_ = (
            transformer_reward_simulator(
                user_id=val_user_id_,
                item_id=val_item_id_,
                item_tokens=val_sentence_,
            )
            .cpu()
            .detach()
        )

    pos_id = torch.where(torch.tensor((val_df["reward"] > 0).values))
    neg_id = torch.where(torch.tensor((val_df["reward"] <= 0).values))
    bins = torch.linspace(0, 1, steps=41)

    plt.style.use("ggplot")
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.0), sharey=True)
    ax.hist(
        val_prediction_[pos_id],
        bins=bins,
        alpha=0.5,
        label="positive",
    )
    ax.hist(
        val_prediction_[neg_id],
        bins=bins,
        alpha=0.5,
        label="negative",
    )
    ax.set_title("reward simulation result")
    ax.set_ylabel("count")
    ax.set_xlabel("simulated reward")
    ax.legend(loc="center right")
    plt.savefig("reward_distribution.png", dpi=300, bbox_inches="tight")


@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    print(f"The current working directory is {Path().cwd()}")
    print(f"The original working directory is {hydra.utils.get_original_cwd()}")
    print()

    process(
        setting=cfg.reward_model.setting,
        model_name=cfg.reward_model.model_name,
        dataset_path=cfg.reward_model.dataset_path,
        base_model_id=cfg.reward_model.base_model_id,
        tokenizer_id=cfg.reward_model.tokenizer_id,
        use_tokenizer_fast=cfg.reward_model.use_tokenizer_fast,
        dim_emb=cfg.reward_model.dim_emb,
        device=cfg.reward_model.device,
        random_state=cfg.setting.random_state,
    )


if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()
    print("total runtime:", format_runtime(start, finish))
