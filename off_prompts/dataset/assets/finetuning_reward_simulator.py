"""Scripts for finetuning the reward simulator."""
from typing import Optional
from pathlib import Path
import random
import time
import gc

import hydra
from omegaconf import DictConfig

import pandas as pd

import torch
from torch.optim import Adam

from transformers import (
    AutoModel,
    AutoTokenizer,
)
from transformers import set_seed

from off_prompts.dataset import CollaborativeFilteringRewardSimulator as CFRewardSimulator
from off_prompts.dataset import TransformerRewardSimulator

from reward_finetuner import RewardFinetuner

from utils import format_runtime


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

    # reward simulator
    naive_cf_reward_simulator = CFRewardSimulator(
        n_users=n_users,
        n_items=n_items,
        dim_emb=dim_emb,
        device=device,
        random_state=random_state,
    )
    transformer_reward_simulator = TransformerRewardSimulator(
        n_users=n_users,
        n_items=n_items,
        dim_emb=dim_emb,
        base_model=base_model,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
    )

    # reward finetuner
    print("fitting naive collaborative filtering..")
    naive_cf_finetuner = RewardFinetuner(
        model=naive_cf_reward_simulator,
        optimizer=Adam,
        optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.0},
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        random_state=random_state,
    )
    naive_cf_reward_simulator = naive_cf_finetuner.finetune_model(
        df=df,
        loss_type="MSE",
        n_epochs=300,
        save_path=f"{setting}_naive_cf_params.pt",
        random_state=random_state,
    )
    # initialize user embedding with naive cf and finetune the model
    print("fitting transformer-based collaborative filtering..")
    transformer_finetuner = RewardFinetuner(
        model=transformer_reward_simulator,
        optimizer=Adam,
        optimizer_kwargs={"lr": 1e-3, "weight_decay": 0.0},
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        random_state=random_state,
    )

    transformer_finetuner.init_with_reference_model(
        parent_model=naive_cf_reward_simulator,
    )
    del naive_cf_reward_simulator
    del naive_cf_finetuner
    gc.collect()

    transformer_reward_simulator = transformer_finetuner.finetune_model(
        df=df,
        loss_type="MSE",
        n_epochs=300,
        save_path=f"{setting}_{model_name}_reward_simulator.pt",
        random_state=random_state,
    )


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
