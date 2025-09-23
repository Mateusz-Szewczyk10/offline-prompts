"""Scripts for preprocessing user and query information."""
from typing import Optional
from pathlib import Path
import random
import time

import hydra
from omegaconf import DictConfig

import pandas as pd

import torch
from torch.optim import Adam
from transformers import AutoModel, AutoTokenizer
from transformers import set_seed

from off_prompts.dataset import (
    CollaborativeFilteringRewardSimulator as CFRewardSimulator,
)
from off_prompts.dataset import TransformerRewardSimulator
from off_prompts.dataset import RewardFinetuner

from utils import format_runtime


def process(
    setting: str,
    dataset_path: str,
    model_path: str,
    base_model_id: str,
    tokenizer_id: str,
    dim_emb: int,
    use_naive_cf_user_embs: bool = False,
    use_tokenizer_fast: bool = False,
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

    df = pd.read_csv(dataset_path)
    n_users = df["user_id"].nunique()
    n_items = df["item_id"].nunique()

    emb_name = "naive_cf" if use_naive_cf_user_embs else "transformer"

    if use_naive_cf_user_embs:
        tokenizer = None
        tokenizer_kwargs = None

        model = CFRewardSimulator(
            n_users=n_users,
            n_items=n_items,
            dim_emb=dim_emb,
            device=device,
            random_state=random_state,
        )

    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id, use_fast=use_tokenizer_fast
        )
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        base_model = AutoModel.from_pretrained(base_model_id)
        base_model.resize_token_embeddings(len(tokenizer))
        base_model.to(device)

        tokenizer_kwargs = {
            "add_special_tokens": True,
            "padding": True,
            "truncation": True,
            "max_length": 20,
            "return_tensors": "pt",
        }

        model = TransformerRewardSimulator(
            n_users=n_users,
            n_items=n_items,
            dim_emb=dim_emb,
            base_model=base_model,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
        )

    model_loader = RewardFinetuner(
        model=model,
        optimizer=Adam,
        optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.0},
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        random_state=random_state,
    )
    model = model_loader.load(model_path, is_init=True)

    # save user embeddings
    with torch.no_grad():
        user_ids = torch.arange(n_users).to(device)
        user_embs = model.user_embedding(user_ids).to("cpu").detach()
        torch.save(user_embs, f"{setting}_{emb_name}_user_embeddings.pt")

    # save query infos
    unique_items = df.drop_duplicates(subset="item_id")
    unique_items = unique_items.sort_values(by="item_id")
    queries = unique_items["title"]

    df = pd.DataFrame()
    df["query"] = queries
    df.to_csv(f"{setting}_query.csv", index=False)


@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    print(f"The current working directory is {Path().cwd()}")
    print(f"The original working directory is {hydra.utils.get_original_cwd()}")
    print()

    process(
        setting=cfg.user_query_simulation.setting,
        dataset_path=cfg.user_query_simulation.dataset_path,
        model_path=cfg.user_query_simulation.model_path,
        base_model_id=cfg.user_query_simulation.base_model_id,
        tokenizer_id=cfg.user_query_simulation.tokenizer_id,
        dim_emb=cfg.user_query_simulation.dim_emb,
        use_naive_cf_user_embs=cfg.user_query_simulation.use_naive_cf_user_embs,
        use_tokenizer_fast=cfg.user_query_simulation.use_tokenizer_fast,
        random_state=cfg.user_query_simulation.random_state,
    )


if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()
    print("total runtime:", format_runtime(start, finish))
