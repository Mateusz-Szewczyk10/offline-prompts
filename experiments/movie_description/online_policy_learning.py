"""Scripts for runninf online policy learning."""
from typing import Optional, Dict, Any
from pathlib import Path
from copy import deepcopy
import time
import random

import hydra
from omegaconf import DictConfig
from utils import assert_configuration, format_runtime, fix_seed, get_optimizer

import torch
from torch.optim import Optimizer
from transformers import set_seed

from function import (
    load_dataset,
    load_encoders,
    train_and_save_online_policy,
)


def _process(
    n_actions: int,
    dim_query: int,
    dim_prompt: int,
    dim_sentence: int,
    reward_type: str,
    reward_std: float,
    n_epochs: int,
    n_steps_per_epoch: int,
    n_epochs_per_log: int,
    lr: float,
    optimizer: Optimizer,
    path_to_user_embeddings: str,
    path_to_queries: str,
    path_to_query_embeddings: str,
    path_to_interaction_data: str,
    path_to_candidate_prompts: str,
    path_to_prompt_embeddings: str,
    path_to_finetuned_params: str,
    path_to_query_pca_matrix: str,
    path_to_prompt_pca_matrix: str,
    path_to_sentence_pca_matrix: str,
    reward_simulator_type: str,
    device: str = "cuda",
    base_random_state: Optional[int] = None,
    use_wandb: bool = False,
    **kwargs,
):
    fix_seed(base_random_state)

    dataset = load_dataset(
        n_actions=n_actions,
        reward_type=reward_type,
        reward_std=reward_std,
        path_to_user_embeddings=path_to_user_embeddings,
        path_to_queries=path_to_queries,
        path_to_query_embeddings=path_to_query_embeddings,
        path_to_interaction_data=path_to_interaction_data,
        path_to_candidate_prompts=path_to_candidate_prompts,
        path_to_prompt_embeddings=path_to_prompt_embeddings,
        path_to_finetuned_params=path_to_finetuned_params,
        reward_simulator_type=reward_simulator_type,
        dim_sentence=dim_sentence,
        save_path_sentence_encoder=Path(path_to_sentence_pca_matrix),
        device=device,
        random_state=base_random_state,
    )

    query_encoder, prompt_encoder, sentence_encoder = load_encoders(
        dataset=dataset,
        dim_query=dim_query,
        dim_prompt=dim_prompt,
        dim_sentence=dim_sentence,
        save_path_query_encoder=Path(path_to_query_pca_matrix),
        save_path_prompt_encoder=Path(path_to_prompt_pca_matrix),
        save_path_sentence_encoder=Path(path_to_sentence_pca_matrix),
        device=device,
        random_state=base_random_state,
    )

    Path(f"logs/policy_online/").mkdir(parents=True, exist_ok=True)
    Path(f"logs/learning_process/policy_online/").mkdir(parents=True, exist_ok=True)

    train_and_save_online_policy(
        dataset=dataset,
        query_encoder=query_encoder,
        prompt_encoder=prompt_encoder,
        sentence_encoder=sentence_encoder,
        n_epochs=n_epochs,
        n_steps_per_epoch=n_steps_per_epoch,
        n_epochs_per_log=n_epochs_per_log,
        lr=lr,
        optimizer=optimizer,
        save_path=Path(
            f"logs/policy_online/"
            f"{reward_simulator_type}_{n_actions}_{dim_query}_{dim_prompt}_{dim_sentence}_{reward_type}_{reward_std}_{device}_{base_random_state}",
        ),
        save_path_logs=Path(
            f"logs/learning_process/policy_online/"
            f"{reward_simulator_type}_{n_actions}_{dim_query}_{dim_prompt}_{dim_sentence}_{reward_type}_{reward_std}_{device}_{base_random_state}",
        ),
        device=device,
        random_state=base_random_state,
        use_wandb=use_wandb,
    )


def process(
    conf: Dict[str, Any],
):
    _process(**conf)


@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    assert_configuration(cfg)
    print(f"The current working directory is {Path().cwd()}")
    print(f"The original working directory is {hydra.utils.get_original_cwd()}")
    print()

    conf = {
        "setting": cfg.setting.setting,
        "n_samples": cfg.setting.n_samples,
        "n_actions": cfg.setting.n_actions,
        "beta": cfg.setting.beta,
        "dim_query": cfg.setting.dim_query,
        "dim_prompt": cfg.setting.dim_prompt,
        "dim_sentence": cfg.setting.dim_sentence,
        "reward_type": cfg.setting.reward_type,
        "reward_std": cfg.setting.reward_std,
        "n_epochs": cfg.online.n_epochs,
        "n_steps_per_epoch": cfg.online.n_steps_per_epoch,
        "n_epochs_per_log": cfg.online.n_epochs_per_log,
        "lr": cfg.online.lr,
        "optimizer": get_optimizer(cfg.online.optimizer),
        "device": cfg.setting.device,
        "path_to_user_embeddings": cfg.setting.path_to_user_embeddings,
        "path_to_queries": cfg.setting.path_to_queries,
        "path_to_query_embeddings": cfg.setting.path_to_query_embeddings,
        "path_to_interaction_data": cfg.setting.path_to_interaction_data,
        "path_to_candidate_prompts": cfg.setting.path_to_candidate_prompts,
        "path_to_prompt_embeddings": cfg.setting.path_to_prompt_embeddings,
        "path_to_finetuned_params": cfg.setting.path_to_finetuned_params,
        "path_to_query_pca_matrix": cfg.setting.path_to_query_pca_matrix,
        "path_to_prompt_pca_matrix": cfg.setting.path_to_prompt_pca_matrix,
        "path_to_sentence_pca_matrix": cfg.setting.path_to_sentence_pca_matrix,
        "reward_simulator_type": cfg.setting.reward_simulator_type,
        "base_random_state": cfg.setting.base_random_state,
        "use_wandb": cfg.setting.use_wandb,
    }
    process(conf)


if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()
    print("total runtime:", format_runtime(start, finish))
