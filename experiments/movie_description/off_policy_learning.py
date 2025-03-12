"""Scripts for running OPL methods."""
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
    load_online_policy,
    load_logging_policy,
    load_logged_data,
    load_reward_predictor,
    load_kernel_marginal_estimator,
    generate_and_save_logged_data,
    train_and_save_single_stage_policy,
    train_and_save_two_stage_policy,
    train_and_save_dso_policy,
)


def _process(
    n_samples: int,
    n_actions: int,
    beta: float,
    dim_query: int,
    dim_prompt: int,
    dim_sentence: int,
    reward_type: str,
    reward_std: float,
    gradient_type: str,
    kernel_type: str,
    n_epochs: int,
    n_steps_per_epoch: int,
    n_epochs_per_log: int,
    lr: float,
    optimizer: Optimizer,
    tau: float,
    n_clusters: int,
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
    is_two_stage_policy: bool = False,
    is_dso: bool = False,
    device: str = "cuda",
    base_random_state: Optional[int] = None,
    dataset_random_state: Optional[int] = None,
    optimizer_random_state: Optional[int] = None,
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

    base_policy = load_online_policy(
        dataset=dataset,
        query_encoder=query_encoder,
        prompt_encoder=prompt_encoder,
        sentence_encoder=sentence_encoder,
        save_path=Path(
            f"logs/policy_online/"
            f"{reward_simulator_type}_{n_actions}_{dim_query}_{dim_prompt}_{dim_sentence}_{reward_type}_{reward_std}_{device}_{base_random_state}",
        ),
        device=device,
        random_state=base_random_state,
    )

    logging_policy = load_logging_policy(
        dataset=dataset,
        base_policy=base_policy,
        beta=beta,
        device=device,
        random_state=base_random_state,
    )

    fix_seed(dataset_random_state)
    logged_feedback = generate_and_save_logged_data(
        dataset=dataset,
        logging_policy=logging_policy,
        n_samples=n_samples,
        save_path=Path(
            f"logs/logged_feedback/"
            f"{reward_simulator_type}_{n_actions}_{dim_query}_{dim_prompt}_{dim_sentence}_{reward_type}_{reward_std}_{device}_{n_samples}_{beta}_{base_random_state}_{dataset_random_state}.pkl"
        ),
    )

    if gradient_type in ["regression-based", "hybrid"]:
        prompt_reward_predictor = load_reward_predictor(
            dataset=dataset,
            query_encoder=query_encoder,
            prompt_encoder=prompt_encoder,
            save_path=Path(
                f"logs/prompt_reward_predictor/"
                f"{reward_simulator_type}_{n_actions}_{dim_query}_{dim_prompt}_{dim_sentence}_{reward_type}_{reward_std}_{n_samples}_{beta}_{device}_{base_random_state}_{dataset_random_state}",
            ),
            device=device,
            random_state=dataset_random_state,
        )
    else:
        prompt_reward_predictor = None

    if not (is_two_stage_policy or is_dso):
        Path(
            f"logs/policy_single_{gradient_type}/"
        ).mkdir(
            parents=True, exist_ok=True,
        )
        Path(
            f"logs/learning_process/policy_single_{gradient_type}/"
        ).mkdir(
            parents=True, exist_ok=True,
        )
        fix_seed(optimizer_random_state)

        train_and_save_single_stage_policy(
            dataset=dataset,
            logged_feedback=logged_feedback,
            prompt_reward_predictor=prompt_reward_predictor,
            query_encoder=query_encoder,
            sentence_encoder=sentence_encoder,
            gradient_type=gradient_type,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            n_epochs_per_log=n_epochs_per_log,
            lr=lr,
            optimizer=optimizer,
            save_path=Path(
                f"logs/policy_single_{gradient_type}/"
                f"{reward_simulator_type}_{n_actions}_{dim_query}_{dim_prompt}_{dim_sentence}_{reward_type}_{reward_std}_{n_samples}_{beta}_{device}_{base_random_state}_{dataset_random_state}_{optimizer_random_state}",
            ),
            save_path_logs=Path(
                f"logs/learning_process/policy_single_{gradient_type}/"
                f"{reward_simulator_type}_{n_actions}_{dim_query}_{dim_prompt}_{dim_sentence}_{reward_type}_{reward_std}_{n_samples}_{beta}_{device}_{base_random_state}_{dataset_random_state}_{optimizer_random_state}",
            ),
            device=device,
            random_state=optimizer_random_state,
            base_random_state=dataset_random_state,
            use_wandb=use_wandb,
        )

    if is_two_stage_policy:
        Path(
            f"logs/policy_two_stage/"
        ).mkdir(
            parents=True, exist_ok=True,
        )
        Path(
            f"logs/learning_process/policy_two_stage/"
        ).mkdir(
            parents=True, exist_ok=True,
        )
        fix_seed(optimizer_random_state)

        train_and_save_two_stage_policy(
            dataset=dataset,
            logged_feedback=logged_feedback,
            prompt_reward_predictor=prompt_reward_predictor,
            query_encoder=query_encoder,
            prompt_encoder=prompt_encoder,
            sentence_encoder=sentence_encoder,
            gradient_type=gradient_type,
            n_clusters=n_clusters,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            n_epochs_per_log=n_epochs_per_log,
            lr=lr,
            optimizer=optimizer,
            save_path=Path(
                f"logs/policy_two_stage/"
                f"{reward_simulator_type}_{n_actions}_{dim_query}_{dim_prompt}_{dim_sentence}_{reward_type}_{reward_std}_{n_samples}_{beta}_{device}_{base_random_state}_{dataset_random_state}_{optimizer_random_state}",
            ),
            save_path_logs=Path(
                f"logs/learning_process/policy_two_stage/"
                f"{reward_simulator_type}_{n_actions}_{dim_query}_{dim_prompt}_{dim_sentence}_{reward_type}_{reward_std}_{n_samples}_{beta}_{device}_{base_random_state}_{dataset_random_state}_{optimizer_random_state}",
            ),
            device=device,
            random_state=optimizer_random_state,
            base_random_state=dataset_random_state,
            use_wandb=use_wandb,
        )

    elif is_dso:
        if kernel_type == "uniform":
            tau = tau * 3.0

        kernel_marginal_estimator = load_kernel_marginal_estimator(
            dataset=dataset,
            query_encoder=query_encoder,
            sentence_encoder=sentence_encoder,
            kernel_type=kernel_type,
            tau=tau,
            save_path=Path(
                f"logs/kernel_marginal_estimator_{kernel_type}/"
                f"{reward_simulator_type}_{n_actions}_{dim_query}_{dim_prompt}_{dim_sentence}_{reward_type}_{reward_std}_{n_samples}_{beta}_{tau}_{device}_{base_random_state}_{dataset_random_state}",
            ),
            device=device,
            random_state=dataset_random_state,
        )

        Path(
            f"logs/policy_dso_{kernel_type}/"
        ).mkdir(
            parents=True, exist_ok=True,
        )
        Path(
            f"logs/learning_process/policy_dso_{kernel_type}/"
        ).mkdir(
            parents=True, exist_ok=True,
        )
        fix_seed(optimizer_random_state)

        train_and_save_dso_policy(
            dataset=dataset,
            logged_feedback=logged_feedback,
            kernel_marginal_estimator=kernel_marginal_estimator,
            query_encoder=query_encoder,
            sentence_encoder=sentence_encoder,
            gradient_type=gradient_type,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            n_epochs_per_log=n_epochs_per_log,
            lr=lr,
            optimizer=optimizer,
            save_path=Path(
                f"logs/policy_dso_{kernel_type}/"
                f"{reward_simulator_type}_{n_actions}_{dim_query}_{dim_prompt}_{dim_sentence}_{reward_type}_{reward_std}_{n_samples}_{beta}_{tau}_{device}_{base_random_state}_{dataset_random_state}_{optimizer_random_state}",
            ),
            save_path_logs=Path(
                f"logs/learning_process/policy_dso_{kernel_type}/"
                f"{reward_simulator_type}_{n_actions}_{dim_query}_{dim_prompt}_{dim_sentence}_{reward_type}_{reward_std}_{n_samples}_{beta}_{tau}_{device}_{base_random_state}_{dataset_random_state}_{optimizer_random_state}",
            ),
            device=device,
            random_state=optimizer_random_state,
            base_random_state=dataset_random_state,
            use_wandb=use_wandb,
        )


def process(conf: Dict[str, Any],):
    dataset_n_random_state = conf["dataset_n_random_state"]
    dataset_start_random_state = conf["dataset_start_random_state"]
    optimizer_n_random_state = conf["optimizer_n_random_state"]
    optimizer_start_random_state = conf["optimizer_start_random_state"]
    conf_ = deepcopy(conf)

    for dataset_random_state in range(dataset_start_random_state, dataset_n_random_state):
        for optimizer_random_state in range(optimizer_start_random_state, optimizer_n_random_state):
            conf_["dataset_random_state"] = dataset_random_state
            conf_["optimizer_random_state"] = optimizer_random_state
            _process(**conf_)


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
        "n_epochs": cfg.offline.n_epochs,
        "n_steps_per_epoch": cfg.offline.n_steps_per_epoch,
        "n_epochs_per_log": cfg.offline.n_epochs_per_log,
        "lr": cfg.offline.lr,
        "optimizer": get_optimizer(cfg.offline.optimizer),
        "gradient_type": cfg.offline.gradient_type,
        "is_two_stage_policy": cfg.offline.is_two_stage_policy,
        "is_dso": cfg.offline.is_dso,
        "kernel_type": cfg.offline.kernel_type,
        "tau": cfg.offline.tau,
        "n_clusters": cfg.offline.n_clusters,
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
        "dataset_n_random_state": cfg.offline.dataset_n_random_state,
        "dataset_start_random_state": cfg.offline.dataset_start_random_state,
        "optimizer_n_random_state": cfg.offline.optimizer_n_random_state,
        "optimizer_start_random_state": cfg.offline.optimizer_start_random_state,
        "use_wandb": cfg.setting.use_wandb,
    }
    process(conf)


if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()
    print("total runtime:", format_runtime(start, finish))
