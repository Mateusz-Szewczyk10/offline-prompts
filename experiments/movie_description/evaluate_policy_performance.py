"""Script for evaluating the policy performance."""
from typing import Optional
from pathlib import Path
from copy import deepcopy
import time
import random

import torch
import numpy as np
import pandas as pd
from transformers import set_seed

import hydra
from omegaconf import DictConfig
from utils import (
    assert_configuration,
    assert_configuration_for_evaluation,
    format_runtime,
    fix_seed,
)

from function import (
    load_dataset,
    load_encoders,
    load_logging_policy,
    load_uniform_policy,
    load_regression_greedy_policy,
    load_online_policy,
    load_single_stage_policy,
    load_two_stage_policy,
    load_dso_policy,
    load_reward_predictor,
    load_kernel_marginal_estimator,
    evaluate_policy,
    get_baseline_performance,
    get_skyline_performance,
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
    is_dso: bool = False,
    is_two_stage_policy: bool = False,
    is_logging_policy: bool = False,
    is_online_policy: bool = False,
    is_uniform_policy: bool = False,
    is_regression_based_greedy: bool = False,
    is_no_prompt_baseline: bool = False,
    is_skyline: bool = False,
    device: str = "cuda",
    base_random_state: Optional[int] = None,
    dataset_random_state: Optional[int] = None,
    optimizer_random_state: Optional[int] = None,
    eval_random_state: Optional[int] = None,
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

    if is_logging_policy:
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

        fix_seed(eval_random_state)
        policy_value = evaluate_policy(dataset=dataset, policy=logging_policy,)

    elif is_uniform_policy:
        uniform_policy = load_uniform_policy(
            dataset=dataset, device=device, random_state=base_random_state,
        )

        fix_seed(eval_random_state)
        policy_value = evaluate_policy(dataset=dataset, policy=uniform_policy,)

    elif is_regression_based_greedy:
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
        regression_based_greedy_policy = load_regression_greedy_policy(
            dataset=dataset,
            prompt_reward_predictor=prompt_reward_predictor,
            device=device,
            random_state=dataset_random_state,
        )

        fix_seed(eval_random_state)
        policy_value = evaluate_policy(
            dataset=dataset, policy=regression_based_greedy_policy,
        )
    
    elif is_no_prompt_baseline:
        fix_seed(eval_random_state)
        policy_value = get_baseline_performance(
            dataset=dataset,
        )

    elif is_skyline:
        fix_seed(eval_random_state)
        policy_value = get_skyline_performance(
            dataset=dataset,
        )

    elif is_online_policy:
        policy = load_online_policy(
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

        fix_seed(eval_random_state)
        policy_value = evaluate_policy(dataset=dataset, policy=policy,)

    elif not (is_two_stage_policy or is_dso):
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

        policy = load_single_stage_policy(
            dataset=dataset,
            prompt_reward_predictor=prompt_reward_predictor,
            query_encoder=query_encoder,
            sentence_encoder=sentence_encoder,
            save_path=Path(
                f"logs/policy_single_{gradient_type}/"
                f"{reward_simulator_type}_{n_actions}_{dim_query}_{dim_prompt}_{dim_sentence}_{reward_type}_{reward_std}_{n_samples}_{beta}_{device}_{base_random_state}_{dataset_random_state}_{optimizer_random_state}",
            ),
            device=device,
            random_state=optimizer_random_state,
        )

        fix_seed(eval_random_state)
        policy_value = evaluate_policy(dataset=dataset, policy=policy,)

    elif is_two_stage_policy:
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

        policy = load_two_stage_policy(
            dataset=dataset,
            prompt_reward_predictor=prompt_reward_predictor,
            query_encoder=query_encoder,
            prompt_encoder=prompt_encoder,
            sentence_encoder=sentence_encoder,
            n_clusters=n_clusters,
            save_path=Path(
                f"logs/policy_two_stage/"
                f"{reward_simulator_type}_{n_actions}_{dim_query}_{dim_prompt}_{dim_sentence}_{reward_type}_{reward_std}_{n_samples}_{beta}_{device}_{base_random_state}_{dataset_random_state}_{optimizer_random_state}",
            ),
            device=device,
            random_state=optimizer_random_state,
        )

        fix_seed(eval_random_state)
        policy_value = evaluate_policy(dataset=dataset, policy=policy,)

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
        policy = load_dso_policy(
            dataset=dataset,
            kernel_marginal_estimator=kernel_marginal_estimator,
            query_encoder=query_encoder,
            sentence_encoder=sentence_encoder,
            save_path=Path(
                f"logs/policy_dso_{kernel_type}/"
                f"{reward_simulator_type}_{n_actions}_{dim_query}_{dim_prompt}_{dim_sentence}_{reward_type}_{reward_std}_{n_samples}_{beta}_{tau}_{device}_{base_random_state}_{dataset_random_state}_{optimizer_random_state}",
            ),
            device=device,
            random_state=optimizer_random_state,
        )

        fix_seed(eval_random_state)
        policy_value = evaluate_policy(dataset=dataset, policy=policy,)

    return policy_value


def process(conf: DictConfig):
    evaluation_setting = conf["evaluation_setting"]
    reward_simulator_type = conf["reward_simulator_type"]
    kernel_type = conf["kernel_type"]

    base_random_state = conf["base_random_state"]

    dataset_start_random_state = conf["dataset_start_random_state"]
    dataset_n_random_state = conf["dataset_n_random_state"]

    optimizer_start_random_state = conf["optimizer_start_random_state"]
    optimizer_n_random_state = conf["optimizer_n_random_state"]

    conf_ = deepcopy(conf)
    conf_["gradient_type"] = None

    df = pd.DataFrame()

    if conf["report_logging_policy"]:
        print("evaluating logging policy..")
        logging_performance = []

        for dataset_random_state in range(dataset_start_random_state, dataset_n_random_state):
            for optimizer_random_state in range(optimizer_start_random_state, optimizer_n_random_state):
                conf_["dataset_random_state"] = dataset_random_state
                conf_["optimizer_random_state"] = optimizer_random_state
                performance_ = _process(**conf_, is_logging_policy=True,)
                logging_performance.append(performance_)

        df["logging"] = logging_performance

    if conf["report_uniform_policy"]:
        print("evaluating uniform policy..")
        uniform_performance = []

        for dataset_random_state in range(dataset_start_random_state, dataset_n_random_state):
            for optimizer_random_state in range(optimizer_start_random_state, optimizer_n_random_state):
                conf_["dataset_random_state"] = dataset_random_state
                conf_["optimizer_random_state"] = optimizer_random_state
                performance_ = _process(**conf_, is_uniform_policy=True,)
                uniform_performance.append(performance_)

        df["uniform"] = uniform_performance

    if conf["report_no_prompt_baseline"]:
        print("evaluating no-prompt baseline..")
        no_prompt_performance = []

        for dataset_random_state in range(dataset_start_random_state, dataset_n_random_state):
            for optimizer_random_state in range(optimizer_start_random_state, optimizer_n_random_state):
                conf_["dataset_random_state"] = dataset_random_state
                conf_["optimizer_random_state"] = optimizer_random_state
                performance_ = _process(**conf_, is_no_prompt_baseline=True,)
                no_prompt_performance.append(performance_)

        df["no-prompt"] = no_prompt_performance

    if conf["report_skyline"]:
        print("evaluating skyline..")
        skyline_performance = []

        for dataset_random_state in range(dataset_start_random_state, dataset_n_random_state):
            for optimizer_random_state in range(optimizer_start_random_state, optimizer_n_random_state):
                conf_["dataset_random_state"] = dataset_random_state
                conf_["optimizer_random_state"] = optimizer_random_state
                performance_ = _process(**conf_, is_skyline=True,)
                skyline_performance.append(performance_)

        df["skyline"] = skyline_performance

    if conf["report_online_policy"]:
        print("evaluating online policy..")
        online_performance = []

        for dataset_random_state in range(dataset_start_random_state, dataset_n_random_state):
            for optimizer_random_state in range(optimizer_start_random_state, optimizer_n_random_state):
                conf_["dataset_random_state"] = dataset_random_state
                conf_["optimizer_random_state"] = optimizer_random_state
                performance_ = _process(**conf_, is_skyline=True,)
                performance_ = _process(**conf_, is_online_policy=True,)
                online_performance.append(performance_)

        df["online"] = online_performance

    if conf["report_regression_based_greedy"]:
        print("evaluating regression-based greedy policy..")
        greedy_performance = []

        for dataset_random_state in range(dataset_start_random_state, dataset_n_random_state):
            for optimizer_random_state in range(optimizer_start_random_state, optimizer_n_random_state):
                conf_["dataset_random_state"] = dataset_random_state
                conf_["optimizer_random_state"] = optimizer_random_state
                performance_ = _process(**conf_, is_skyline=True,)
                greedy_performance.append(performance_)

        df["greedy"] = greedy_performance

    if conf["single_stage_pg"] is not None:
        for gradient_type in conf["single_stage_pg"]:
            print(f"evaluating single stage policy gradient: {gradient_type}..")
            conf_["is_two_stage_policy"] = False
            conf_["is_dso"] = False
            conf_["gradient_type"] = gradient_type
            performance = []

            for dataset_random_state in range(dataset_start_random_state, dataset_n_random_state):
                for optimizer_random_state in range(optimizer_start_random_state, optimizer_n_random_state):
                    conf_["dataset_random_state"] = dataset_random_state
                    conf_["optimizer_random_state"] = optimizer_random_state
                    performance.append(_process(**conf_,))

            df[f"{gradient_type} (single)"] = performance

    if conf["dso_pg"] is not None:
        print("evaluating direct sentence policy gradient..")
        conf_["is_two_stage_policy"] = False
        conf_["is_dso"] = True
        conf_["gradient_type"] = gradient_type
        performance = []

        for dataset_random_state in range(dataset_start_random_state, dataset_n_random_state):
            for optimizer_random_state in range(optimizer_start_random_state, optimizer_n_random_state):
                conf_["dataset_random_state"] = dataset_random_state
                conf_["optimizer_random_state"] = optimizer_random_state
                performance.append(_process(**conf_,))

        df["DSO"] = performance

    if conf["two_stage_pg"] is not None:
        print("evaluating two stage policy gradient..")
        conf_["is_two_stage_policy"] = True
        conf_["is_dso"] = False
        conf_["gradient_type"] = gradient_type
        performance = []

        for dataset_random_state in range(dataset_start_random_state, dataset_n_random_state):
            for optimizer_random_state in range(optimizer_start_random_state, optimizer_n_random_state):
                conf_["dataset_random_state"] = dataset_random_state
                conf_["optimizer_random_state"] = optimizer_random_state
                performance.append(_process(**conf_,))
        
        df["POTEC"] = performance

    df["dataset_random_state"] = np.repeat(
        np.arange(dataset_start_random_state, dataset_n_random_state), 
        optimizer_n_random_state - optimizer_start_random_state,
    )
    df["optimizer_random_state"] = np.tile(
        np.arange(optimizer_start_random_state, optimizer_n_random_state),
        dataset_n_random_state - dataset_start_random_state,
    )
    df.to_csv(
        f"logs/{evaluation_setting}_{reward_simulator_type}_{kernel_type}_{dataset_start_random_state}_{dataset_n_random_state}_{optimizer_start_random_state}_{optimizer_n_random_state}.csv",
        index=False,
    )


@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    assert_configuration(cfg)
    assert_configuration_for_evaluation(cfg)
    print(f"The current working directory is {Path().cwd()}")
    print(f"The original working directory is {hydra.utils.get_original_cwd()}")
    print()

    conf = {
        "setting": cfg.setting.setting,
        "evaluation_setting": cfg.evaluation.setting,
        "n_samples": cfg.setting.n_samples,
        "n_actions": cfg.setting.n_actions,
        "beta": cfg.setting.beta,
        "dim_query": cfg.setting.dim_query,
        "dim_prompt": cfg.setting.dim_prompt,
        "dim_sentence": cfg.setting.dim_sentence,
        "reward_type": cfg.setting.reward_type,
        "reward_std": cfg.setting.reward_std,
        "kernel_type": cfg.offline.kernel_type,
        "tau": cfg.offline.tau,
        "n_clusters": cfg.offline.n_clusters,
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
        "device": cfg.setting.device,
        "base_random_state": cfg.setting.base_random_state,
        "dataset_n_random_state": cfg.offline.dataset_n_random_state,
        "dataset_start_random_state": cfg.offline.dataset_start_random_state,
        "optimizer_n_random_state": cfg.offline.optimizer_n_random_state,
        "optimizer_start_random_state": cfg.offline.optimizer_start_random_state,
        "eval_random_state": cfg.evaluation.eval_random_state,
        "report_logging_policy": cfg.evaluation.report_logging_policy,
        "report_online_policy": cfg.evaluation.report_online_policy,
        "report_uniform_policy": cfg.evaluation.report_uniform_policy,
        "report_regression_based_greedy": cfg.evaluation.report_regression_based_greedy,
        "report_no_prompt_baseline": cfg.evaluation.report_no_prompt_baseline,
        "report_skyline": cfg.evaluation.report_skyline,
        "single_stage_pg": cfg.evaluation.single_stage_pg,
        "two_stage_pg": cfg.evaluation.two_stage_pg,
        "dso_pg": cfg.evaluation.dso_pg,
    }
    process(conf)


if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()
    print("total runtime:", format_runtime(start, finish))
