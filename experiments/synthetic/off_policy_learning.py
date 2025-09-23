"""Scripts for running the OPL methods."""
from typing import Optional, Dict, Any
from pathlib import Path
from copy import deepcopy
import time

import hydra
from omegaconf import DictConfig
from utils import assert_configuration, format_runtime

from function import (
    load_dataset,
    load_logging_reward_predictor,
    load_logging_policy,
    generate_logged_data,
    load_reward_predictor,
    load_kernel_marginal_estimator,
    train_and_save_single_stage_policy,
    train_and_save_two_stage_policy,
    train_and_save_dso_policy,
)


def _process(
    n_samples: int,
    n_actions: int,
    beta: float,
    dim_context: int,
    dim_query: int,
    dim_action_embedding: int,
    dim_auxiliary_output: int,
    action_output_mapping_noise: float,
    mapping_function: str,
    reward_std: float,
    reward_std_for_regression: float,
    gradient_type: str,
    kernel_type: str,
    clustering_type: str,
    n_epochs: int,
    n_steps_per_epoch: int,
    n_steps_per_epoch_predictor: int,
    n_epochs_per_log: int,
    tau: float,
    output_noise: float,
    n_samples_to_approximate: int,
    n_clusters: int,
    is_two_stage_policy: bool = False,
    is_two_stage_regression: bool = False,
    is_pessimistic_regression: bool = False,
    is_monte_carlo_estimation: bool = False,
    is_dso: bool = False,
    device: str = "cuda",
    random_state: Optional[int] = None,
    **kwargs,
):
    dataset = load_dataset(
        n_actions=n_actions,
        dim_context=dim_context,
        dim_query=dim_query,
        dim_action_embedding=dim_action_embedding,
        dim_auxiliary_output=dim_auxiliary_output,
        action_output_mapping_noise=action_output_mapping_noise,
        mapping_function=mapping_function,
        reward_std=reward_std,
        device=device,
        random_state=random_state,
    )

    logging_action_reward_predictor = load_logging_reward_predictor(
        dataset=dataset,
        save_path=Path(
            f"logs/logging_reward_predictor/"
            f"{mapping_function}_{n_actions}_{dim_context}_{dim_query}_{dim_auxiliary_output}_{action_output_mapping_noise}_{reward_std}_{device}_{random_state}",
        ),
        device=device,
        random_state=random_state,
    )

    logging_policy = load_logging_policy(
        dataset=dataset,
        logging_action_reward_predictor=logging_action_reward_predictor,
        beta=beta,
        device=device,
        random_state=random_state,
    )
    logged_feedback = generate_logged_data(
        dataset=dataset,
        logging_policy=logging_policy,
        n_samples=n_samples,
    )

    pessimistic_indicator = "_pessimistic" if is_pessimistic_regression else ""
    two_stage_indicator = "_two_stage" if is_two_stage_regression else ""
    output_reward_predictor, action_reward_predictor = load_reward_predictor(
        dataset=dataset,
        save_path_output_reward_predictor=Path(
            f"logs/output_reward_predictor{pessimistic_indicator}{two_stage_indicator}/"
            f"{mapping_function}_{n_actions}_{dim_context}_{dim_query}_{dim_auxiliary_output}_{action_output_mapping_noise}_{reward_std}_{reward_std_for_regression}_{device}_{random_state}_{n_samples}_{beta}",
        ),
        save_path_action_reward_predictor=Path(
            f"logs/action_reward_predictor{pessimistic_indicator}{two_stage_indicator}/"
            f"{mapping_function}_{n_actions}_{dim_context}_{dim_query}_{dim_auxiliary_output}_{action_output_mapping_noise}_{reward_std}_{reward_std_for_regression}_{device}_{random_state}_{n_samples}_{beta}",
        ),
        device=device,
        random_state=random_state,
    )

    two_stage_indicator = "_two_stage_regression" if is_two_stage_regression else ""
    if not (is_two_stage_policy or is_dso):
        Path(
            f"logs/policy_single_{gradient_type}{pessimistic_indicator}{two_stage_indicator}/"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )
        Path(
            f"logs/learning_process/policy_single_{gradient_type}{pessimistic_indicator}{two_stage_indicator}/"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )
        train_and_save_single_stage_policy(
            dataset=dataset,
            logged_feedback=logged_feedback,
            action_reward_predictor=action_reward_predictor,
            gradient_type=gradient_type,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            n_epochs_per_log=n_epochs_per_log,
            save_path=Path(
                f"logs/policy_single_{gradient_type}{pessimistic_indicator}{two_stage_indicator}/"
                f"{mapping_function}_{n_actions}_{dim_context}_{dim_query}_{dim_auxiliary_output}_{action_output_mapping_noise}_{reward_std}_{reward_std_for_regression}_{device}_{random_state}_{n_samples}_{beta}_{n_epochs}_{n_steps_per_epoch}",
            ),
            save_path_logs=Path(
                f"logs/learning_process/policy_single_{gradient_type}{pessimistic_indicator}{two_stage_indicator}/"
                f"{mapping_function}_{n_actions}_{dim_context}_{dim_query}_{dim_auxiliary_output}_{action_output_mapping_noise}_{reward_std}_{reward_std_for_regression}_{device}_{random_state}_{n_samples}_{beta}_{n_epochs}_{n_steps_per_epoch}",
            ),
            device=device,
            random_state=random_state,
        )

    elif is_dso:
        if kernel_type == "uniform":
            tau = tau * 3.0

        kernel_marginal_estimator = load_kernel_marginal_estimator(
            dataset=dataset,
            kernel_type=kernel_type,
            tau=tau,
            output_noise=output_noise,
            save_path=Path(
                f"logs/kernel_marginal_estimator_{kernel_type}/"
                f"{mapping_function}_{n_actions}_{dim_context}_{dim_query}_{dim_auxiliary_output}_{action_output_mapping_noise}_{device}_{random_state}_{n_samples}_{beta}_{tau}_{output_noise}",
            ),
            device=device,
            random_state=random_state,
        )

        monte_carlo_indicator = "_monte_carlo" if is_monte_carlo_estimation else ""
        Path(
            f"logs/policy_dso_{gradient_type}{pessimistic_indicator}{two_stage_indicator}{monte_carlo_indicator}/"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )
        Path(
            f"logs/learning_process/policy_dso_{gradient_type}{pessimistic_indicator}{two_stage_indicator}{monte_carlo_indicator}/"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )
        train_and_save_dso_policy(
            dataset=dataset,
            logged_feedback=logged_feedback,
            kernel_marginal_estimator=kernel_marginal_estimator,
            action_reward_predictor=action_reward_predictor,
            output_reward_predictor=output_reward_predictor,
            gradient_type=gradient_type,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            n_epochs_per_log=n_epochs_per_log,
            use_monte_carlo=is_monte_carlo_estimation,
            n_samples_to_approximate=n_samples_to_approximate,
            save_path=Path(
                f"logs/policy_dso_{gradient_type}{pessimistic_indicator}{two_stage_indicator}{monte_carlo_indicator}/"
                f"{mapping_function}_{n_actions}_{dim_context}_{dim_query}_{dim_auxiliary_output}_{action_output_mapping_noise}_{reward_std}_{reward_std_for_regression}_{device}_{random_state}_{n_samples}_{beta}_{n_epochs}_{n_steps_per_epoch}_{n_steps_per_epoch_predictor}_{kernel_type}_{tau}_{output_noise}_{n_samples_to_approximate}",
            ),
            save_path_logs=Path(
                f"logs/learning_process/policy_dso_{gradient_type}{pessimistic_indicator}{two_stage_indicator}{monte_carlo_indicator}/"
                f"{mapping_function}_{n_actions}_{dim_context}_{dim_query}_{dim_auxiliary_output}_{action_output_mapping_noise}_{reward_std}_{reward_std_for_regression}_{device}_{random_state}_{n_samples}_{beta}_{n_epochs}_{n_steps_per_epoch}_{n_steps_per_epoch_predictor}_{kernel_type}_{tau}_{output_noise}_{n_samples_to_approximate}",
            ),
            device=device,
            random_state=random_state,
        )

    else:
        Path(
            f"logs/policy_two_{gradient_type}_{clustering_type}{pessimistic_indicator}{two_stage_indicator}/"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )
        Path(
            f"logs/learning_process/policy_two_{gradient_type}_{clustering_type}{pessimistic_indicator}{two_stage_indicator}/"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )
        train_and_save_two_stage_policy(
            dataset=dataset,
            logged_feedback=logged_feedback,
            logging_policy=logging_policy,
            action_reward_predictor=action_reward_predictor,
            clustering_type=clustering_type,
            gradient_type=gradient_type,
            n_clusters=n_clusters,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            n_epochs_per_log=n_epochs_per_log,
            save_path=Path(
                f"logs/policy_two_{gradient_type}_{clustering_type}{pessimistic_indicator}{two_stage_indicator}/"
                f"{mapping_function}_{n_actions}_{n_clusters}_{dim_context}_{dim_query}_{dim_auxiliary_output}_{action_output_mapping_noise}_{reward_std}_{reward_std_for_regression}_{device}_{random_state}_{n_samples}_{beta}_{n_epochs}_{n_steps_per_epoch}",
            ),
            save_path_logs=Path(
                f"logs/learning_process/policy_two_{gradient_type}_{clustering_type}{pessimistic_indicator}{two_stage_indicator}/"
                f"{mapping_function}_{n_actions}_{n_clusters}_{dim_context}_{dim_query}_{dim_auxiliary_output}_{action_output_mapping_noise}_{reward_std}_{reward_std_for_regression}_{device}_{random_state}_{n_samples}_{beta}_{n_epochs}_{n_steps_per_epoch}",
            ),
            device=device,
            random_state=random_state,
        )


def process(
    conf: Dict[str, Any],
):
    setting = conf["setting"]
    n_random_state = conf["n_random_state"]
    start_random_state = conf["start_random_state"]

    if setting == "data_size":
        conf_ = deepcopy(conf)

        for random_state in range(start_random_state, n_random_state):
            conf_["random_state"] = random_state

            for n_samples in conf["n_samples"]:
                conf_["n_samples"] = n_samples

                _process(**conf_)

    elif setting == "candidate_actions":
        conf_ = deepcopy(conf)

        for random_state in range(start_random_state, n_random_state):
            conf_["random_state"] = random_state

            for n_actions in conf["n_actions"]:
                conf_["n_actions"] = n_actions

                _process(**conf_)

    elif setting == "logging_policy":
        conf_ = deepcopy(conf)

        for random_state in range(start_random_state, n_random_state):
            conf_["random_state"] = random_state

            for beta in conf["beta"]:
                conf_["beta"] = beta

                _process(**conf_)

    elif setting == "output_noise":
        conf_ = deepcopy(conf)

        for random_state in range(start_random_state, n_random_state):
            conf_["random_state"] = random_state

            for output_noise in conf["output_noise"]:
                conf_["output_noise"] = output_noise

                _process(**conf_)

    elif setting == "reward_noise":
        conf_ = deepcopy(conf)

        for random_state in range(start_random_state, n_random_state):
            conf_["random_state"] = random_state

            for reward_std in conf["reward_std"]:
                conf_["reward_std"] = reward_std
                conf_["reward_std_for_regression"] = reward_std

                _process(**conf_)

    elif setting == "reward_regression":
        conf_ = deepcopy(conf)

        for random_state in range(start_random_state, n_random_state):
            conf_["random_state"] = random_state

            for reward_std_for_regression in conf["reward_std_for_regression"]:
                conf_["reward_std_for_regression"] = reward_std_for_regression

                _process(**conf_)

    elif setting == "kernel_bandwidth":
        conf_ = deepcopy(conf)

        for random_state in range(start_random_state, n_random_state):
            conf_["random_state"] = random_state

            for tau in conf["tau"]:
                conf_["tau"] = tau

                _process(**conf_)

    else:
        conf_ = deepcopy(conf)

        for random_state in range(start_random_state, n_random_state):
            conf_["random_state"] = random_state

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
        "dim_context": cfg.setting.dim_context,
        "dim_query": cfg.setting.dim_query,
        "dim_action_embedding": cfg.setting.dim_action_embedding,
        "dim_auxiliary_output": cfg.setting.dim_auxiliary_output,
        "action_output_mapping_noise": cfg.setting.action_output_mapping_noise,
        "mapping_function": cfg.setting.mapping_function,
        "reward_std": cfg.setting.reward_std,
        "reward_std_for_regression": cfg.setting.reward_std_for_regression,
        "n_epochs": cfg.setting.n_epochs,
        "n_steps_per_epoch": cfg.setting.n_steps_per_epoch,
        "n_steps_per_epoch_predictor": cfg.setting.n_steps_per_epoch_predictor,
        "n_epochs_per_log": cfg.setting.n_epochs_per_log,
        "clustering_type": cfg.setting.clustering_type,
        "gradient_type": cfg.setting.gradient_type,
        "is_two_stage_policy": cfg.setting.is_two_stage_policy,
        "is_two_stage_regression": cfg.setting.is_two_stage_regression,
        "is_pessimistic_regression": cfg.setting.is_pessimistic_regression,
        "is_dso": cfg.setting.is_dso,
        "kernel_type": cfg.setting.kernel_type,
        "tau": cfg.setting.tau,
        "output_noise": cfg.setting.output_noise,
        "is_monte_carlo_estimation": cfg.setting.is_monte_carlo_estimation,
        "n_samples_to_approximate": cfg.setting.n_samples_to_approximate,
        "n_clusters": cfg.setting.n_clusters,
        "device": cfg.setting.device,
        "n_random_state": cfg.setting.n_random_state,
        "start_random_state": cfg.setting.start_random_state,
    }
    process(conf)


if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()
    print("total runtime:", format_runtime(start, finish))
