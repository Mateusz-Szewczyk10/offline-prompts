"""Scripts for generating logged data and regressing rewards."""
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
    train_and_save_reward_predictor,
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
    is_pessimistic_regression: bool = False,
    is_two_stage_regression: bool = False,
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

    dataset.reward_std = reward_std_for_regression

    logged_feedback = generate_logged_data(
        dataset=dataset,
        logging_policy=logging_policy,
        n_samples=n_samples,
    )
    logged_feedback = {
        "context": logged_feedback["context"],
        "query": logged_feedback["query"],
        "action": logged_feedback["action"],
        "auxiliary_output": logged_feedback["auxiliary_output"],
        "reward": logged_feedback["reward"],
        "logging_policy": logged_feedback["logging_policy"],
    }

    pessimistic_indicator = "_pessimistic" if is_pessimistic_regression else ""
    two_stage_indicator = "_two_stage" if is_two_stage_regression else ""
    Path(
        f"logs/output_reward_predictor{pessimistic_indicator}{two_stage_indicator}"
    ).mkdir(exist_ok=True, parents=True)
    Path(
        f"logs/action_reward_predictor{pessimistic_indicator}{two_stage_indicator}"
    ).mkdir(exist_ok=True, parents=True)
    train_and_save_reward_predictor(
        dataset=dataset,
        logged_feedback=logged_feedback,
        is_pessimistic=is_pessimistic_regression,
        is_two_stage=is_two_stage_regression,
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
