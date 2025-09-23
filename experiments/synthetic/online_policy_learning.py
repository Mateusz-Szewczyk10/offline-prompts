"""Scripts for running online policy learning."""
from typing import Optional, Dict, Any
from pathlib import Path
from copy import deepcopy
import time

import hydra
from omegaconf import DictConfig
from utils import assert_configuration, format_runtime

from function import (
    load_dataset,
    train_and_save_online_policy,
)


def _process(
    n_actions: int,
    dim_context: int,
    dim_query: int,
    dim_action_embedding: int,
    dim_auxiliary_output: int,
    action_output_mapping_noise: float,
    mapping_function: str,
    reward_std: float,
    n_epochs: int,
    n_steps_per_epoch: int,
    n_epochs_per_log: int,
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

    Path(f"logs/policy_online/").mkdir(parents=True, exist_ok=True)
    Path(f"logs/learning_process/policy_online/").mkdir(parents=True, exist_ok=True)

    train_and_save_online_policy(
        dataset=dataset,
        n_epochs=n_epochs,
        n_steps_per_epoch=n_steps_per_epoch,
        n_epochs_per_log=n_epochs_per_log,
        save_path=Path(
            f"logs/policy_online/"
            f"{mapping_function}_{n_actions}_{dim_context}_{dim_query}_{dim_auxiliary_output}_{action_output_mapping_noise}_{reward_std}_{device}_{random_state}_{n_epochs}_{n_steps_per_epoch}",
        ),
        save_path_logs=Path(
            f"logs/learning_process/policy_online/"
            f"{mapping_function}_{n_actions}_{dim_context}_{dim_query}_{dim_auxiliary_output}_{action_output_mapping_noise}_{reward_std}_{device}_{random_state}_{n_epochs}_{n_steps_per_epoch}",
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

    if setting == "candidate_actions":
        conf_ = deepcopy(conf)

        for random_state in range(start_random_state, n_random_state):
            conf_["random_state"] = random_state

            for n_actions in conf["n_actions"]:
                conf_["n_actions"] = n_actions

                _process(**conf_)

    elif setting == "reward_noise":
        conf_ = deepcopy(conf)

        for random_state in range(start_random_state, n_random_state):
            conf_["random_state"] = random_state

            for reward_std in conf["reward_std"]:
                conf_["reward_std"] = reward_std
                conf_["reward_std_for_regression"] = reward_std

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
