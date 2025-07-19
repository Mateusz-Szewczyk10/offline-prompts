"""Scripts for evaluating the policy performance."""
from typing import Optional
from pathlib import Path
from copy import deepcopy
import time
import math

import pandas as pd

import hydra
from omegaconf import DictConfig
from utils import (
    assert_configuration,
    assert_configuration_for_evaluation,
    format_runtime,
)

from function import (
    load_dataset,
    load_logging_reward_predictor,
    load_logging_policy,
    load_oracle_policy,
    load_uniform_policy,
    load_regression_greedy_policy,
    load_online_policy,
    load_single_stage_policy,
    load_two_stage_policy,
    load_dso_policy,
    load_reward_predictor,
    load_kernel_marginal_estimator,
    evaluate_policy,
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
    tau: float,
    output_noise: float,
    n_samples_to_approximate: int,
    n_clusters: int,
    is_dso: bool = False,
    is_two_stage_policy: bool = False,
    is_pessimistic_regression: bool = False,
    is_two_stage_regression: bool = False,
    is_logging_policy: bool = False,
    is_oracle_policy: bool = False,
    is_online_policy: bool = False,
    is_uniform_policy: bool = False,
    is_regression_based_greedy: bool = False,
    is_monte_carlo_estimation: bool = False,
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

    pessimistic_indicator = "_pessimistic" if is_pessimistic_regression else ""
    two_stage_indicator = "_two_stage" if is_two_stage_regression else ""
    two_stage_indicator_ = "_two_stage_regression" if is_two_stage_regression else ""

    if is_logging_policy:
        logging_reward_predictor = load_logging_reward_predictor(
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
            logging_action_reward_predictor=logging_reward_predictor,
            beta=beta,
            device=device,
            random_state=random_state,
        )
        policy_value = evaluate_policy(
            dataset=dataset,
            policy=logging_policy,
        )

    elif is_oracle_policy:
        oracle_policy = load_oracle_policy(
            dataset=dataset,
            device=device,
            random_state=random_state,
        )
        policy_value = evaluate_policy(
            dataset=dataset,
            policy=oracle_policy,
            is_oracle_policy=True,
        )

    elif is_uniform_policy:
        uniform_policy = load_uniform_policy(
            dataset=dataset,
            device=device,
            random_state=random_state,
        )
        policy_value = evaluate_policy(
            dataset=dataset,
            policy=uniform_policy,
        )

    elif is_regression_based_greedy:
        _, action_reward_predictor = load_reward_predictor(
            dataset=dataset,
            save_path_output_reward_predictor=Path(
                f"logs/output_reward_predictor{pessimistic_indicator}/"
                f"{mapping_function}_{n_actions}_{dim_context}_{dim_query}_{dim_auxiliary_output}_{action_output_mapping_noise}_{reward_std}_{reward_std_for_regression}_{device}_{random_state}_{n_samples}_{beta}",
            ),
            save_path_action_reward_predictor=Path(
                f"logs/action_reward_predictor{pessimistic_indicator}/"
                f"{mapping_function}_{n_actions}_{dim_context}_{dim_query}_{dim_auxiliary_output}_{action_output_mapping_noise}_{reward_std}_{reward_std_for_regression}_{device}_{random_state}_{n_samples}_{beta}",
            ),
            device=device,
            random_state=random_state,
        )
        regression_based_greedy_policy = load_regression_greedy_policy(
            dataset=dataset,
            action_reward_predictor=action_reward_predictor,
            device=device,
            random_state=random_state,
        )
        policy_value = evaluate_policy(
            dataset=dataset,
            policy=regression_based_greedy_policy,
        )

    elif is_online_policy:
        policy = load_online_policy(
            dataset=dataset,
            save_path=Path(
                f"logs/policy_online/"
                f"{mapping_function}_{n_actions}_{dim_context}_{dim_query}_{dim_auxiliary_output}_{action_output_mapping_noise}_{reward_std}_{device}_{random_state}_{n_epochs}_{n_steps_per_epoch}",
            ),
            device=device,
            random_state=random_state,
        )
        policy_value = evaluate_policy(
            dataset=dataset,
            policy=policy,
        )

    elif not (is_two_stage_policy or is_dso):
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
        policy = load_single_stage_policy(
            dataset=dataset,
            action_reward_predictor=action_reward_predictor,
            save_path=Path(
                f"logs/policy_single_{gradient_type}{pessimistic_indicator}{two_stage_indicator_}/"
                f"{mapping_function}_{n_actions}_{dim_context}_{dim_query}_{dim_auxiliary_output}_{action_output_mapping_noise}_{reward_std}_{reward_std_for_regression}_{device}_{random_state}_{n_samples}_{beta}_{n_epochs}_{n_steps_per_epoch}",
            ),
            device=device,
            random_state=random_state,
        )
        policy_value = evaluate_policy(
            dataset=dataset,
            policy=policy,
        )

    elif is_dso:
        if kernel_type == "uniform":
            tau = tau * 3.0

        monte_carlo_indicator = "_monte_carlo" if is_monte_carlo_estimation else ""

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
        policy = load_dso_policy(
            dataset=dataset,
            kernel_marginal_estimator=kernel_marginal_estimator,
            output_reward_predictor=output_reward_predictor,
            save_path=Path(
                f"logs/policy_dso_{gradient_type}{pessimistic_indicator}{two_stage_indicator_}{monte_carlo_indicator}/"
                f"{mapping_function}_{n_actions}_{dim_context}_{dim_query}_{dim_auxiliary_output}_{action_output_mapping_noise}_{reward_std}_{reward_std_for_regression}_{device}_{random_state}_{n_samples}_{beta}_{n_epochs}_{n_steps_per_epoch}_{n_steps_per_epoch_predictor}_{kernel_type}_{tau}_{output_noise}_{n_samples_to_approximate}",
            ),
            device=device,
            random_state=random_state,
        )
        policy_value = evaluate_policy(
            dataset=dataset,
            policy=policy,
        )

    else:
        logging_reward_predictor = load_logging_reward_predictor(
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
            logging_action_reward_predictor=logging_reward_predictor,
            beta=beta,
            device=device,
            random_state=random_state,
        )
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
        policy = load_two_stage_policy(
            dataset=dataset,
            logging_policy=logging_policy,
            action_reward_predictor=action_reward_predictor,
            clustering_type=clustering_type,
            n_clusters=n_clusters,
            save_path=Path(
                f"logs/policy_two_{gradient_type}_{clustering_type}{pessimistic_indicator}{two_stage_indicator_}/"
                f"{mapping_function}_{n_actions}_{n_clusters}_{dim_context}_{dim_query}_{dim_auxiliary_output}_{action_output_mapping_noise}_{reward_std}_{reward_std_for_regression}_{device}_{random_state}_{n_samples}_{beta}_{n_epochs}_{n_steps_per_epoch}",
            ),
            device=device,
            random_state=random_state,
        )
        policy_value = evaluate_policy(
            dataset=dataset,
            policy=policy,
        )
    return policy_value


def process(conf: DictConfig):
    setting = conf["setting"]
    evaluation_setting = conf["evaluation_setting"]

    mapping_function = conf["mapping_function"]
    kernel_type = conf["kernel_type"]
    reward_predictor_type = conf["reward_predictor_type"]
    n_random_state = conf["n_random_state"]

    is_two_stage_regression = conf["is_two_stage_regression"]
    two_stage_indicator = "_two_stage_regression" if is_two_stage_regression else ""

    is_monte_carlo = conf["is_monte_carlo_estimation"]
    monte_carlo_indicator = "_monte_carlo" if is_monte_carlo else ""

    if "naive" in reward_predictor_type:
        df = pd.DataFrame()
    if "pessimistic" in reward_predictor_type:
        df_pssm = pd.DataFrame()

    if setting == "data_size":
        target_column = "n_samples"
    elif setting == "candidate_actions":
        target_column = "n_actions"
    elif setting == "logging_policy":
        target_column = "beta"
    elif setting == "reward_noise":
        target_column = "reward_std"
    elif setting == "reward_regression":
        target_column = "reward_std_for_regression"
    elif setting == "output_noise":
        target_column = "output_noise"
    elif setting == "kernel_bandwidth":
        target_column = "tau"

    conf_ = deepcopy(conf)
    target_values = conf_[target_column]
    conf_["gradient_type"] = None

    configs = []
    random_states = []
    for value in target_values:
        for random_state in range(n_random_state):
            configs.append(value)
            random_states.append(random_state)

    if conf["report_logging_policy"]:
        print("evaluating logging policy..")
        logging_performance = []
        logging_performance_pssm = []

        for value in target_values:
            conf_[target_column] = value

            for random_state in range(n_random_state):
                conf_["random_state"] = random_state

                if "naive" in reward_predictor_type:
                    conf_["is_pessimistic_regression"] = False
                    logging_performance.append(
                        _process(
                            **conf_,
                            is_logging_policy=True,
                        )
                    )
                if "pessimistic" in reward_predictor_type:
                    conf_["is_pessimistic_regression"] = True
                    logging_performance_pssm.append(
                        _process(
                            **conf_,
                            is_logging_policy=True,
                        )
                    )
        if "naive" in reward_predictor_type:
            df["logging"] = logging_performance

        if "pessimistic" in reward_predictor_type:
            df_pssm["logging"] = logging_performance_pssm

    if conf["report_oracle_policy"]:
        print("evaluating oracle policy..")
        oracle_performance = []
        oracle_performance_pssm = []

        for value in target_values:
            conf_[target_column] = value

            for random_state in range(n_random_state):
                conf_["random_state"] = random_state

                if "naive" in reward_predictor_type:
                    conf_["is_pessimistic_regression"] = False
                    oracle_performance.append(
                        _process(
                            **conf_,
                            is_oracle_policy=True,
                        )
                    )

                if "pessimistic" in reward_predictor_type:
                    conf_["is_pessimistic_regression"] = True
                    oracle_performance_pssm.append(
                        _process(
                            **conf_,
                            is_oracle_policy=True,
                        )
                    )

        if "naive" in reward_predictor_type:
            df["optimal"] = oracle_performance

        if "pessimistic" in reward_predictor_type:
            df_pssm["optimal"] = oracle_performance_pssm

    if conf["report_uniform_policy"]:
        print("evaluating uniform policy..")
        uniform_performance = []
        uniform_performance_pssm = []

        for value in target_values:
            conf_[target_column] = value

            for random_state in range(n_random_state):
                conf_["random_state"] = random_state

                if "naive" in reward_predictor_type:
                    conf_["is_pessimistic_regression"] = False
                    uniform_performance.append(
                        _process(
                            **conf_,
                            is_uniform_policy=True,
                        )
                    )

                if "pessimistic" in reward_predictor_type:
                    conf_["is_pessimistic_regression"] = True
                    uniform_performance_pssm.append(
                        _process(
                            **conf_,
                            is_uniform_policy=True,
                        )
                    )

        if "naive" in reward_predictor_type:
            df["uniform"] = uniform_performance

        if "pessimistic" in reward_predictor_type:
            df_pssm["uniform"] = uniform_performance_pssm

    if conf["report_regression_based_greedy"]:
        print("evaluating regression-based greedy policy..")
        greedy_performance = []
        greedy_performance_pssm = []

        for value in target_values:
            conf_[target_column] = value

            if setting == "reward_noise":
                conf_["reward_std_for_regression"] = value

            for random_state in range(n_random_state):
                conf_["random_state"] = random_state

                if "naive" in reward_predictor_type:
                    conf_["is_pessimistic_regression"] = False
                    greedy_performance.append(
                        _process(
                            **conf_,
                            is_regression_based_greedy=True,
                        )
                    )
                if "pessimistic" in reward_predictor_type:
                    conf_["is_pessimistic_regression"] = True
                    greedy_performance_pssm.append(
                        _process(
                            **conf_,
                            is_regression_based_greedy=True,
                        )
                    )

        if "naive" in reward_predictor_type:
            df["greedy"] = greedy_performance

        if "pessimistic" in reward_predictor_type:
            df_pssm["greedy"] = greedy_performance_pssm

    if conf["report_online_policy"]:
        print("evaluating online policy..")
        online_performance = []
        online_performance_pssm = []

        for value in target_values:
            conf_[target_column] = value

            if setting == "reward_noise":
                conf_["reward_std_for_regression"] = value

            for random_state in range(n_random_state):
                conf_["random_state"] = random_state

                if "naive" in reward_predictor_type:
                    conf_["is_pessimistic_regression"] = False
                    online_performance.append(
                        _process(
                            **conf_,
                            is_online_policy=True,
                        )
                    )
                if "pessimistic" in reward_predictor_type:
                    conf_["is_pessimistic_regression"] = True
                    online_performance_pssm.append(
                        _process(
                            **conf_,
                            is_online_policy=True,
                        )
                    )

        if "naive" in reward_predictor_type:
            df["online"] = online_performance

        if "pessimistic" in reward_predictor_type:
            df_pssm["online"] = online_performance_pssm

    if conf["single_stage_pg"] is not None:
        for gradient_type in conf["single_stage_pg"]:
            print(f"evaluating single stage policy gradient: {gradient_type}..")
            conf_["is_two_stage_policy"] = False
            conf_["is_dso"] = False
            conf_["gradient_type"] = gradient_type
            performance = []
            performance_pssm = []

            for value in target_values:
                conf_[target_column] = value

                if setting == "reward_noise":
                    conf_["reward_std_for_regression"] = value

                for random_state in range(n_random_state):
                    conf_["random_state"] = random_state

                    if "naive" in reward_predictor_type:
                        conf_["is_pessimistic_regression"] = False
                        performance.append(
                            _process(
                                **conf_,
                            )
                        )

                    if "pessimistic" in reward_predictor_type:
                        conf_["is_pessimistic_regression"] = True
                        performance_pssm.append(
                            _process(
                                **conf_,
                            )
                        )

            if "naive" in reward_predictor_type:
                df[f"{gradient_type} (single)"] = performance

            if "pessimistic" in reward_predictor_type:
                df_pssm[f"{gradient_type} (single)"] = performance_pssm

    if conf["dso_pg"] is not None:
        for gradient_type in conf["dso_pg"]:
            print(f"evaluating direct sentence policy gradient: {gradient_type}..")
            conf_["is_two_stage_policy"] = False
            conf_["is_dso"] = True
            conf_["gradient_type"] = gradient_type
            performance = []
            performance_pssm = []

            for value in target_values:
                conf_[target_column] = value

                if setting == "reward_noise":
                    conf_["reward_std_for_regression"] = value

                for random_state in range(n_random_state):
                    conf_["random_state"] = random_state

                    if "naive" in reward_predictor_type:
                        conf_["is_pessimistic_regression"] = False
                        performance.append(
                            _process(
                                **conf_,
                            )
                        )

                    if "pessimistic" in reward_predictor_type:
                        conf_["is_pessimistic_regression"] = True
                        performance_pssm.append(
                            _process(
                                **conf_,
                            )
                        )

            if "naive" in reward_predictor_type:
                df[f"{gradient_type} (dso)"] = performance

            if "pessimistic" in reward_predictor_type:
                df_pssm[f"{gradient_type} (dso)"] = performance_pssm

    if conf["two_stage_pg"] is not None:
        for gradient_type in conf["two_stage_pg"]:
            for clustering_type in conf["clustering_type"]:
                print(
                    f"evaluating two stage policy gradient: {gradient_type} ({clustering_type}).."
                )
                conf_["is_two_stage_policy"] = True
                conf_["is_dso"] = False
                conf_["gradient_type"] = gradient_type
                conf_["clustering_type"] = clustering_type
                performance = []
                performance_pssm = []

                for value in target_values:
                    conf_[target_column] = value

                    if setting == "reward_noise":
                        conf_["reward_std_for_regression"] = value

                    for random_state in range(n_random_state):
                        conf_["random_state"] = random_state

                        if "naive" in reward_predictor_type:
                            conf_["is_pessimistic_regression"] = False
                            performance.append(
                                _process(
                                    **conf_,
                                )
                            )
                        if "pessimistic" in reward_predictor_type:
                            conf_["is_pessimistic_regression"] = True
                            performance_pssm.append(
                                _process(
                                    **conf_,
                                )
                            )

                if "naive" in reward_predictor_type:
                    df[f"{gradient_type} ({clustering_type})"] = performance

                if "pessimistic" in reward_predictor_type:
                    df_pssm[f"{gradient_type} ({clustering_type})"] = performance_pssm

    if "naive" in reward_predictor_type:
        df["target_value"] = configs
        df["random_state"] = random_states
        df.to_csv(
            f"logs/{setting}_{mapping_function}_{evaluation_setting}{two_stage_indicator}_{kernel_type}{monte_carlo_indicator}.csv",
            index=False,
        )

    if "pessimistic" in reward_predictor_type:
        df_pssm["target_value"] = configs
        df_pssm["random_state"] = random_states
        df_pssm.to_csv(
            f"logs/{setting}_{mapping_function}_{evaluation_setting}_pessimistic{two_stage_indicator}_{kernel_type}{monte_carlo_indicator}.csv",
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
        "ia_two_stage_policy": False,
        "is_dso": False,
        "kernel_type": cfg.setting.kernel_type,
        "tau": cfg.setting.tau,
        "output_noise": cfg.setting.output_noise,
        "is_monte_carlo_estimation": cfg.setting.is_monte_carlo_estimation,
        "n_samples_to_approximate": cfg.setting.n_samples_to_approximate,
        "n_clusters": cfg.setting.n_clusters,
        "device": cfg.setting.device,
        "n_random_state": cfg.setting.n_random_state,
        "report_logging_policy": cfg.evaluation.report_logging_policy,
        "report_oracle_policy": cfg.evaluation.report_oracle_policy,
        "report_online_policy": cfg.evaluation.report_online_policy,
        "report_uniform_policy": cfg.evaluation.report_uniform_policy,
        "report_regression_based_greedy": cfg.evaluation.report_regression_based_greedy,
        "is_two_stage_regression": cfg.setting.is_two_stage_regression,
        "single_stage_pg": cfg.evaluation.single_stage_pg,
        "two_stage_pg": cfg.evaluation.two_stage_pg,
        "dso_pg": cfg.evaluation.dso_pg,
        "clustering_type": cfg.evaluation.clustering_type,
        "reward_predictor_type": cfg.evaluation.reward_predictor_type,
    }
    process(conf)


if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()
    print("total runtime:", format_runtime(start, finish))
