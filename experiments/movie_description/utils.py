"""Useful tools for experiments."""
from typing import Optional
from omegaconf import DictConfig

import random
import torch
from torch.optim import Adam, Adagrad, SGD
from transformers import set_seed


def assert_configuration(cfg: DictConfig):
    # base setting
    setting = cfg.setting.setting
    assert setting == "default"

    n_samples = cfg.setting.n_samples
    assert isinstance(n_samples, int) and n_samples > 0

    n_actions = cfg.setting.n_actions
    assert isinstance(n_actions, int) and n_actions > 0

    beta = cfg.setting.beta
    assert isinstance(beta, float) and 0.0 <= beta

    dim_query = cfg.setting.dim_query
    assert isinstance(dim_query, int) and dim_query > 0

    dim_prompt = cfg.setting.dim_prompt
    assert isinstance(dim_prompt, int) and dim_prompt > 0

    dim_sentence = cfg.setting.dim_sentence
    assert isinstance(dim_sentence, int) and dim_sentence > 0

    reward_simulator_type = cfg.setting.reward_simulator_type
    assert reward_simulator_type in [
        "distilbert",
        "prompt-cossim",
        "sentence-cossim",
    ]

    reward_type = cfg.setting.reward_type
    assert reward_type in ["binary", "continuous"]

    reward_std = cfg.setting.reward_std
    assert isinstance(reward_std, float) and reward_std >= 0

    # online
    n_epochs = cfg.online.n_epochs
    assert isinstance(n_epochs, int) and n_epochs > 0

    n_steps_per_epoch = cfg.online.n_steps_per_epoch
    assert isinstance(n_steps_per_epoch, int) and n_steps_per_epoch > 0

    lr = cfg.online.lr
    assert isinstance(lr, float) and 0 < lr < 1

    optimizer = cfg.online.optimizer
    assert optimizer in [
        "Adam",
        "Adagrad",
        "SGD",
    ]

    # offline
    n_epochs = cfg.offline.n_epochs
    assert isinstance(n_epochs, int) and n_epochs > 0

    n_steps_per_epoch = cfg.offline.n_steps_per_epoch
    assert isinstance(n_steps_per_epoch, int) and n_steps_per_epoch > 0

    n_epochs_per_log = cfg.offline.n_epochs_per_log
    assert isinstance(n_epochs_per_log, int) and 0 < n_epochs_per_log <= n_epochs

    lr = cfg.offline.lr
    assert isinstance(lr, float) and 0 < lr < 1

    optimizer = cfg.offline.optimizer
    assert optimizer in [
        "Adam",
        "Adagrad",
        "SGD",
    ]

    # predictor
    n_epochs_predictor = cfg.offline.n_epochs_predictor
    assert isinstance(n_epochs, int) and n_epochs > 0

    n_steps_per_epoch_predictor = cfg.offline.n_steps_per_epoch_predictor
    assert isinstance(n_steps_per_epoch, int) and n_steps_per_epoch > 0

    lr_predictor = cfg.offline.lr_predictor
    assert isinstance(lr, float) and 0 < lr < 1

    optimizer_predictor = cfg.offline.optimizer_predictor
    assert optimizer_predictor in [
        "Adam",
        "Adagrad",
        "SGD",
    ]

    # other offline configs
    kernel_type = cfg.offline.kernel_type
    assert kernel_type in ["gaussian", "uniform"]

    clustering_type = cfg.offline.clustering_type
    assert clustering_type == "fixed-action"

    gradient_type = cfg.offline.gradient_type
    assert gradient_type in [
        "regression-based",
        "IS-based",
        "hybrid",
    ]

    is_two_stage_policy = cfg.offline.is_two_stage_policy
    assert isinstance(is_two_stage_policy, bool)

    is_dso = cfg.offline.is_dso
    assert isinstance(is_dso, bool)

    assert not (is_dso and is_two_stage_policy)

    tau = cfg.offline.tau
    assert isinstance(tau, float) and tau >= 0

    n_clusters = cfg.offline.n_clusters
    assert isinstance(n_clusters, int) and n_clusters > 0

    device = cfg.setting.device
    assert device == "cuda"

    # path
    path_to_user_embeddings = cfg.setting.path_to_user_embeddings
    assert isinstance(
        path_to_user_embeddings, str
    ) and path_to_user_embeddings.endswith(".pt")

    path_to_queries = cfg.setting.path_to_queries
    assert isinstance(path_to_queries, str) and path_to_queries.endswith(".csv")

    path_to_query_embeddings = cfg.setting.path_to_query_embeddings
    assert isinstance(
        path_to_query_embeddings, str
    ) and path_to_query_embeddings.endswith(".pt")

    path_to_interaction_data = cfg.setting.path_to_interaction_data
    if path_to_interaction_data == "None":
        cfg.setting.path_to_interaction_data = None
    else:
        assert isinstance(
            path_to_interaction_data, str
        ) and path_to_interaction_data.endswith(".csv")

    path_to_candidate_prompts = cfg.setting.path_to_candidate_prompts
    assert isinstance(
        path_to_candidate_prompts, str
    ) and path_to_candidate_prompts.endswith(".csv")

    path_to_prompt_embeddings = cfg.setting.path_to_prompt_embeddings
    assert isinstance(
        path_to_prompt_embeddings, str
    ) and path_to_prompt_embeddings.endswith(".pt")

    path_to_finetuned_params = cfg.setting.path_to_finetuned_params
    assert isinstance(
        path_to_finetuned_params, str
    ) and path_to_finetuned_params.endswith(".pt")

    path_to_query_pca_matrix = cfg.setting.path_to_query_pca_matrix
    assert isinstance(
        path_to_query_pca_matrix, str
    ) and path_to_query_pca_matrix.endswith(".pt")

    path_to_prompt_pca_matrix = cfg.setting.path_to_prompt_pca_matrix
    assert isinstance(
        path_to_prompt_pca_matrix, str
    ) and path_to_prompt_pca_matrix.endswith(".pt")

    path_to_sentence_pca_matrix = cfg.setting.path_to_sentence_pca_matrix
    assert isinstance(
        path_to_sentence_pca_matrix, str
    ) and path_to_sentence_pca_matrix.endswith(".pt")

    # random state
    base_random_state = cfg.setting.base_random_state
    assert isinstance(base_random_state, int) and base_random_state >= 0

    dataset_n_random_state = cfg.offline.dataset_n_random_state
    assert isinstance(dataset_n_random_state, int) and dataset_n_random_state > 0

    dataset_start_random_state = cfg.offline.dataset_start_random_state
    assert (
        isinstance(dataset_start_random_state, int) and dataset_start_random_state >= 0
    )

    optimizer_n_random_state = cfg.offline.optimizer_n_random_state
    assert isinstance(optimizer_n_random_state, int) and optimizer_n_random_state > 0

    optimizer_start_random_state = cfg.offline.optimizer_start_random_state
    assert (
        isinstance(optimizer_start_random_state, int)
        and optimizer_start_random_state >= 0
    )


def assert_configuration_for_evaluation(cfg: DictConfig):
    evaluation_setting = cfg.evaluation.setting
    assert evaluation_setting == "default"

    report_logging_policy = cfg.evaluation.report_logging_policy
    assert isinstance(report_logging_policy, bool)

    report_online_policy = cfg.evaluation.report_online_policy
    assert isinstance(report_online_policy, bool)

    report_uniform_policy = cfg.evaluation.report_uniform_policy
    assert isinstance(report_uniform_policy, bool)

    report_regression_based_greedy = cfg.evaluation.report_regression_based_greedy
    assert isinstance(report_regression_based_greedy, bool)

    report_no_prompt_baseline = cfg.evaluation.report_no_prompt_baseline
    assert isinstance(report_no_prompt_baseline, bool)

    report_skyline = cfg.evaluation.report_skyline
    assert isinstance(report_skyline, bool)

    reward_simulator_type = cfg.setting.reward_simulator_type
    if reward_simulator_type != "distilbert":
        assert not report_no_prompt_baseline

    single_stage_pg = cfg.evaluation.single_stage_pg
    for value in single_stage_pg:
        assert value in [
            "regression-based",
            "IS-based",
            "hybrid",
        ]

    two_stage_pg = cfg.evaluation.two_stage_pg
    for value in two_stage_pg:
        assert value in [
            "hybrid",
        ]

    dso_pg = cfg.evaluation.dso_pg
    for value in dso_pg:
        assert value in [
            "IS-based",
        ]

    val_random_state = cfg.evaluation.val_random_state
    assert isinstance(val_random_state, int) and val_random_state >= 0


def get_optimizer(optimizer_name: str):
    if optimizer_name == "Adam":
        optimizer = Adam
    elif optimizer_name == "Adagrad":
        optimizer = Adagrad
    elif optimizer_name == "SGD":
        optimizer = SGD
    return optimizer


def fix_seed(random_state: Optional[int] = None):
    if random_state is not None:
        random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed(random_state)
        set_seed(random_state)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_runtime(start: int, finish: int):
    runtime = finish - start
    hour = int(runtime // 3600)
    min = int((runtime // 60) % 60)
    sec = int(runtime % 60)
    return f"{hour}h.{min}m.{sec}s"
