"""Useful tools for the synthtic experiment."""
from omegaconf import DictConfig


def assert_configuration(cfg: DictConfig):
    setting = cfg.setting.setting
    assert setting in [
        "default",
        "data_size",
        "candidate_actions",
        "reward_noise",
        "kernel_bandwidth",
    ]

    n_samples = cfg.setting.n_samples
    if setting == "data_size":
        for value in n_samples:
            assert isinstance(value, int) and value > 0
    else:
        assert isinstance(n_samples, int) and n_samples > 0

    n_actions = cfg.setting.n_actions
    if setting == "candidate_actions":
        for value in n_actions:
            assert isinstance(value, int) and value > 0
    else:
        assert isinstance(n_actions, int) and n_actions > 0

    beta = cfg.setting.beta
    if setting == "logging_policy":
        for value in beta:
            assert isinstance(value, float) and 0.0 <= value
    else:
        assert isinstance(beta, float) and 0.0 <= beta

    reward_std = cfg.setting.reward_std
    if setting == "reward_noise":
        for value in reward_std:
            assert isinstance(value, float) and value >= 0
    else:
        assert isinstance(reward_std, float) and reward_std >= 0

    reward_std_for_regression = cfg.setting.reward_std_for_regression
    if setting == "reward_regression":
        for value in reward_std_for_regression:
            assert isinstance(value, float) and value >= 0.0
    else:
        assert (
            isinstance(reward_std_for_regression, float)
            and reward_std_for_regression >= 0.0
        )

    output_noise = cfg.setting.output_noise
    if setting == "output_noise":
        for value in output_noise:
            assert isinstance(value, float) and value >= 0
    else:
        assert isinstance(output_noise, float) and output_noise >= 0

    mapping_function = cfg.setting.mapping_function
    assert mapping_function in [
        "linear-linear",
        "trigonometric-linear",
    ]

    dim_context = cfg.setting.dim_context
    assert isinstance(dim_context, int) and dim_context > 0

    dim_query = cfg.setting.dim_query
    assert isinstance(dim_query, int) and dim_query > 0

    dim_action_embedding = cfg.setting.dim_action_embedding
    assert isinstance(dim_action_embedding, int) and dim_action_embedding > 0

    dim_auxiliary_output = cfg.setting.dim_auxiliary_output
    assert isinstance(dim_auxiliary_output, int) and dim_auxiliary_output > 0

    action_output_mapping_noise = cfg.setting.action_output_mapping_noise
    assert (
        isinstance(action_output_mapping_noise, float)
        and action_output_mapping_noise >= 0.0
    )

    n_epochs = cfg.setting.n_epochs
    assert isinstance(n_epochs, int) and n_epochs > 0

    n_steps_per_epoch = cfg.setting.n_steps_per_epoch
    assert isinstance(n_steps_per_epoch, int) and n_steps_per_epoch > 0

    n_steps_per_epoch_predictor = cfg.setting.n_steps_per_epoch_predictor
    assert (
        isinstance(n_steps_per_epoch_predictor, int) and n_steps_per_epoch_predictor > 0
    )

    n_epochs_per_log = cfg.setting.n_epochs_per_log
    assert isinstance(n_epochs_per_log, int) and 0 < n_epochs_per_log <= n_epochs

    kernel_type = cfg.setting.kernel_type
    assert kernel_type in ["gaussian", "uniform"]

    clustering_type = cfg.setting.clustering_type
    assert clustering_type in [
        "fixed-action",
    ]

    gradient_type = cfg.setting.gradient_type
    assert gradient_type in [
        "regression-based",
        "IS-based",
        "hybrid",
    ]

    is_two_stage_policy = cfg.setting.is_two_stage_policy
    assert isinstance(is_two_stage_policy, bool)

    is_pessimistic_regression = cfg.setting.is_pessimistic_regression
    assert isinstance(is_pessimistic_regression, bool)

    is_two_stage_regression = cfg.setting.is_two_stage_regression
    assert isinstance(is_two_stage_regression, bool)

    is_dso = cfg.setting.is_dso
    assert isinstance(is_dso, bool)

    assert not (is_dso and is_two_stage_policy)

    tau = cfg.setting.tau
    if setting == "kernel_bandwidth":
        for value in tau:
            assert isinstance(value, float) and value >= 0
    else:
        assert isinstance(tau, float) and tau >= 0

    n_samples_to_approximate = cfg.setting.n_samples_to_approximate
    if setting == "n_samples_to_approximate":
        for value in n_samples_to_approximate:
            assert isinstance(value, int) and value > 0
    else:
        assert (
            isinstance(n_samples_to_approximate, int) and n_samples_to_approximate > 0
        )

    device = cfg.setting.device
    assert device in [
        "cpu",
        "cuda",
    ]

    n_random_state = cfg.setting.n_random_state
    assert isinstance(n_random_state, int) and n_random_state > 0

    start_random_state = cfg.setting.start_random_state
    assert isinstance(start_random_state, int) and start_random_state >= 0


def assert_configuration_for_evaluation(cfg: DictConfig):
    evaluation_setting = cfg.evaluation.setting
    assert evaluation_setting in [
        "IS-based",
        "hybrid",
        "default",
    ]

    report_logging_policy = cfg.evaluation.report_logging_policy
    assert isinstance(report_logging_policy, bool)

    report_oracle_policy = cfg.evaluation.report_oracle_policy
    assert isinstance(report_oracle_policy, bool)

    report_regression_based_greedy = cfg.evaluation.report_regression_based_greedy
    assert isinstance(report_regression_based_greedy, bool)

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
            "IS-based",
            "hybrid",
        ]

    dso_pg = cfg.evaluation.dso_pg
    for value in dso_pg:
        assert value in [
            "IS-based",
        ]

    clustering_type = cfg.evaluation.clustering_type
    for value in clustering_type:
        assert value == "fixed-action"

    reward_predictor_type = cfg.evaluation.reward_predictor_type
    for value in reward_predictor_type:
        assert value in [
            "naive",
            "pessimistic",
        ]


def format_runtime(start: int, finish: int):
    runtime = finish - start
    hour = int(runtime // 3600)
    min = int((runtime // 60) % 60)
    sec = int(runtime % 60)
    return f"{hour}h.{min}m.{sec}s"
