"""Functions used in the synthetic experiment."""
import pickle
from typing import Optional, Union, Any, Dict, List
from pathlib import Path
from copy import deepcopy

from off_prompts_syn.dataset import SyntheticDataset
from off_prompts_syn.dataset.function import CandidateActionsGenerator
from off_prompts_syn.dataset.function import (
    AuxiliaryOutputGenerator,
    TrigonometricAuxiliaryOutputGenerator,
)
from off_prompts_syn.dataset.function import RewardSimulator
from off_prompts_syn.opl import PolicyLearner, KernelPolicyLearner
from off_prompts_syn.opl import OutputRewardLearner
from off_prompts_syn.opl import ActionRewardLearner
from off_prompts_syn.opl import MarginalDensityLearner
from off_prompts_syn.policy import NeuralActionPolicy as ActionPolicy
from off_prompts_syn.policy import NeuralClusterPolicy as ClusterPolicy
from off_prompts_syn.policy import NeuralOutputRewardPredictor as OutputRewardPredictor
from off_prompts_syn.policy import NeuralActionRewardPredictor as ActionRewardPredictor
from off_prompts_syn.policy import (
    NeuralMarginalDensityEstimator as KernelMarginalEstimator,
)
from off_prompts_syn.policy import KmeansActionClustering
from off_prompts_syn.policy import (
    BasePolicy,
    BaseActionPolicyModel,
    BaseClusterPolicyModel,
)
from off_prompts_syn.policy import (
    SoftmaxPolicy,
    EpsilonGreedyPolicy,
    UniformRandomPolicy,
)
from off_prompts_syn.policy import TwoStagePolicy
from off_prompts_syn.utils import gaussian_kernel, uniform_kernel


Policy = Union[BasePolicy, BaseActionPolicyModel, BaseClusterPolicyModel]


def load_dataset(
    n_actions: int,
    dim_context: int,
    dim_query: int,
    dim_action_embedding: int,
    dim_auxiliary_output: int,
    action_output_mapping_noise: float,
    mapping_function: str,
    reward_std: float,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    candidate_action_generator = CandidateActionsGenerator(
        n_actions=n_actions,
        dim_action_embedding=dim_action_embedding,
        device=device,
        random_state=random_state,
    )

    if mapping_function == "linear-linear":
        auxiliary_output_generator = AuxiliaryOutputGenerator(
            dim_query=dim_query,
            dim_action_embedding=dim_action_embedding,
            dim_auxiliary_output=dim_auxiliary_output,
            noise_level=action_output_mapping_noise,
            device=device,
            random_state=random_state,
        )
    elif mapping_function == "trigonometric-linear":
        auxiliary_output_generator = TrigonometricAuxiliaryOutputGenerator(
            dim_query=dim_query,
            dim_action_embedding=dim_action_embedding,
            dim_auxiliary_output=dim_auxiliary_output,
            noise_level=action_output_mapping_noise,
            device=device,
            random_state=random_state,
        )
    else:
        raise NotImplementedError()

    reward_simulator = RewardSimulator(
        dim_context=dim_context,
        dim_query=dim_query,
        dim_auxiliary_output=dim_auxiliary_output,
        device=device,
        random_state=random_state,
    )

    dataset = SyntheticDataset(
        n_actions=n_actions,
        dim_context=dim_context,
        dim_query=dim_query,
        dim_action_embedding=dim_action_embedding,
        dim_auxiliary_output=dim_auxiliary_output,
        candidate_action_generator=candidate_action_generator,
        auxiliary_output_generator=auxiliary_output_generator,
        reward_simulator=reward_simulator,
        reward_std=reward_std,
        device=device,
        random_state=random_state,
    )
    return dataset


def train_and_save_logging_reward_predictor(
    dataset: SyntheticDataset,
    save_path: Path,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    dataset_ = deepcopy(dataset)
    dataset_.random_state = random_state + 1

    uniform_policy = UniformRandomPolicy(
        action_list=dataset.action_list,
        device=device,
        random_state=random_state,
    )
    logged_feedback_for_pretraining = dataset_.sample_dataset(
        policy=uniform_policy,
        n_samples=10000,
    )
    action_reward_predictor = ActionRewardPredictor(
        action_list=dataset.action_list,
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        device=device,
        random_state=random_state,
    )
    action_reward_learner = ActionRewardLearner(
        model=action_reward_predictor,
        action_list=dataset.action_list,
        auxiliary_output_generator=dataset.auxiliary_output_generator,
        optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.0},
        env=dataset,
        random_state=random_state,
    )
    logging_action_reward_predictor = action_reward_learner.offline_training(
        logged_feedback=logged_feedback_for_pretraining,
        save_path=save_path,
        random_state=random_state,
    )


def load_logging_reward_predictor(
    dataset: SyntheticDataset,
    save_path: Path,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    action_reward_predictor = ActionRewardPredictor(
        action_list=dataset.action_list,
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        device=device,
        random_state=random_state,
    )
    action_reward_learner = ActionRewardLearner(
        model=action_reward_predictor,
        action_list=dataset.action_list,
        auxiliary_output_generator=dataset.auxiliary_output_generator,
        optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.0},
        env=dataset,
        random_state=random_state,
    )
    logging_action_reward_predictor = action_reward_learner.load(
        path=save_path,
        is_init=True,
    )
    return logging_action_reward_predictor


def load_logging_policy(
    dataset: SyntheticDataset,
    logging_action_reward_predictor: ActionRewardLearner,
    beta: float,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    logging_policy = SoftmaxPolicy(
        action_list=dataset.action_list,
        base_model=logging_action_reward_predictor,
        beta=beta,
        device=device,
        random_state=random_state,
    )
    return logging_policy


def generate_logged_data(
    dataset: SyntheticDataset,
    logging_policy: BasePolicy,
    n_samples: int,
):
    logged_feedback = dataset.sample_dataset(
        policy=logging_policy,
        n_samples=n_samples,
    )
    return logged_feedback


def train_and_save_reward_predictor(
    dataset: SyntheticDataset,
    logged_feedback: Dict[str, Any],
    is_pessimistic: bool,
    is_two_stage: bool,
    save_path_output_reward_predictor: Optional[Path],
    save_path_action_reward_predictor: Path,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    output_reward_predictor = OutputRewardPredictor(
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        dim_auxiliary_output=dataset.dim_auxiliary_output,
        device=device,
        random_state=random_state,
    )
    output_reward_learner = OutputRewardLearner(
        action_list=dataset.action_list,
        model=output_reward_predictor,
        auxiliary_output_generator=dataset.auxiliary_output_generator,
        optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.0},
        random_state=random_state,
    )
    output_reward_predictor = output_reward_learner.offline_training(
        logged_feedback=logged_feedback,
        is_pessimistic=is_pessimistic,
        save_path=save_path_output_reward_predictor,
    )
    action_reward_predictor = ActionRewardPredictor(
        action_list=dataset.action_list,
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        device=device,
        random_state=random_state,
    )
    action_reward_learner = ActionRewardLearner(
        model=action_reward_predictor,
        action_list=dataset.action_list,
        auxiliary_output_generator=dataset.auxiliary_output_generator,
        optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.0},
        env=dataset,
        random_state=random_state,
    )
    if is_two_stage:
        action_reward_predictor = action_reward_learner.offline_cloning(
            logged_feedback=logged_feedback,
            teacher_model=output_reward_predictor,
            save_path=save_path_action_reward_predictor,
        )
    else:
        action_reward_predictor = action_reward_learner.offline_training(
            logged_feedback=logged_feedback,
            is_pessimistic=is_pessimistic,
            save_path=save_path_action_reward_predictor,
        )


def load_reward_predictor(
    dataset: SyntheticDataset,
    save_path_output_reward_predictor: Path,
    save_path_action_reward_predictor: Path,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    output_reward_predictor = OutputRewardPredictor(
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        dim_auxiliary_output=dataset.dim_auxiliary_output,
        device=device,
        random_state=random_state,
    )
    output_reward_learner = OutputRewardLearner(
        action_list=dataset.action_list,
        model=output_reward_predictor,
        optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.0},
        random_state=random_state,
    )
    output_reward_predictor = output_reward_learner.load(
        path=save_path_output_reward_predictor,
        is_init=True,
    )
    action_reward_predictor = ActionRewardPredictor(
        action_list=dataset.action_list,
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        device=device,
        random_state=random_state,
    )
    action_reward_learner = ActionRewardLearner(
        model=action_reward_predictor,
        action_list=dataset.action_list,
        auxiliary_output_generator=dataset.auxiliary_output_generator,
        optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.0},
        env=dataset,
        random_state=random_state,
    )
    action_reward_predictor = action_reward_learner.load(
        path=save_path_action_reward_predictor,
        is_init=True,
    )
    return output_reward_predictor, action_reward_predictor


def train_and_save_kernel_marginal_estimator(
    dataset: SyntheticDataset,
    logged_feedback: Dict[str, Any],
    kernel_type: str,
    tau: float,
    output_noise: float,
    save_path: Path,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    if kernel_type == "gaussian":
        kernel_function = gaussian_kernel
    elif kernel_type == "uniform":
        kernel_function = uniform_kernel

    kernel_marginal_estimator = KernelMarginalEstimator(
        action_list=dataset.action_list,
        auxiliary_output_generator=dataset.auxiliary_output_generator,
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        kernel_function=kernel_function,
        kernel_kwargs={"tau": tau},
        emb_noise=output_noise,
        device=device,
        random_state=random_state,
    )
    marginal_density_learner = MarginalDensityLearner(
        model=kernel_marginal_estimator,
        action_list=dataset.action_list,
        auxiliary_output_generator=dataset.auxiliary_output_generator,
        random_state=random_state,
        optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.0},
    )
    kernel_marginal_estimator = marginal_density_learner.simulation_training(
        logged_feedback=logged_feedback,
        save_path=save_path,
    )


def load_kernel_marginal_estimator(
    dataset: SyntheticDataset,
    kernel_type: str,
    tau: float,
    output_noise: float,
    save_path: Path,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    if kernel_type == "gaussian":
        kernel_function = gaussian_kernel
    elif kernel_type == "uniform":
        kernel_function = uniform_kernel

    kernel_marginal_estimator = KernelMarginalEstimator(
        action_list=dataset.action_list,
        auxiliary_output_generator=dataset.auxiliary_output_generator,
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        kernel_function=kernel_function,
        kernel_kwargs={"tau": tau},
        emb_noise=output_noise,
        device=device,
        random_state=random_state,
    )
    marginal_density_learner = MarginalDensityLearner(
        model=kernel_marginal_estimator,
        action_list=dataset.action_list,
        auxiliary_output_generator=dataset.auxiliary_output_generator,
        random_state=random_state,
        optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.0},
    )
    kernel_marginal_estimator = marginal_density_learner.load(
        path=save_path,
        is_init=True,
    )
    return kernel_marginal_estimator


def load_regression_greedy_policy(
    dataset: SyntheticDataset,
    action_reward_predictor: ActionRewardPredictor,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    greedy_policy = EpsilonGreedyPolicy(
        action_list=dataset.action_list,
        base_model=action_reward_predictor,
        device=device,
        random_state=random_state,
    )
    return greedy_policy


def train_and_save_online_policy(
    dataset: SyntheticDataset,
    n_epochs: int,
    n_steps_per_epoch: int,
    n_epochs_per_log: int,
    save_path: Path,
    save_path_logs: str,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    action_reward_predictor = ActionRewardPredictor(
        action_list=dataset.action_list,
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        device=device,
        random_state=random_state,
    )
    policy = ActionPolicy(
        n_actions=dataset.n_actions,
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        device=device,
        random_state=random_state,
    )
    policy_learner = PolicyLearner(
        model=policy,
        action_list=dataset.action_list,
        action_reward_predictor=action_reward_predictor,
        optimizer_kwargs={"lr": 5e-4, "weight_decay": 0.0},
        env=dataset,
        random_state=random_state,
    )
    policy, learning_process = policy_learner.online_policy_gradient(
        return_training_logs=True,
        n_epochs=n_epochs,
        n_steps_per_epoch=n_steps_per_epoch,
        n_epochs_per_log=n_epochs_per_log,
        save_path=save_path,
    )
    with open(save_path_logs, "wb") as f:
        pickle.dump(learning_process, f)


def train_and_save_single_stage_policy(
    dataset: SyntheticDataset,
    logged_feedback: Dict[str, Any],
    action_reward_predictor: ActionRewardPredictor,
    gradient_type: str,
    n_epochs: int,
    n_steps_per_epoch: int,
    n_epochs_per_log: int,
    save_path: Path,
    save_path_logs: str,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    policy = ActionPolicy(
        n_actions=dataset.n_actions,
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        device=device,
        random_state=random_state,
    )
    policy_learner = PolicyLearner(
        model=policy,
        action_list=dataset.action_list,
        action_reward_predictor=action_reward_predictor,
        optimizer_kwargs={"lr": 5e-4, "weight_decay": 0.0},
        env=dataset,
        random_state=random_state,
    )
    if gradient_type == "regression-based":
        policy, learning_process = policy_learner.model_based_policy_gradient(
            logged_feedback=logged_feedback,
            return_training_logs=True,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            n_epochs_per_log=n_epochs_per_log,
            save_path=save_path,
        )
    elif gradient_type == "IS-based":
        (
            policy,
            learning_process,
        ) = policy_learner.importance_sampling_based_policy_gradient(
            logged_feedback=logged_feedback,
            return_training_logs=True,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            n_epochs_per_log=n_epochs_per_log,
            save_path=save_path,
        )
    elif gradient_type == "hybrid":
        policy, learning_process = policy_learner.hybrid_policy_gradient(
            logged_feedback=logged_feedback,
            return_training_logs=True,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            n_epochs_per_log=n_epochs_per_log,
            save_path=save_path,
        )
    with open(save_path_logs, "wb") as f:
        pickle.dump(learning_process, f)


def train_and_save_dso_policy(
    dataset: SyntheticDataset,
    logged_feedback: Dict[str, Any],
    kernel_marginal_estimator: KernelMarginalEstimator,
    action_reward_predictor: ActionRewardPredictor,
    output_reward_predictor: OutputRewardPredictor,
    gradient_type: str,
    n_epochs: int,
    n_steps_per_epoch: int,
    n_epochs_per_log: int,
    use_monte_carlo: bool,
    n_samples_to_approximate: int,
    save_path: Path,
    save_path_logs: str,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    policy = ActionPolicy(
        n_actions=dataset.n_actions,
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        device=device,
        random_state=random_state,
    )
    policy_learner = KernelPolicyLearner(
        model=policy,
        action_list=dataset.action_list,
        kernel_marginal_estimator=kernel_marginal_estimator,
        action_reward_predictor=action_reward_predictor,
        output_reward_predictor=output_reward_predictor,
        optimizer_kwargs={"lr": 5e-4, "weight_decay": 0.0},
        env=dataset,
        random_state=random_state,
    )
    if gradient_type == "IS-based":
        (
            policy,
            learning_process,
        ) = policy_learner.importance_sampling_based_policy_gradient(
            logged_feedback=logged_feedback,
            return_training_logs=True,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            n_epochs_per_log=n_epochs_per_log,
            use_monte_carlo=use_monte_carlo,
            n_samples_to_approximate=n_samples_to_approximate,
            save_path=save_path,
        )
    elif gradient_type == "hybrid":
        policy, learning_process = policy_learner.hybrid_policy_gradient(
            logged_feedback=logged_feedback,
            return_training_logs=True,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            n_epochs_per_log=n_epochs_per_log,
            use_monte_carlo=use_monte_carlo,
            n_samples_to_approximate=n_samples_to_approximate,
            save_path=save_path,
        )
    with open(save_path_logs, "wb") as f:
        pickle.dump(learning_process, f)


def train_and_save_two_stage_policy(
    dataset: SyntheticDataset,
    logged_feedback: Dict[str, Any],
    logging_policy: BasePolicy,
    action_reward_predictor: ActionRewardPredictor,
    clustering_type: str,
    gradient_type: str,
    n_clusters: int,
    n_epochs: int,
    n_steps_per_epoch: int,
    n_epochs_per_log: int,
    save_path: Path,
    save_path_logs: str,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    second_stage_policy = EpsilonGreedyPolicy(
        action_list=dataset.action_list,
        base_model=action_reward_predictor,
        device=device,
        random_state=random_state,
    )
    first_stage_policy = ClusterPolicy(
        n_actions=n_clusters,
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        dim_feature_emb=dataset.dim_action_embedding,
        device=device,
        random_state=random_state,
    )
    clustering_policy = KmeansActionClustering(
        n_clusters=n_clusters,
        action_list=dataset.action_list,
        device=device,
        random_state=random_state,
    )

    policy_learner = PolicyLearner(
        model=first_stage_policy,
        second_stage_policy=second_stage_policy,
        clustering_policy=clustering_policy,
        action_list=dataset.action_list,
        action_reward_predictor=action_reward_predictor,
        optimizer_kwargs={"lr": 5e-4, "weight_decay": 0.0},
        env=dataset,
        random_state=random_state,
    )

    if gradient_type == "IS-based":
        (
            first_stage_policy,
            learning_process,
        ) = policy_learner.importance_sampling_based_policy_gradient(
            logged_feedback=logged_feedback,
            return_training_logs=True,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            n_epochs_per_log=n_epochs_per_log,
            save_path=save_path,
        )
    elif gradient_type == "hybrid":
        first_stage_policy, learning_process = policy_learner.hybrid_policy_gradient(
            logged_feedback=logged_feedback,
            return_training_logs=True,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            n_epochs_per_log=n_epochs_per_log,
            save_path=save_path,
        )

    with open(save_path_logs, "wb") as f:
        pickle.dump(learning_process, f)


def load_online_policy(
    dataset: SyntheticDataset,
    save_path: Path,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    action_reward_predictor = ActionRewardPredictor(
        action_list=dataset.action_list,
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        device=device,
        random_state=random_state,
    )
    policy = ActionPolicy(
        n_actions=dataset.n_actions,
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        device=device,
        random_state=random_state,
    )
    policy_learner = PolicyLearner(
        model=policy,
        action_list=dataset.action_list,
        action_reward_predictor=action_reward_predictor,
        optimizer_kwargs={"lr": 5e-4, "weight_decay": 0.0},
        env=dataset,
        random_state=random_state,
    )
    policy = policy_learner.load(
        path=save_path,
        is_init=True,
    )
    return policy


def load_single_stage_policy(
    dataset: SyntheticDataset,
    action_reward_predictor: ActionRewardPredictor,
    save_path: Path,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    policy = ActionPolicy(
        n_actions=dataset.n_actions,
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        device=device,
        random_state=random_state,
    )
    policy_learner = PolicyLearner(
        model=policy,
        action_list=dataset.action_list,
        action_reward_predictor=action_reward_predictor,
        optimizer_kwargs={"lr": 5e-4, "weight_decay": 0.0},
        env=dataset,
        random_state=random_state,
    )
    policy = policy_learner.load(
        path=save_path,
        is_init=True,
    )
    return policy


def load_dso_policy(
    dataset: SyntheticDataset,
    kernel_marginal_estimator: KernelMarginalEstimator,
    output_reward_predictor: OutputRewardPredictor,
    save_path: Path,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    policy = ActionPolicy(
        n_actions=dataset.n_actions,
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        device=device,
        random_state=random_state,
    )
    policy_learner = KernelPolicyLearner(
        model=policy,
        action_list=dataset.action_list,
        kernel_marginal_estimator=kernel_marginal_estimator,
        output_reward_predictor=output_reward_predictor,
        optimizer_kwargs={"lr": 5e-4, "weight_decay": 0.0},
        env=dataset,
        random_state=random_state,
    )
    policy = policy_learner.load(
        path=save_path,
        is_init=True,
    )
    return policy


def load_two_stage_policy(
    dataset: SyntheticDataset,
    logging_policy: BasePolicy,
    action_reward_predictor: ActionRewardPredictor,
    clustering_type: str,
    n_clusters: int,
    save_path: Path,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    second_stage_policy = EpsilonGreedyPolicy(
        action_list=dataset.action_list,
        base_model=action_reward_predictor,
        device=device,
        random_state=random_state,
    )
    first_stage_policy = ClusterPolicy(
        n_actions=n_clusters,
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        dim_feature_emb=dataset.dim_action_embedding,
        device=device,
        random_state=random_state,
    )
    clustering_policy = KmeansActionClustering(
        n_clusters=n_clusters,
        action_list=dataset.action_list,
        device=device,
        random_state=random_state,
    )

    policy_learner = PolicyLearner(
        model=first_stage_policy,
        second_stage_policy=second_stage_policy,
        clustering_policy=clustering_policy,
        action_list=dataset.action_list,
        action_reward_predictor=action_reward_predictor,
        optimizer_kwargs={"lr": 5e-4, "weight_decay": 0.0},
        env=dataset,
        random_state=random_state,
    )
    first_stage_policy = policy_learner.load(
        path=save_path,
        is_init=True,
    )
    policy = TwoStagePolicy(
        first_stage_policy=first_stage_policy,
        second_stage_policy=second_stage_policy,
        clustering_policy=clustering_policy,
        device=device,
    )
    return policy


def load_oracle_policy(
    dataset: SyntheticDataset,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    optimal_policy = EpsilonGreedyPolicy(
        action_list=dataset.action_list,
        device=device,
        random_state=random_state,
    )
    return optimal_policy


def load_uniform_policy(
    dataset: SyntheticDataset,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    uniform_policy = UniformRandomPolicy(
        action_list=dataset.action_list,
        device=device,
        random_state=random_state,
    )
    return uniform_policy


def evaluate_policy(
    dataset: SyntheticDataset,
    policy: Policy,
    is_oracle_policy: bool = False,
):
    policy_value = dataset.calc_expected_policy_value(
        policy=policy,
        is_oracle_policy=is_oracle_policy,
    )
    return policy_value
