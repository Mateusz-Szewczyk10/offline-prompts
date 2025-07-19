"""Functions used in the experiment."""
import pickle
from typing import Optional, Union, Any, Dict
from pathlib import Path

import torch
from torch.optim import Optimizer
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from off_prompts.dataset import SemiSyntheticDataset
from off_prompts.dataset.base import BaseEncoder
from off_prompts.dataset.encoder import TransformerEncoder, NNSentenceEncoder
from off_prompts.dataset.prompt_formatter import MovielensPromptFormatter
from off_prompts.dataset.function import DefaultContextQueryLoader
from off_prompts.dataset.function import DefaultCandidateActionsLoader
from off_prompts.dataset.frozen_llm import AutoFrozenLLM
from off_prompts.dataset.reward_simulator import TransformerRewardSimulator
from off_prompts.dataset.reward_simulator import (
    PromptCossimRewardSimulator,
    SentenceCossimRewardSimulator,
)
from off_prompts.opl import PolicyLearner, KernelPolicyLearner
from off_prompts.opl import PromptRewardLearner, SentenceRewardLearner
from off_prompts.opl import MarginalDensityLearner
from off_prompts.opl import PolicyEvaluator
from off_prompts.policy import PromptPolicy
from off_prompts.policy import ClusterPolicy
from off_prompts.policy import SentenceRewardPredictor
from off_prompts.policy import PromptRewardPredictor
from off_prompts.policy import KernelMarginalDensityEstimator
from off_prompts.policy import KmeansPromptClustering
from off_prompts.policy import BasePolicy, BasePromptPolicyModel, BaseClusterPolicyModel
from off_prompts.policy import SoftmaxPolicy, EpsilonGreedyPolicy, UniformRandomPolicy
from off_prompts.policy import TwoStagePolicy
from off_prompts.utils import gaussian_kernel, uniform_kernel


Policy = Union[BasePolicy, BasePromptPolicyModel, BaseClusterPolicyModel]


def load_dataset(
    n_actions: int,
    reward_type: str,
    reward_std: float,
    path_to_user_embeddings: str,
    path_to_queries: str,
    path_to_query_embeddings: str,
    path_to_interaction_data: str,
    path_to_candidate_prompts: str,
    path_to_prompt_embeddings: str,
    path_to_finetuned_params: str,
    reward_simulator_type: str,
    dim_sentence: int,
    save_path_sentence_encoder: str,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    context_query_loader = DefaultContextQueryLoader(
        path_to_user_embeddings=path_to_user_embeddings,
        path_to_queries=path_to_queries,
        path_to_query_embeddings=path_to_query_embeddings,
        path_to_interaction_data=path_to_interaction_data,
        device=device,
        random_state=random_state,
    )
    candidate_actions_loader = DefaultCandidateActionsLoader(
        n_actions=1000,
        path_to_candidate_prompts=path_to_candidate_prompts,
        path_to_prompt_embeddings=path_to_prompt_embeddings,
        random_state=random_state,
    )

    frozen_llm_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
    )
    frozen_llm_tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        truncation=True,
        do_lower_case=True,
        use_fast=True,
    )
    frozen_llm_tokenizer_kwargs = {
        "add_special_tokens": True,
        "truncation": True,
        "max_length": 20,
        "pad_to_max_length": True,
        "return_tensors": "pt",
    }
    frozen_llm_prompt_formatter = MovielensPromptFormatter(
        tokenizer=frozen_llm_tokenizer,
        tokenizer_kwargs=frozen_llm_tokenizer_kwargs,
        device=device,
    )
    pattern = (
        r"Broadly describe in a sentence the genres of the movie without including the name or any specifics of.*?\n\n",
    )[0]

    frozen_llm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    frozen_llm_model.resize_token_embeddings(len(frozen_llm_tokenizer))
    frozen_llm_model.to(device)

    frozen_llm = AutoFrozenLLM(
        prompt_formatter=frozen_llm_prompt_formatter,
        model=frozen_llm_model,
        tokenizer=frozen_llm_tokenizer,
        tokenizer_kwargs=frozen_llm_tokenizer_kwargs,
        pattern=pattern,
        device=device,
        random_state=random_state,
    )

    if reward_simulator_type == "distilbert":
        reward_simulator_base_model = AutoModel.from_pretrained(
            "distilbert-base-uncased",
        )
        reward_simulator_tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased",
            truncation=True,
            do_lower_case=True,
            use_fast=True,
        )
        reward_simulator_tokenizer_kwargs = {
            "add_special_tokens": True,
            "truncation": True,
            "max_length": 20,
            "pad_to_max_length": True,
            "return_tensors": "pt",
        }

        reward_simulator_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        reward_simulator_base_model.resize_token_embeddings(
            len(reward_simulator_tokenizer)
        )
        reward_simulator_base_model.to(device)

        reward_simulator = TransformerRewardSimulator(
            n_users=context_query_loader.n_users,
            n_items=context_query_loader.n_queries,
            base_model=reward_simulator_base_model,
            tokenizer=reward_simulator_tokenizer,
            tokenizer_kwargs=reward_simulator_tokenizer_kwargs,
            dim_emb=20,
            device=device,
            random_state=random_state,
        )
        reward_simulator.load_state_dict(torch.load(path_to_finetuned_params))

    elif reward_simulator_type == "prompt-cossim":
        reward_simulator = PromptCossimRewardSimulator(
            user_embeddings=context_query_loader.user_embeddings,
            query_embeddings=context_query_loader.query_embeddings,
            prompt_embeddings=candidate_actions_loader.prompt_embeddings,
            device=device,
            random_state=random_state,
        )

    else:  # "sentence-cossim"
        base_model = frozen_llm.model
        tokenizer = frozen_llm.tokenizer
        tokenizer_kwargs = frozen_llm.tokenizer_kwargs

        sentence_prefix_prompt = " "
        sentence_postfix_prompt = " "
        sentence_prefix_max_length = 1
        sentence_postfix_max_length = 1

        sentence_encoder = TransformerEncoder(
            base_model=base_model,
            tokenizer=tokenizer,
            tokenizer_kwargs=tokenizer_kwargs,
            is_causal_lm=True,
            prefix_prompt=sentence_prefix_prompt,
            postfix_prompt=sentence_postfix_prompt,
            prefix_tokens_max_length=sentence_prefix_max_length,
            postfix_tokens_max_length=sentence_postfix_max_length,
            max_length=20,
            dim_emb=dim_sentence,
            device=device,
            random_state=random_state,
        )
        sentence_encoder.load(save_path_sentence_encoder)

        reward_simulator = SentenceCossimRewardSimulator(
            user_embeddings=context_query_loader.user_embeddings,
            query_embeddings=context_query_loader.query_embeddings,
            sentence_encoder=sentence_encoder,
            device=device,
            random_state=random_state,
        )

    dataset = SemiSyntheticDataset(
        context_query_loader=context_query_loader,
        candidate_actions_loader=candidate_actions_loader,
        frozen_llm=frozen_llm,
        reward_simulator=reward_simulator,
        frozen_llm_prompt_formatter=frozen_llm_prompt_formatter,
        reward_type=reward_type,
        reward_std=reward_std,
        device=device,
        random_state=random_state,
    )
    return dataset


def load_encoders(
    dataset: SemiSyntheticDataset,
    dim_query: int,
    dim_prompt: int,
    dim_sentence: int,
    save_path_query_encoder: Path,
    save_path_prompt_encoder: Path,
    save_path_sentence_encoder: Path,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    base_model = dataset.frozen_llm.model
    tokenizer = dataset.frozen_llm.tokenizer
    tokenizer_kwargs = dataset.frozen_llm.tokenizer_kwargs

    query_prefix_prompt = "Broadly describe in a sentence the genres of the movie without including the name or any specifics of the movie.\nTitle: "
    query_postfix_prompt = " "
    query_prefix_max_length = 22
    query_postfix_max_length = 1

    prompt_prefix_prompt = "Associate the word - "
    prompt_postfix_prompt = " - in the context of movie genres"
    prompt_prefix_max_length = 4
    prompt_postfix_max_length = 7

    sentence_prefix_prompt = " "
    sentence_postfix_prompt = " "
    sentence_prefix_max_length = 1
    sentence_postfix_max_length = 1

    query_encoder = TransformerEncoder(
        base_model=base_model,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        is_causal_lm=True,
        prefix_prompt=query_prefix_prompt,
        postfix_prompt=query_postfix_prompt,
        prefix_tokens_max_length=query_prefix_max_length,
        postfix_tokens_max_length=query_postfix_max_length,
        max_length=5,
        dim_emb=dim_query,
        device=device,
        random_state=random_state,
    )
    prompt_encoder = TransformerEncoder(
        base_model=base_model,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        is_causal_lm=True,
        prefix_prompt=prompt_prefix_prompt,
        postfix_prompt=prompt_postfix_prompt,
        prefix_tokens_max_length=prompt_prefix_max_length,
        postfix_tokens_max_length=prompt_postfix_max_length,
        max_length=2,
        dim_emb=dim_prompt,
        device=device,
        random_state=random_state,
    )
    sentence_encoder = TransformerEncoder(
        base_model=base_model,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        is_causal_lm=True,
        prefix_prompt=sentence_prefix_prompt,
        postfix_prompt=sentence_postfix_prompt,
        prefix_tokens_max_length=sentence_prefix_max_length,
        postfix_tokens_max_length=sentence_postfix_max_length,
        max_length=20,
        dim_emb=dim_sentence,
        device=device,
        random_state=random_state,
    )

    query_encoder.load(save_path_query_encoder)
    prompt_encoder.load(save_path_prompt_encoder)
    sentence_encoder.load(save_path_sentence_encoder)

    return query_encoder, prompt_encoder, sentence_encoder


def load_logging_policy(
    dataset: SemiSyntheticDataset,
    base_policy: Policy,
    beta: float,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    logging_policy = SoftmaxPolicy(
        action_list=dataset.action_list,
        base_model=base_policy,
        beta=beta,
        device=device,
        random_state=random_state,
    )
    return logging_policy


def generate_and_save_logged_data(
    dataset: SemiSyntheticDataset,
    logging_policy: BasePolicy,
    n_samples: int,
    save_path: Path,
):
    logged_feedback = dataset.sample_dataset(
        policy=logging_policy,
        n_samples=n_samples,
    )

    with open(save_path, "wb") as f:
        pickle.dump(logged_feedback, f)

    return logged_feedback


def load_logged_data(
    save_path: Path,
):
    with open(save_path, "rb") as f:
        logged_feedback = pickle.load(save_path)

    return logged_feedback


def train_and_save_reward_predictor(
    dataset: SemiSyntheticDataset,
    logged_feedback: Dict[str, Any],
    query_encoder: BaseEncoder,
    prompt_encoder: BaseEncoder,
    n_epochs: int,
    n_steps_per_epoch: int,
    lr: float,
    optimizer: Optimizer,
    save_path: Path,
    device: str = "cuda",
    random_state: Optional[int] = None,
    use_wandb: bool = False,
):
    prompt_reward_predictor = PromptRewardPredictor(
        dim_context=dataset.dim_context,
        action_list=dataset.action_list,
        query_encoder=query_encoder,
        prompt_encoder=prompt_encoder,
        device=device,
        random_state=random_state,
    )
    prompt_reward_learner = PromptRewardLearner(
        model=prompt_reward_predictor,
        action_list=dataset.action_list,
        query_embeddings=dataset.query_embeddings,
        prompt_embeddings=dataset.prompt_embeddings,
        frozen_llm=dataset.frozen_llm,
        query_encoder=query_encoder,
        prompt_encoder=prompt_encoder,
        optimizer=optimizer,
        optimizer_kwargs={"lr": lr, "weight_decay": 0.0},
        env=dataset,
        random_state=random_state,
    )
    prompt_reward_predictor = prompt_reward_learner.offline_training(
        n_epochs=n_epochs,
        n_steps_per_epoch=n_steps_per_epoch,
        logged_feedback=logged_feedback,
        save_path=save_path,
        use_wandb=use_wandb,
        experiment_name=f"movielens-regression-D{random_state}",
    )


def load_reward_predictor(
    dataset: SemiSyntheticDataset,
    query_encoder: BaseEncoder,
    prompt_encoder: BaseEncoder,
    save_path: Path,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    prompt_reward_predictor = PromptRewardPredictor(
        dim_context=dataset.dim_context,
        action_list=dataset.action_list,
        query_encoder=query_encoder,
        prompt_encoder=prompt_encoder,
        device=device,
        random_state=random_state,
    )
    prompt_reward_learner = PromptRewardLearner(
        model=prompt_reward_predictor,
        action_list=dataset.action_list,
        query_embeddings=dataset.query_embeddings,
        prompt_embeddings=dataset.prompt_embeddings,
        frozen_llm=dataset.frozen_llm,
        query_encoder=query_encoder,
        prompt_encoder=prompt_encoder,
        env=dataset,
        random_state=random_state,
    )
    prompt_reward_predictor = prompt_reward_learner.load(
        path=save_path,
        is_init=True,
    )
    return prompt_reward_predictor


def train_and_save_kernel_marginal_estimator(
    dataset: SemiSyntheticDataset,
    logged_feedback: Dict[str, Any],
    query_encoder: BaseEncoder,
    sentence_encoder: BaseEncoder,
    kernel_type: str,
    tau: float,
    n_epochs: int,
    n_steps_per_epoch: int,
    lr: float,
    optimizer: Optimizer,
    save_path: Path,
    device: str = "cuda",
    random_state: Optional[int] = None,
    use_wandb: bool = False,
):
    if kernel_type == "gaussian":
        kernel_function = gaussian_kernel
    elif kernel_type == "uniform":
        kernel_function = uniform_kernel

    kernel_marginal_estimator = KernelMarginalDensityEstimator(
        action_list=dataset.action_list,
        dim_context=dataset.dim_context,
        frozen_llm=dataset.frozen_llm,
        query_encoder=query_encoder,
        sentence_encoder=sentence_encoder,
        kernel_function=kernel_function,
        kernel_kwargs={"tau": tau},
        device=device,
        random_state=random_state,
    )
    marginal_density_learner = MarginalDensityLearner(
        model=kernel_marginal_estimator,
        action_list=dataset.action_list,
        query_embeddings=dataset.query_embeddings,
        prompt_embeddings=dataset.prompt_embeddings,
        frozen_llm=dataset.frozen_llm,
        random_state=random_state,
        optimizer=optimizer,
        optimizer_kwargs={"lr": lr, "weight_decay": 0.0},
    )
    kernel_marginal_estimator = marginal_density_learner.simulation_training(
        n_epochs=n_epochs,
        n_steps_per_epoch=n_steps_per_epoch,
        logged_feedback=logged_feedback,
        save_path=save_path,
        use_wandb=use_wandb,
        experiment_name=f"movielens-kernel-D{random_state}",
    )


def train_and_save_eval_kernel_marginal_estimator(
    dataset: SemiSyntheticDataset,
    logged_feedback: Dict[str, Any],
    eval_policy: Policy,
    query_encoder: BaseEncoder,
    sentence_encoder: BaseEncoder,
    kernel_type: str,
    tau: float,
    n_epochs: int,
    n_steps_per_epoch: int,
    lr: float,
    optimizer: Optimizer,
    save_path: Path,
    device: str = "cuda",
    random_state: Optional[int] = None,
    use_wandb: bool = False,
):
    logging_policy = logged_feedback["logging_policy"]
    logged_feedback["logging_policy"] = eval_policy

    if kernel_type == "gaussian":
        kernel_function = gaussian_kernel
    elif kernel_type == "uniform":
        kernel_function = uniform_kernel

    kernel_marginal_estimator = KernelMarginalDensityEstimator(
        action_list=dataset.action_list,
        dim_context=dataset.dim_context,
        frozen_llm=dataset.frozen_llm,
        query_encoder=query_encoder,
        sentence_encoder=sentence_encoder,
        kernel_function=kernel_function,
        kernel_kwargs={"tau": tau},
        device=device,
        random_state=random_state,
    )
    marginal_density_learner = MarginalDensityLearner(
        model=kernel_marginal_estimator,
        action_list=dataset.action_list,
        query_embeddings=dataset.query_embeddings,
        prompt_embeddings=dataset.prompt_embeddings,
        frozen_llm=dataset.frozen_llm,
        random_state=random_state,
        optimizer=optimizer,
        optimizer_kwargs={"lr": lr, "weight_decay": 0.0},
    )
    kernel_marginal_estimator = marginal_density_learner.simulation_training(
        n_epochs=n_epochs,
        n_steps_per_epoch=n_steps_per_epoch,
        logged_feedback=logged_feedback,
        save_path=save_path,
        use_wandb=use_wandb,
        experiment_name=f"movielens-eval-kernel-D{random_state}",
    )
    logged_feedback["logging_policy"] = logging_policy


def load_kernel_marginal_estimator(
    dataset: SemiSyntheticDataset,
    query_encoder: BaseEncoder,
    sentence_encoder: BaseEncoder,
    kernel_type: str,
    tau: float,
    save_path: Path,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    if kernel_type == "gaussian":
        kernel_function = gaussian_kernel
    elif kernel_type == "uniform":
        kernel_function = uniform_kernel

    kernel_marginal_estimator = KernelMarginalDensityEstimator(
        action_list=dataset.action_list,
        dim_context=dataset.dim_context,
        frozen_llm=dataset.frozen_llm,
        query_encoder=query_encoder,
        sentence_encoder=sentence_encoder,
        kernel_function=kernel_function,
        kernel_kwargs={"tau": tau},
        device=device,
        random_state=random_state,
    )
    marginal_density_learner = MarginalDensityLearner(
        model=kernel_marginal_estimator,
        action_list=dataset.action_list,
        query_embeddings=dataset.query_embeddings,
        prompt_embeddings=dataset.prompt_embeddings,
        frozen_llm=dataset.frozen_llm,
        random_state=random_state,
    )
    kernel_marginal_estimator = marginal_density_learner.load(
        path=save_path,
        is_init=True,
    )
    return kernel_marginal_estimator


def load_regression_greedy_policy(
    dataset: SemiSyntheticDataset,
    prompt_reward_predictor: PromptRewardPredictor,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    greedy_policy = EpsilonGreedyPolicy(
        action_list=dataset.action_list,
        base_model=prompt_reward_predictor,
        device=device,
        random_state=random_state,
    )
    return greedy_policy


def train_and_save_online_policy(
    dataset: SemiSyntheticDataset,
    query_encoder: BaseEncoder,
    prompt_encoder: BaseEncoder,
    sentence_encoder: BaseEncoder,
    n_epochs: int,
    n_steps_per_epoch: int,
    n_epochs_per_log: int,
    lr: float,
    optimizer: Optimizer,
    save_path: Path,
    save_path_logs: str,
    device: str = "cuda",
    random_state: Optional[int] = None,
    use_wandb: bool = False,
):
    prompt_reward_predictor = PromptRewardPredictor(
        dim_context=dataset.dim_context,
        action_list=dataset.action_list,
        query_encoder=query_encoder,
        prompt_encoder=prompt_encoder,
        device=device,
        random_state=random_state,
    )
    policy = PromptPolicy(
        n_actions=dataset.n_actions,
        dim_context=dataset.dim_context,
        query_encoder=query_encoder,
        device=device,
        random_state=random_state,
    )
    policy_learner = PolicyLearner(
        model=policy,
        action_list=dataset.action_list,
        query_embeddings=dataset.query_embeddings,
        prompt_embeddings=dataset.prompt_embeddings,
        prompt_reward_predictor=prompt_reward_predictor,
        query_encoder=query_encoder,
        sentence_encoder=sentence_encoder,
        optimizer=optimizer,
        optimizer_kwargs={"lr": lr, "weight_decay": 0.0},
        env=dataset,
        random_state=random_state,
    )
    policy, learning_process = policy_learner.online_policy_gradient(
        return_training_logs=True,
        n_epochs=n_epochs,
        n_steps_per_epoch=n_steps_per_epoch,
        n_epochs_per_log=n_epochs_per_log,
        save_path=save_path,
        use_wandb=use_wandb,
        experiment_name="movielens-online",
    )
    with open(save_path_logs, "wb") as f:
        pickle.dump(learning_process, f)


def train_and_save_single_stage_policy(
    dataset: SemiSyntheticDataset,
    logged_feedback: Dict[str, Any],
    prompt_reward_predictor: PromptRewardPredictor,
    query_encoder: BaseEncoder,
    sentence_encoder: BaseEncoder,
    gradient_type: str,
    n_epochs: int,
    n_steps_per_epoch: int,
    n_epochs_per_log: int,
    lr: float,
    optimizer: Optimizer,
    save_path: Path,
    save_path_logs: str,
    device: str = "cuda",
    random_state: Optional[int] = None,
    base_random_state: Optional[int] = None,
    use_wandb: bool = False,
):
    policy = PromptPolicy(
        n_actions=dataset.n_actions,
        dim_context=dataset.dim_context,
        query_encoder=query_encoder,
        device=device,
        random_state=random_state,
    )
    policy_learner = PolicyLearner(
        model=policy,
        action_list=dataset.action_list,
        query_embeddings=dataset.query_embeddings,
        prompt_embeddings=dataset.prompt_embeddings,
        prompt_reward_predictor=prompt_reward_predictor,
        query_encoder=query_encoder,
        sentence_encoder=sentence_encoder,
        optimizer=optimizer,
        optimizer_kwargs={"lr": lr, "weight_decay": 0.0},
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
            use_wandb=use_wandb,
            experiment_name=f"movielens-model-D{base_random_state}",
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
            use_wandb=use_wandb,
            experiment_name=f"movielens-IS-D{base_random_state}",
        )
    elif gradient_type == "hybrid":
        policy, learning_process = policy_learner.hybrid_policy_gradient(
            logged_feedback=logged_feedback,
            return_training_logs=True,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            n_epochs_per_log=n_epochs_per_log,
            save_path=save_path,
            use_wandb=use_wandb,
            experiment_name=f"movielens-hybrid-D{base_random_state}",
        )
    with open(save_path_logs, "wb") as f:
        pickle.dump(learning_process, f)


def train_and_save_dso_policy(
    dataset: SemiSyntheticDataset,
    logged_feedback: Dict[str, Any],
    kernel_marginal_estimator: KernelMarginalDensityEstimator,
    query_encoder: BaseEncoder,
    sentence_encoder: BaseEncoder,
    gradient_type: str,
    n_epochs: int,
    n_steps_per_epoch: int,
    n_epochs_per_log: int,
    lr: float,
    optimizer: Optimizer,
    save_path: Path,
    save_path_logs: str,
    device: str = "cuda",
    random_state: Optional[int] = None,
    base_random_state: Optional[int] = None,
    use_wandb: bool = False,
):
    policy = PromptPolicy(
        n_actions=dataset.n_actions,
        dim_context=dataset.dim_context,
        query_encoder=query_encoder,
        device=device,
        random_state=random_state,
    )
    policy_learner = KernelPolicyLearner(
        model=policy,
        action_list=dataset.action_list,
        query_embeddings=dataset.query_embeddings,
        prompt_embeddings=dataset.prompt_embeddings,
        kernel_marginal_estimator=kernel_marginal_estimator,
        frozen_llm=dataset.frozen_llm,
        query_encoder=query_encoder,
        sentence_encoder=sentence_encoder,
        optimizer=optimizer,
        optimizer_kwargs={"lr": lr, "weight_decay": 0.0},
        env=dataset,
        random_state=random_state,
    )
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
        use_wandb=use_wandb,
        experiment_name=f"movielens-DSO-D{base_random_state}",
    )
    with open(save_path_logs, "wb") as f:
        pickle.dump(learning_process, f)


def train_and_save_two_stage_policy(
    dataset: SemiSyntheticDataset,
    logged_feedback: Dict[str, Any],
    prompt_reward_predictor: PromptRewardPredictor,
    query_encoder: BaseEncoder,
    prompt_encoder: BaseEncoder,
    sentence_encoder: BaseEncoder,
    gradient_type: str,
    n_clusters: int,
    n_epochs: int,
    n_steps_per_epoch: int,
    n_epochs_per_log: int,
    lr: float,
    optimizer: Optimizer,
    save_path: Path,
    save_path_logs: str,
    device: str = "cuda",
    random_state: Optional[int] = None,
    base_random_state: Optional[int] = None,
    use_wandb: bool = False,
):
    second_stage_policy = EpsilonGreedyPolicy(
        action_list=dataset.action_list,
        base_model=prompt_reward_predictor,
        device=device,
        random_state=random_state,
    )
    first_stage_policy = ClusterPolicy(
        n_actions=n_clusters,
        dim_context=dataset.dim_context,
        query_encoder=query_encoder,
        cluster_center_encoder=prompt_encoder,
        device=device,
        random_state=random_state,
    )
    clustering_policy = KmeansPromptClustering(
        n_clusters=n_clusters,
        action_list=dataset.action_list,
        prompt_encoder=prompt_encoder,
        device=device,
        random_state=random_state,
    )

    policy_learner = PolicyLearner(
        model=first_stage_policy,
        second_stage_policy=second_stage_policy,
        clustering_policy=clustering_policy,
        action_list=dataset.action_list,
        query_embeddings=dataset.query_embeddings,
        prompt_embeddings=dataset.prompt_embeddings,
        prompt_reward_predictor=prompt_reward_predictor,
        query_encoder=query_encoder,
        sentence_encoder=sentence_encoder,
        optimizer=optimizer,
        optimizer_kwargs={"lr": lr, "weight_decay": 0.0},
        env=dataset,
        random_state=random_state,
    )
    first_stage_policy, learning_process = policy_learner.hybrid_policy_gradient(
        logged_feedback=logged_feedback,
        return_training_logs=True,
        n_epochs=n_epochs,
        n_steps_per_epoch=n_steps_per_epoch,
        n_epochs_per_log=n_epochs_per_log,
        save_path=save_path,
        use_wandb=use_wandb,
        experiment_name=f"movielens-POTEC-D{base_random_state}",
    )
    with open(save_path_logs, "wb") as f:
        pickle.dump(learning_process, f)


def load_online_policy(
    dataset: SemiSyntheticDataset,
    query_encoder: BaseEncoder,
    prompt_encoder: BaseEncoder,
    sentence_encoder: BaseEncoder,
    save_path: Path,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    prompt_reward_predictor = PromptRewardPredictor(
        dim_context=dataset.dim_context,
        action_list=dataset.action_list,
        query_encoder=query_encoder,
        prompt_encoder=prompt_encoder,
        device=device,
        random_state=random_state,
    )
    policy = PromptPolicy(
        n_actions=dataset.n_actions,
        dim_context=dataset.dim_context,
        query_encoder=query_encoder,
        device=device,
        random_state=random_state,
    )
    policy_learner = PolicyLearner(
        model=policy,
        action_list=dataset.action_list,
        query_embeddings=dataset.query_embeddings,
        prompt_embeddings=dataset.prompt_embeddings,
        prompt_reward_predictor=prompt_reward_predictor,
        query_encoder=query_encoder,
        sentence_encoder=sentence_encoder,
        env=dataset,
        random_state=random_state,
    )
    policy = policy_learner.load(
        path=save_path,
        is_init=True,
    )
    return policy


def load_single_stage_policy(
    dataset: SemiSyntheticDataset,
    prompt_reward_predictor: PromptRewardPredictor,
    query_encoder: BaseEncoder,
    sentence_encoder: BaseEncoder,
    save_path: Path,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    policy = PromptPolicy(
        n_actions=dataset.n_actions,
        dim_context=dataset.dim_context,
        query_encoder=query_encoder,
        device=device,
        random_state=random_state,
    )
    policy_learner = PolicyLearner(
        model=policy,
        action_list=dataset.action_list,
        query_embeddings=dataset.query_embeddings,
        prompt_embeddings=dataset.prompt_embeddings,
        prompt_reward_predictor=prompt_reward_predictor,
        query_encoder=query_encoder,
        sentence_encoder=sentence_encoder,
        env=dataset,
        random_state=random_state,
    )
    policy = policy_learner.load(
        path=save_path,
        is_init=True,
    )
    return policy


def load_dso_policy(
    dataset: SemiSyntheticDataset,
    kernel_marginal_estimator: KernelMarginalDensityEstimator,
    query_encoder: BaseEncoder,
    sentence_encoder: BaseEncoder,
    save_path: Path,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    policy = PromptPolicy(
        n_actions=dataset.n_actions,
        dim_context=dataset.dim_context,
        query_encoder=query_encoder,
        device=device,
        random_state=random_state,
    )
    policy_learner = KernelPolicyLearner(
        model=policy,
        action_list=dataset.action_list,
        query_embeddings=dataset.query_embeddings,
        prompt_embeddings=dataset.prompt_embeddings,
        kernel_marginal_estimator=kernel_marginal_estimator,
        frozen_llm=dataset.frozen_llm,
        query_encoder=query_encoder,
        sentence_encoder=sentence_encoder,
        env=dataset,
        random_state=random_state,
    )
    policy = policy_learner.load(
        path=save_path,
        is_init=True,
    )
    return policy


def load_two_stage_policy(
    dataset: SemiSyntheticDataset,
    prompt_reward_predictor: PromptRewardPredictor,
    query_encoder: BaseEncoder,
    prompt_encoder: BaseEncoder,
    sentence_encoder: BaseEncoder,
    n_clusters: int,
    save_path: Path,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    second_stage_policy = EpsilonGreedyPolicy(
        action_list=dataset.action_list,
        base_model=prompt_reward_predictor,
        device=device,
        random_state=random_state,
    )
    first_stage_policy = ClusterPolicy(
        n_actions=n_clusters,
        dim_context=dataset.dim_context,
        query_encoder=query_encoder,
        cluster_center_encoder=prompt_encoder,
        device=device,
        random_state=random_state,
    )
    clustering_policy = KmeansPromptClustering(
        n_clusters=n_clusters,
        action_list=dataset.action_list,
        prompt_encoder=prompt_encoder,
        device=device,
        random_state=random_state,
    )

    policy_learner = PolicyLearner(
        model=first_stage_policy,
        second_stage_policy=second_stage_policy,
        clustering_policy=clustering_policy,
        action_list=dataset.action_list,
        query_embeddings=dataset.query_embeddings,
        prompt_embeddings=dataset.prompt_embeddings,
        prompt_reward_predictor=prompt_reward_predictor,
        query_encoder=query_encoder,
        sentence_encoder=sentence_encoder,
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


def load_uniform_policy(
    dataset: SemiSyntheticDataset,
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
    dataset: SemiSyntheticDataset,
    policy: Policy,
):
    policy_value = dataset.calc_expected_policy_value(
        policy=policy,
        n_samples_to_approximate=10000,
    )
    return policy_value


def off_policy_evaluation(
    dataset: SemiSyntheticDataset,
    logged_feedback: Dict[str, Any],
    policy: Policy,
    estimator_name: str,
    n_clusters: int,
    prompt_reward_predictor: Optional[PromptRewardPredictor] = None,
    kernel_marginal_estimator: Optional[KernelMarginalDensityEstimator] = None,
    query_encoder: Optional[BaseEncoder] = None,
    prompt_encoder: Optional[BaseEncoder] = None,
    random_state: Optional[int] = None,
):
    if estimator_name == "OffCEM":
        clustering_policy = KmeansPromptClustering(
            n_clusters=n_clusters,
            action_list=dataset.action_list,
            prompt_encoder=prompt_encoder,
            device=prompt_reward_predictor.device,
            random_state=random_state,
        )
    else:
        clustering_policy = None

    ope = PolicyEvaluator(
        env=dataset,
        prompt_reward_predictor=prompt_reward_predictor,
        clustering_policy=clustering_policy,
        action_list=dataset.action_list,
        query_embeddings=dataset.query_embeddings,
        prompt_embeddings=dataset.prompt_embeddings,
        query_encoder=query_encoder,
        frozen_llm=dataset.frozen_llm,
        random_state=random_state,
    )

    if estimator_name == "regression-based":
        estimation = ope.regression_based_policy_evaluation(
            logged_feedback=logged_feedback,
            eval_policy=policy,
        )
    elif estimator_name == "IS-based":
        estimation = ope.importance_sampling_based_policy_evaluation(
            logged_feedback=logged_feedback,
            eval_policy=policy,
        )
    elif estimator_name == "hybrid":
        estimation = ope.hybrid_policy_evaluation(
            logged_feedback=logged_feedback,
            eval_policy=policy,
        )
    elif estimator_name == "OffCEM":
        estimation = ope.offcem_policy_evaluation(
            logged_feedback=logged_feedback,
            eval_policy=policy,
        )
    elif estimator_name == "kernelIS":
        estimation = ope.kernel_IS_based_policy_evaluation(
            logged_feedback=logged_feedback,
            eval_policy=policy,
            logging_kernel_density_model=kernel_marginal_estimator,
        )
    return estimation


def get_baseline_performance(
    dataset: SemiSyntheticDataset,
):
    policy_value = dataset.calc_expected_policy_value(
        policy=None,
        n_samples_to_approximate=10000,
    )
    return policy_value


def get_skyline_performance(
    dataset: SemiSyntheticDataset,
):
    policy_value = dataset.calc_skyline_policy_value(
        n_samples_to_approximate=10,
    )
    return policy_value
