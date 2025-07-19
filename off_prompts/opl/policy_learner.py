"""Learner class for the first and second stage policies."""
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Union, Any, Dict
from pathlib import Path
from tqdm.auto import tqdm
from operator import itemgetter
import random
import wandb

import torch
from torch.optim import Optimizer, Adam
from torch.utils.data import RandomSampler, DataLoader
from accelerate import Accelerator

from ..dataset.benchmark import SemiSyntheticDataset
from ..dataset.base import BaseFrozenLLM, BaseEncoder
from ..policy.base import (
    BasePolicy,
    BasePromptPolicyModel,
    BaseClusterPolicyModel,
    BaseClusteringModel,
    BasePromptRewardModel,
    BaseSentenceRewardModel,
    BaseKernelMarginalDensityModel,
)
from ..policy.policy import TwoStagePolicy
from ..types import LoggedDataset, Sentence
from ..utils import check_logged_feedback, torch_seed, tokenize, to_device

from .dataset import TorchLoggedDataset

Policy = Union[BasePolicy, BasePromptPolicyModel, BaseClusterPolicyModel]


@dataclass
class PolicyLearner:
    model: Union[BasePromptPolicyModel, BaseClusterPolicyModel]
    action_list: Sentence
    query_embeddings: Optional[torch.Tensor] = None
    prompt_embeddings: Optional[torch.Tensor] = None
    second_stage_policy: Optional[Policy] = None
    clustering_policy: Optional[BaseClusteringModel] = None
    prompt_reward_predictor: Optional[BasePromptRewardModel] = None
    query_encoder: Optional[BaseEncoder] = None
    sentence_encoder: Optional[BaseEncoder] = None
    optimizer: Optimizer = Adam
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    env: Optional[SemiSyntheticDataset] = None
    random_state: Optional[int] = None

    """Learner class for action policy.

    Imported as: :class:`off_prompts.opl.PolicyLearner`

    Parameters
    -------
    model: BasePromptPolicyModel or BaseClusterPolicyModel
        Action or cluster policy (either first stage or single stage policy) to train.

    action_list: Sentence
        Mapping from action id to discrete prompts.

    query_embeddings: torch.Tensor, shape (n_items, dim_query_emb), default=None
        Mapping from item id to its (query) embeddings.

    prompt_embeddings: torch.Tensor, shape (n_items, dim_prompt_emb), default=None
        Mapping from item id to its (prompt) embeddings.

    second_stage_policy: Policy, default=None
        (Pre-trained) second stage policy. This must be given when training a first stage policy.

    clustering_policy: BaseClusteringModel, default=None
        (Pre-trained) clustering policy that determines the action clustering for each context. This must be given when training a first stage policy.

    prompt_reward_predictor: BasePromptRewardModel, default=None
        (Pre-trained) action reward predictor. This must be given when using a model-based or hybrid appraoch to train a policy.

    query_encoder: BaseEncoder, default=None
        Encoder of query.

    sentence_encoder: BaseEncoder, default=None
        Encoder of sentence.

    optimizer: torch.optim.Optimizer, default=Adam
        Class of optimizer (not an instance).

    optimizer_kwargs: dict, default=None
        Arguments of optimizer.

    env: SyntheticDataset, default=None
        Online environment for evaluation.

    random_state: int, default=None
        Random state.
    
    """

    def __post_init__(self):
        self.trained_model = None
        self.n_actions = len(self.action_list)

        if self.second_stage_policy is not None:
            if self.clustering_policy is None:
                raise ValueError(
                    "clustering_policy must be given when using two-stage policy."
                )
            if self.clustering_policy.n_clusters != self.model.n_actions:
                raise RuntimeError(
                    "Expected clustering_policy.n_clusters == model.n_actions (number of output of the first stage policy), but found False."
                )
            if not (
                self.clustering_policy.n_actions
                == self.second_stage_policy.n_actions
                == self.n_actions
            ):
                raise RuntimeError(
                    "Expected clustering_policy.n_actions == second_stage_policy.n_actions == len(action_list), but found False."
                )

        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {
                "lr": 1e-4,
                "momentum": 0.9,
            }

        if self.random_state is None:
            raise ValueError("random_state must be given")

        self.device = self.model.device

    def load(self, path: Path, is_init: bool = False):
        """Load model."""
        if is_init:
            self.model.load_state_dict(torch.load(path))
            model = self.model
        else:
            self.trained_model.load_state_dict(torch.load(path))
            model = self.trained_model

        return model

    def save(self, path: Path):
        """Save Model."""
        torch.save(self.trained_model.state_dict(), path)

    def seed(self, random_state: int):
        """Fix seed."""
        random.seed(random_state)
        torch_seed(random_state)

    def _init_policy_with_uniform_random(
        self,
        model: Union[BasePromptPolicyModel, BaseClusterPolicyModel],
        optimizer: Optimizer,
        dataloader: Optional[DataLoader] = None,
        accelerator: Optional[Accelerator] = None,
        batch_size: int = 32,
        n_steps_for_initialization: int = 1000,
        is_two_stage_policy: bool = False,
    ):
        """Initialize policy with a uniform random policy."""
        uniform_prob = 1 / self.n_actions

        if dataloader is not None:
            data_iterator = iter(dataloader)

        for j in range(n_steps_for_initialization):
            if self.env is not None:
                (
                    user_id_,
                    item_id_,
                    context_,
                    query_,
                    query_embeddings_,
                ) = self.env.context_query_loader.sample_context_and_query(
                    n_samples=batch_size,
                    return_query_embeddings=True,
                )

            elif dataloader is not None:
                try:
                    batch_ = next(data_iterator)
                except StopIteration:
                    # Reinitialize the iterator if the dataset is exhausted before reaching n_steps_per_epoch
                    data_iterator = iter(dataloader)
                    batch_ = next(data_iterator)

                context_ = to_device(batch_["context"], device=accelerator.device)
                query_ = to_device(batch_["query"], device=accelerator.device)

            else:
                raise ValueError("dataloader is not give.")

            if is_two_stage_policy:
                cluster_centers_ = self.clustering_policy.retrieve_cluster_centers(
                    context=context_,
                    query=query_,
                    resample_clustering=True,
                )
                cluster_ = model.sample_action(
                    context=context_,
                    query=query_embeddings_,
                    cluster_centers=cluster_centers_,
                    return_cpu_tensor=False,
                )
                prob_ = model.calc_prob_given_action(
                    context=context_,
                    query=query_embeddings_,
                    action=cluster_,
                    cluster_centers=cluster_centers_,
                    return_cpu_tensor=False,
                    is_log_prob=False,
                    calc_gradient=True,
                )
            else:
                action_ = model.sample_action(
                    context=context_,
                    query=query_embeddings_,
                    return_cpu_tensor=False,
                )
                prob_ = model.calc_prob_given_action(
                    context=context_,
                    query=query_embeddings_,
                    action=action_,
                    return_cpu_tensor=False,
                    is_log_prob=False,
                    calc_gradient=True,
                )

            # to get uniform random probability
            loss_ = torch.square(prob_ - uniform_prob).sum()

            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()

        return model

    def online_policy_gradient(
        self,
        logging_action_choice_prob: Optional[torch.Tensor] = None,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        n_epochs: int = 1000,
        n_steps_per_epoch: int = 10,
        n_epochs_per_log: int = 10,
        batch_size: int = 32,
        make_copy: bool = False,
        return_training_logs: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
        use_wandb: bool = False,
        experiment_name: str = "online",
    ):
        """Train policy in an online manner.

        Note
        -------
        Online policy gradient approximates the policy gradient as follows.

        .. math::

            \\nabla_{\\theta} V(\\pi_{\\theta})
            \\approx \\frac{1}{m} \\sum_{i=1}^m \\mathbb{E}_{x_i \\sim p(x), a_i \\sim \\pi_{\\theta}(a | x_i), r_i \\sim p(r|x_i,a_i)}
            \\left[ \\nabla_{\\theta} \\log \\pi_{\\theta}(a_i | x_i) r_i \\right]

        where we parametrize the policy as :math:`\\pi_{\\theta}` using some parameters :math:`\\theta \\in \\Theta` (e.g., a neural network).
        :math:`x` is the context, :math:`a` is the action, and :math:`r` is the reward.
        :math:`m` is the number of batched samples used in Monte-Carlo sampling.

        References
        -------
        Richard S Sutton and Andrew G Barto.
        "Reinforcement learning: An introduction." 2018.

        Mingkai Deng, Jianyu Wang, Cheng-Ping Hsieh, Yihan Wang, Han Guo, Tianmin Shu, Meng Song, Eric Xing, and Zhiting Hu.
        "RLPrompt: Optimizing discrete text prompts with reinforcement learning." 2022.

        Parameters
        -------
        logging_action_choice_prob: torch.Tensor, shape (n_samples, n_actions), default=None
            For API consistency.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            For API consistency.

        n_epochs: int, default=1000
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=10
            Number of gradient steps within an epoch.

        n_epochs_per_log: int, default=10
            Number of epochs in the logging interval.

        batch_size: int, default=32
            Batch size.

        make_copy: bool, default=False
            Whether to create copy of the model before training.

        return_training_logs: bool, default=False
            Whether to return (true) policy value at each training epoch.

        save_path: Path, default=None
            Path to save the model.

        random_state: int, default=None
            Random state.

        use_wandb: bool, default=False
            Whether to use wandb to report the training statistics.

        experiment_name: str, default="online"
            Experiment name used for wandb reports.

        Return
        -------
        model: Policy
            Trained policy.

        """
        if self.env is None:
            raise RuntimeError(
                "self.env must be given when training a policy online. Please initialize the class with env."
            )

        if random_state is None:
            random_state = self.random_state
        self.seed(random_state)

        if make_copy:
            model = deepcopy(self.model)
        else:
            model = self.model

        optimizer = self.optimizer(model.parameters(), **self.optimizer_kwargs)

        is_two_stage_policy = self.second_stage_policy is not None

        model = self._init_policy_with_uniform_random(
            model=model,
            optimizer=optimizer,
            is_two_stage_policy=is_two_stage_policy,
        )

        train_losses = torch.zeros((n_epochs + 1,))
        policy_values = torch.zeros((n_epochs // n_epochs_per_log + 1,))

        if use_wandb:
            wandb.init(entity="", project=experiment_name)

        if is_two_stage_policy:
            eval_policy = TwoStagePolicy(
                first_stage_policy=model,
                second_stage_policy=self.second_stage_policy,
                clustering_policy=self.clustering_policy,
                device=self.device,
            )
        else:
            eval_policy = model

        if self.env is not None:
            policy_values[0] = self.env.calc_expected_policy_value(
                eval_policy,
                n_samples_to_approximate=1000,
            )
            policy_value = policy_values[0].item()
        else:
            policy_value = torch.zeros((1,))

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train policy online: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "loss": f"{train_losses[i]:.4g}",
                        "policy_value": f"{policy_value:.4g}",
                    }
                )

                for j in range(n_steps_per_epoch):
                    (
                        user_id_,
                        item_id_,
                        context_,
                        query_,
                        query_embeddings_,
                    ) = self.env.context_query_loader.sample_context_and_query(
                        n_samples=batch_size,
                        return_query_embeddings=True,
                    )

                    if is_two_stage_policy:
                        cluster_ = model.sample_action(
                            context=context_,
                            query=query_embeddings_,
                            return_cpu_tensor=False,
                        )
                        log_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_embeddings_,
                            action=cluster_,
                            return_cpu_tensor=False,
                            is_log_prob=True,
                            calc_gradient=True,
                        )
                        prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_embeddings_,
                            action=cluster_,
                            return_cpu_tensor=False,
                            is_log_prob=False,
                            calc_gradient=False,
                        )
                        candidate_actions = (
                            self.clustering_policy.retrieve_cluster_center(
                                context=context_,
                                query=query_embeddings_,
                                cluster=cluster_,
                            )
                        )
                        action_ = self.second_stage_policy.sample_action(
                            context=context_,
                            query=query_embeddings_,
                            candidate_actions=candidate_actions,
                        )
                    else:
                        action_ = model.sample_action(
                            context=context_,
                            query=query_embeddings_,
                            return_cpu_tensor=False,
                        )
                        log_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_embeddings_,
                            action=action_,
                            return_cpu_tensor=False,
                            is_log_prob=True,
                            calc_gradient=True,
                        )
                        prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_embeddings_,
                            action=action_,
                            return_cpu_tensor=False,
                            is_log_prob=False,
                            calc_gradient=False,
                        )

                    reward_ = self.env.sample_reward_given_action(
                        user_id=user_id_,
                        item_id=item_id_,
                        context=context_,
                        query=query_,
                        action=action_,
                        return_cpu_tensor=False,
                    )

                    loss_ = -(log_prob_ * reward_).sum()

                    optimizer.zero_grad()
                    loss_.backward()
                    optimizer.step()

                    train_losses[i + 1] += loss_.item() // n_steps_per_epoch

                if (i + 1) % n_epochs_per_log == 0:
                    if is_two_stage_policy:
                        eval_policy.first_stage_policy = model

                    if self.env is not None:
                        policy_values[
                            (i + 1) // n_epochs_per_log
                        ] = self.env.calc_expected_policy_value(
                            eval_policy,
                            n_samples_to_approximate=1000,
                        )
                        policy_value = policy_values[(i + 1) // n_epochs_per_log].item()
                        # print(policy_value)

                if use_wandb:
                    wandb.log(
                        {
                            "train_loss": train_losses[i + 1],
                            "policy_value": policy_value,
                            "average_prob": prob_.mean(),
                        }
                    )

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        if return_training_logs:
            output = (model, policy_values)
        else:
            output = model

        return output

    def model_based_policy_gradient(
        self,
        logged_feedback: LoggedDataset,
        logging_action_choice_prob: Optional[torch.Tensor] = None,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        n_epochs: int = 1000,
        n_steps_per_epoch: int = 10,
        n_epochs_per_log: int = 10,
        batch_size: int = 32,
        make_copy: bool = False,
        return_training_logs: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
        use_wandb: bool = False,
        experiment_name: str = "model-based",
    ):
        """Train policy from logged data via the model-based approach.

        Note
        -------
        Model-based (regression-based) policy gradient estimates the policy gradient as follows.

        .. math::

            \\nabla_{\\theta} V(\\pi_{\\theta})
            \\approx \\frac{1}{n} \\sum_{i=1}^n \\mathbb{E}_{a \\sim \\pi_{\\theta}(a | x_i)}
            \\left[ \\nabla_{\\theta} \\log \\pi_{\\theta}(a | x_i) \\hat{q}(x_i, a) \\right]

        where we parametrize the policy as :math:`\\pi_{\\theta}` using some parameters :math:`\\theta \\in \\Theta` (e.g., a neural network).
        :math:`x` is the context, :math:`a` is the action, and :math:`r` is the reward.
        :math:`\\hat{q}(x, a) \\approx \\mathbb{E}[r|x,a]` is the predicted reward given context and action. :math:`n` is the number of the data sample.
        Note that we approximate the expectation :math:`\\mathbb{E}[\\cdot]` using a single monte-carlo sample for each index of data, :math:`i`.

        References
        -------
        Vijay Konda and John Tsitsiklis.
        "Actor-critic algorithms." 1999.

        Parameters
        -------
        logged_feedback: LoggedDataset
            Logged data, which contains the following keys.

            .. code-block:: python

                    key: [
                        context,          # (n_samples, )
                        query,            # (n_samples, )
                        user_id,          #  -
                        item_id,          # (n_samples, )
                        action,           #  -
                        output,           #  -
                        reward,           #  -
                        logging_policy,   #  -
                    ]

            See dataset/synthetic.py for the details of each key.

        logging_action_choice_prob: torch.Tensor, shape (n_samples, n_actions), default=None
            For API consistency.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions used for the logging policy in the clustering policy.

        n_epochs: int, default=1000
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=10
            Number of gradient steps within an epoch.

        n_epochs_per_log: int, default=10
            Number of epochs in the logging interval.

        batch_size: int, default=32
            Batch size.

        make_copy: bool, default=False
            Whether to create copy of the model before training.

        return_training_logs: bool, default=False
            Whether to return (true) policy value at each training epoch.

        save_path: Path, default=None
            Path to save the model.

        random_state: int, default=None
            Random state.

        use_wandb: bool, default=False
            Whether to use wandb to report the training statistics.

        experiment_name: str, default="model-based"
            Experiment name used for wandb reports.

        Return
        -------
        model: Policy
            Trained policy.

        """
        if self.prompt_reward_predictor is None:
            raise RuntimeError(
                "prompt_reward_predictor is not provided. Please initialize the class with prompt_reward_predictor."
            )
        if return_training_logs and self.env is None:
            raise RuntimeError(
                "return_training_logs option is feasible when self.env is given. Please initialize the class with env."
            )

        # check_logged_feedback(logged_feedback)

        if random_state is None:
            random_state = self.random_state
        self.seed(random_state)

        if make_copy:
            model = deepcopy(self.model)
        else:
            model = self.model

        optimizer = self.optimizer(model.parameters(), **self.optimizer_kwargs)

        is_two_stage_policy = self.second_stage_policy is not None

        context = logged_feedback["context"]
        query = logged_feedback["query"]
        item_id = logged_feedback["item_id"]
        action = logged_feedback["action"]

        train_dataset = TorchLoggedDataset(
            context=context,
            query=query,
            item_id=item_id,
            action=action,
            query_embeddings=self.query_embeddings,
            query_encoder=self.query_encoder,
        )
        accelerator = Accelerator()

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        model, train_dataloader = accelerator.prepare(model, train_dataloader)

        model = self._init_policy_with_uniform_random(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            accelerator=accelerator,
            is_two_stage_policy=is_two_stage_policy,
        )

        train_size = len(context)
        train_losses = torch.zeros((n_epochs + 1,))
        policy_values = torch.zeros((n_epochs // n_epochs_per_log + 1,))

        if use_wandb:
            wandb.init(entity="", project=experiment_name)

        if is_two_stage_policy:
            eval_policy = TwoStagePolicy(
                first_stage_policy=model,
                second_stage_policy=self.second_stage_policy,
                clustering_policy=self.clustering_policy,
                device=self.device,
            )
        else:
            eval_policy = model

        if self.env is not None:
            policy_values[0] = self.env.calc_expected_policy_value(
                eval_policy,
                n_samples_to_approximate=1000,
            )
            policy_value = policy_values[0].item()
        else:
            policy_value = torch.zeros((1,))

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train policy: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "loss": f"{train_losses[i]:.4g}",
                        "policy_value": f"{policy_value:.4g}",
                    }
                )

                train_iterator = iter(train_dataloader)

                for j in range(n_steps_per_epoch):
                    try:
                        batch_ = next(train_iterator)
                    except StopIteration:
                        # Reinitialize the iterator if the dataset is exhausted before reaching n_steps_per_epoch
                        train_iterator = iter(train_dataloader)
                        batch_ = next(train_iterator)

                    context_ = to_device(batch_["context"], device=accelerator.device)
                    query_ = to_device(batch_["query"], device=accelerator.device)
                    logged_action_ = to_device(
                        batch_["action"], device=accelerator.device
                    )

                    if is_two_stage_policy:
                        cluster_centers_ = (
                            self.clustering_policy.retrieve_cluster_centers(
                                context=context_,
                                query=query_,
                                resample_clustering=True,
                            )
                        )
                        cluster_ = model.sample_action(
                            context=context_,
                            query=query_,
                            cluster_centers=cluster_centers_,
                            return_cpu_tensor=False,
                        )
                        log_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            cluster_centers=cluster_centers_,
                            action=cluster_,
                            return_cpu_tensor=False,
                            is_log_prob=True,
                            calc_gradient=True,
                        )
                        candidate_actions_ = (
                            self.clustering_policy.retrieve_candidate_actions(
                                context=context_,
                                query=query_,
                                cluster=cluster_,
                            )
                        )
                        predicted_reward_ = (
                            self.second_stage_policy.predict_policy_value(
                                context=context_,
                                query=query_,
                                candidate_actions=candidate_actions_,
                                return_per_sample=True,
                            )
                        )

                        prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            cluster_centers=cluster_centers_,
                            action=cluster_,
                            return_cpu_tensor=False,
                            is_log_prob=False,
                            calc_gradient=False,
                        )
                        imitation_prob_ = torch.zeros((1,))

                    else:
                        action_ = model.sample_action(
                            context=context_,
                            query=query_,
                            return_cpu_tensor=False,
                        )
                        log_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=action_,
                            return_cpu_tensor=False,
                            is_log_prob=True,
                            calc_gradient=True,
                        )
                        predicted_reward_ = self.prompt_reward_predictor.predict_value(
                            context=context_,
                            query=query_,
                            action=action_,
                            return_cpu_tensor=False,
                        )

                        prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=action_,
                            return_cpu_tensor=False,
                            is_log_prob=False,
                            calc_gradient=False,
                        )
                        imitation_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=logged_action_,
                            return_cpu_tensor=False,
                            is_log_prob=False,
                            calc_gradient=False,
                        )

                    train_loss_ = -(predicted_reward_ * log_prob_).mean()

                    optimizer.zero_grad()
                    accelerator.backward(train_loss_)
                    optimizer.step()

                    train_losses[i + 1] += train_loss_.item() / n_steps_per_epoch

                if (i + 1) % n_epochs_per_log == 0:
                    if is_two_stage_policy:
                        eval_policy.first_stage_policy = model

                    if self.env is not None:
                        policy_values[
                            (i + 1) // n_epochs_per_log
                        ] = self.env.calc_expected_policy_value(
                            eval_policy,
                            n_samples_to_approximate=1000,
                        )
                        policy_value = policy_values[(i + 1) // n_epochs_per_log].item()
                        # print(policy_value)

                if use_wandb:
                    wandb.log(
                        {
                            "train_loss": train_losses[i + 1],
                            "policy_value": policy_value,
                            "average_imitation": imitation_prob_.mean(),
                            "average_prob": prob_.mean(),
                        }
                    )

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        if return_training_logs:
            output = (model, policy_values)
        else:
            output = model

        return output

    def importance_sampling_based_policy_gradient(
        self,
        logged_feedback: LoggedDataset,
        logging_action_choice_prob: Optional[torch.Tensor] = None,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        clip_threshold: Union[float, int] = 200,
        n_epochs: int = 1000,
        n_steps_per_epoch: int = 10,
        n_epochs_per_log: int = 10,
        batch_size: int = 32,
        make_copy: bool = False,
        return_training_logs: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
        use_wandb: bool = False,
        experiment_name: str = "IS-based",
    ):
        """Train policy from logged data via the importance sampling-based approach.

        Note
        -------
        Importance sampling policy gradient estimates the policy gradient as follows.

        .. math::

            \\nabla_{\\theta} V(\\pi_{\\theta})
            \\approx \\frac{1}{n} \\sum_{i=1}^n \\frac{\\pi_{\\theta}(a_i | x_i)}{\\pi_0(a_i | x_i)} \\nabla_{\\theta} \\log \\pi_{\\theta}(a_i | x_i) r_i

        where we parametrize the policy as :math:`\\pi_{\\theta}` using some parameters :math:`\\theta \\in \\Theta` (e.g., a neural network).
        :math:`x` is the context, :math:`a` is the action (:math:`a_i` is chosen by the logging policy :math:`pi_0`), and :math:`r` is the reward. :math:`n` is the number of the data sample.

        References
        -------
        Adith Swaminathan and Thorsten Joachims.
        "Batch learning from logged bandit feedback through counterfactual risk minimization." 2015.

        Parameters
        -------
        logged_feedback: LoggedDataset
            Logged data, which contains the following keys.

            .. code-block:: python

                    key: [
                        context,          # (n_samples, )
                        query,            # (n_samples, )
                        user_id,          #  -
                        item_id,          # (n_samples, )
                        action,           # (n_samples, )
                        output,           #  -
                        reward,           # (n_samples, )
                        logging_policy,   #  -
                    ]

            See dataset/synthetic.py for the details of each key.

        logging_action_choice_prob: torch.Tensor, shape (n_samples, n_actions), default=None
            Action choice probability of logging policy to all candidate actions.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions used for the logging policy in the clustering policy.

        clip_threshold: float or int, default=200
            Threshold for clipping the importance weight.

        n_epochs: int, default=1000
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=10
            Number of gradient steps within an epoch.

        n_epochs_per_log: int, default=10
            Number of epochs in the logging interval.

        batch_size: int, default=32
            Batch size.

        make_copy: bool, default=False
            Whether to create copy of the model before training.

        return_training_logs: bool, default=False
            Whether to return (true) policy value at each training epoch.

        save_path: Path, default=None
            Path to save the model.

        random_state: int, default=None
            Random state.

        use_wandb: bool, default=False
            Whether to use wandb to report the training statistics.

        experiment_name: str, default="IS-based"
            Experiment name used for wandb reports.

        Return
        -------
        model: Policy
            Trained policy.

        """
        if return_training_logs and self.env is None:
            raise RuntimeError(
                "return_training_logs option is feasible when self.env is given. Please initialize the class with env."
            )

        # check_logged_feedback(logged_feedback)

        if random_state is None:
            random_state = self.random_state
        self.seed(random_state)

        if make_copy:
            model = deepcopy(self.model)
        else:
            model = self.model

        optimizer = self.optimizer(model.parameters(), **self.optimizer_kwargs)

        is_two_stage_policy = self.second_stage_policy is not None

        context = logged_feedback["context"]
        query = logged_feedback["query"]
        item_id = logged_feedback["item_id"]
        action = logged_feedback["action"]
        sentence = logged_feedback["sentence"]
        reward = logged_feedback["reward"]
        logging_policy = logged_feedback["logging_policy"]

        if logging_action_choice_prob is None:
            logging_action_choice_prob = logging_policy.calc_action_choice_probability(
                context=context,
                query=query,
                predicted_reward=logging_predicted_reward,
            )

        train_dataset = TorchLoggedDataset(
            context=context,
            query=query,
            item_id=item_id,
            action=action,
            sentence=sentence,
            reward=reward,
            logging_action_choice_prob=logging_action_choice_prob,
            query_embeddings=self.query_embeddings,
            query_encoder=self.query_encoder,
            sentence_encoder=self.sentence_encoder,
        )
        accelerator = Accelerator()

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        model, train_dataloader = accelerator.prepare(model, train_dataloader)

        model = self._init_policy_with_uniform_random(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            accelerator=accelerator,
            is_two_stage_policy=is_two_stage_policy,
        )

        train_losses = torch.zeros((n_epochs + 1,))
        policy_values = torch.zeros((n_epochs // n_epochs_per_log + 1,))

        if use_wandb:
            wandb.init(entity="", project=experiment_name)

        if is_two_stage_policy:
            eval_policy = TwoStagePolicy(
                first_stage_policy=model,
                second_stage_policy=self.second_stage_policy,
                clustering_policy=self.clustering_policy,
                device=self.device,
            )
        else:
            eval_policy = model

        if self.env is not None:
            policy_values[0] = self.env.calc_expected_policy_value(
                eval_policy,
                n_samples_to_approximate=1000,
            )
            policy_value = policy_values[0].item()
        else:
            policy_value = torch.zeros((1,))

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train policy: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "loss": f"{train_losses[i]:.4g}",
                        "policy_value": f"{policy_value:.4g}",
                    }
                )

                train_loss_logs = torch.zeros((n_steps_per_epoch,))

                train_iterator = iter(train_dataloader)

                for j in range(n_steps_per_epoch):
                    try:
                        batch_ = next(train_iterator)
                    except StopIteration:
                        # Reinitialize the iterator if the dataset is exhausted before reaching n_steps_per_epoch
                        train_iterator = iter(train_dataloader)
                        batch_ = next(train_iterator)

                    context_ = to_device(batch_["context"], device=accelerator.device)
                    query_ = to_device(batch_["query"], device=accelerator.device)
                    action_ = to_device(batch_["action"], device=accelerator.device)
                    sentence_ = to_device(batch_["sentence"], device=accelerator.device)
                    reward_ = to_device(batch_["reward"], device=accelerator.device)
                    logging_action_choice_prob_ = to_device(
                        batch_["logging_action_choice_prob"], device=accelerator.device
                    )
                    logging_pscore_ = logging_action_choice_prob_[
                        to_device(torch.arange(batch_size), device=accelerator.device),
                        action_,
                    ]

                    if is_two_stage_policy:
                        cluster_centers_ = (
                            self.clustering_policy.retrieve_cluster_centers(
                                context=context_,
                                query=query_,
                                resample_clustering=True,
                            )
                        )
                        cluster_ = self.clustering_policy.retrieve_cluster(
                            context=context_,
                            query=query_,
                            action=action_,
                            sentence=sentence_,
                        )
                        log_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=cluster_,
                            cluster_centers=cluster_centers_,
                            return_cpu_tensor=False,
                            is_log_prob=True,
                            calc_gradient=True,
                        )
                        evaluation_cluster_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=cluster_,
                            cluster_centers=cluster_centers_,
                            return_cpu_tensor=False,
                        )
                        logging_cluster_prob_ = (
                            self.clustering_policy.calc_cluster_choice_prob(
                                context=context_,
                                query=query_,
                                action_choice_prob=logging_action_choice_prob_,
                                cluster=cluster_,
                            )
                        )
                        iw_ = evaluation_cluster_prob_ / logging_cluster_prob_
                        iw_ = torch.nan_to_num(iw_)  # avoid zero division

                        prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=cluster_,
                            cluster_centers=cluster_centers_,
                            return_cpu_tensor=False,
                            is_log_prob=False,
                            calc_gradient=False,
                        )
                        imitation_prob_ = torch.zeros((1,))

                    else:
                        log_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=action_,
                            return_cpu_tensor=False,
                            is_log_prob=True,
                            calc_gradient=True,
                        )
                        evaluation_pscore_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=action_,
                            return_cpu_tensor=False,
                        )
                        iw_ = evaluation_pscore_ / logging_pscore_

                        _, prob_ = model.sample_action_and_output_prob(
                            context=context_,
                            query=query_,
                            return_cpu_tensor=False,
                            is_log_prob=False,
                            calc_gradient=False,
                        )
                        imitation_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=action_,
                            return_cpu_tensor=False,
                            is_log_prob=False,
                            calc_gradient=False,
                        )

                    iw_ = torch.clip(iw_, max=clip_threshold)
                    train_loss_ = -(iw_ * log_prob_ * reward_).mean()

                    optimizer.zero_grad()
                    accelerator.backward(train_loss_)
                    optimizer.step()

                    train_losses[i + 1] += train_loss_.item() / n_steps_per_epoch
                    train_loss_logs[j] = train_loss_.item()

                if (i + 1) % n_epochs_per_log == 0:
                    if is_two_stage_policy:
                        eval_policy.first_stage_policy = model

                    if self.env is not None:
                        policy_values[
                            (i + 1) // n_epochs_per_log
                        ] = self.env.calc_expected_policy_value(
                            eval_policy,
                            n_samples_to_approximate=1000,
                        )
                        policy_value = policy_values[(i + 1) // n_epochs_per_log].item()
                        # print(policy_value)
                        # print(train_loss_logs.std().item() / batch_size)
                        # print()

                if use_wandb:
                    wandb.log(
                        {
                            "train_loss": train_losses[i + 1],
                            "policy_value": policy_value,
                            "average_iw": iw_.mean(),
                            "average_imitation": imitation_prob_.mean(),
                            "average_prob": prob_.mean(),
                        }
                    )

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        if return_training_logs:
            output = (model, policy_values)
        else:
            output = model

        return output

    def hybrid_policy_gradient(
        self,
        logged_feedback: LoggedDataset,
        logging_action_choice_prob: Optional[torch.Tensor] = None,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        clip_threshold: Union[float, int] = 200,
        n_epochs: int = 1000,
        n_steps_per_epoch: int = 10,
        n_epochs_per_log: int = 10,
        batch_size: int = 32,
        make_copy: bool = False,
        return_training_logs: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
        use_wandb: bool = False,
        experiment_name: str = "hybrid",
    ):
        """Train policy from logged data via the hybrid approach.

        Note
        -------
        Doubly robust policy gradient estimates the policy gradient as follows.

        .. math::

            \\nabla_{\\theta} V(\\pi_{\\theta})
            \\approx \\frac{1}{n} \\sum_{i=1}^n \\frac{\\pi_{\\theta}(a_i | x_i)}{\\pi_0(a_i | x_i)} \\nabla_{\\theta} \\log \\pi_{\\theta}(a_i | x_i) (r_i - \\hat{q}(x_i, a_i))
            + \\frac{1}{n} \\sum_{i=1}^n \\mathbb{E}_{a \\sim \\pi_{\\theta}(a|x_i)}[\\nabla_{\\theta} \\log \\pi_{\\theta}(a_i | x_i) \\hat{q}(x_i, a)]

        where we parametrize the policy as :math:`\\pi_{\\theta}` using some parameters :math:`\\theta \\in \\Theta` (e.g., a neural network).
        :math:`x` is the context, :math:`a` is the action (:math:`a_i` is chosen by the logging policy :math:`pi_0`), and :math:`r` is the reward.
        :math:`\\hat{q}(x, a) \\approx \\mathbb{E}[r|x,a]` is the predicted reward given context and action. :math:`n` is the number of the data sample.
        Note that we approximate the expectation :math:`\\mathbb{E}[\\cdot]` using a single monte-carlo sample for each index of data, :math:`i`.

        Similarly, when using a two-stage policy, POTEC estimates the policy gradient as follows.

        .. math::

            \\nabla_{\\theta} V(\\pi_{\\theta})
            \\approx \\frac{1}{n} \\sum_{i=1}^n \\frac{\\pi_{\\theta}^{\\text{1st}}(c(a_i)|x_i)}{\\pi_0^{\\text{1st}}(c(a_i)|x_i)} \\nabla_{\\theta} \\log \\pi_{\\theta}^{\\text{1st}}(c(a_i) | x_i) (r_i - \\hat{q}(x_i, a_i))
            + \\frac{1}{n} \\sum_{i=1}^n \\mathbb{E}_{a \\sim \\pi_{\\theta}(a|x_i)}[\\nabla_{\\theta} \\log \\pi_{\\theta}^{\\text{1st}}(c(a_i) | x_i) \\hat{q}(x_i, a)]

        Note that we parameterize the policy as :math:`\\pi_{\\theta}(a|x) = \\sum_{c \\in \\mathcal{C}} \\pi_{\\theta}^{\\text{1st}}(c|x) \\pi^{\\text{2nd}}(a|x,c)` when using POTEC.

        References
        -------
        Miroslav Dudík, John Langford, and Lihong Li.
        "Doubly robust policy evaluation and learning." 2011.

        Yuta Saito, Jihan Yao, and Thorsten Joachims.
        "POTEC: Off-policy learning for large action spaces via two-stage policy decomposition." 2024.

        Parameters
        -------
        logged_feedback: LoggedDataset
            Logged data, which contains the following keys.

            .. code-block:: python

                    key: [
                        context,          # (n_samples, )
                        query,            # (n_samples, )
                        user_id,          #  -
                        item_id,          # (n_samples, )
                        action,           # (n_samples, )
                        output,           # (n_samples, )
                        reward,           # (n_samples, )
                        logging_policy,   #  -
                    ]

            See dataset/synthetic.py for the details of each key.

        logging_action_choice_prob: torch.Tensor, shape (n_samples, n_actions), default=None
            Action choice probability of logging policy to all candidate actions.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions used for the logging policy in the clustering policy.

        clip_threshold: float or int, default=200
            Threshold for clipping the importance weight.

        n_epochs: int, default=1000
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=10
            Number of gradient steps within an epoch.

        n_epochs_per_log: int, default=10
            Number of epochs in the logging interval.

        batch_size: int, default=32
            Batch size.

        make_copy: bool, default=False
            Whether to create copy of the model before training.

        return_training_logs: bool, default=False
            Whether to return (true) policy value at each training epoch.

        save_path: Path, default=None
            Path to save the model.

        random_state: int, default=None
            Random state.

        use_wandb: bool, default=False
            Whether to use wandb to report the training statistics.

        experiment_name: str, default="hybrid"
            Experiment name used for wandb reports.

        Return
        -------
        model: Policy
            Trained policy.

        """
        if self.prompt_reward_predictor is None:
            raise RuntimeError(
                "prompt_reward_predictor is not provided. Please initialize the class with prompt_reward_predictor."
            )
        if return_training_logs and self.env is None:
            raise RuntimeError(
                "return_training_logs option is feasible when self.env is given. Please initialize the class with env."
            )

        # check_logged_feedback(logged_feedback)

        if random_state is None:
            random_state = self.random_state
        self.seed(random_state)

        if make_copy:
            model = deepcopy(self.model)
        else:
            model = self.model

        optimizer = self.optimizer(model.parameters(), **self.optimizer_kwargs)

        is_two_stage_policy = self.second_stage_policy is not None

        context = logged_feedback["context"]
        query = logged_feedback["query"]
        item_id = logged_feedback["item_id"]
        action = logged_feedback["action"]
        reward = logged_feedback["reward"]
        sentence = logged_feedback["sentence"]
        logging_policy = logged_feedback["logging_policy"]

        if logging_action_choice_prob is None:
            logging_action_choice_prob = logging_policy.calc_action_choice_probability(
                context=context,
                query=query,
                predicted_reward=logging_predicted_reward,
            )

        train_dataset = TorchLoggedDataset(
            context=context,
            query=query,
            item_id=item_id,
            action=action,
            sentence=sentence,
            reward=reward,
            query_embeddings=self.query_embeddings,
            logging_action_choice_prob=logging_action_choice_prob,
            query_encoder=self.query_encoder,
            sentence_encoder=self.sentence_encoder,
        )
        accelerator = Accelerator()

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        model, train_dataloader = accelerator.prepare(model, train_dataloader)

        model = self._init_policy_with_uniform_random(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            accelerator=accelerator,
            is_two_stage_policy=is_two_stage_policy,
        )

        train_losses = torch.zeros((n_epochs + 1,))
        policy_values = torch.zeros((n_epochs // n_epochs_per_log + 1,))

        if use_wandb:
            wandb.init(entity="", project=experiment_name)

        if is_two_stage_policy:
            eval_policy = TwoStagePolicy(
                first_stage_policy=model,
                second_stage_policy=self.second_stage_policy,
                clustering_policy=self.clustering_policy,
                device=self.device,
            )
        else:
            eval_policy = model

        if self.env is not None:
            policy_values[0] = self.env.calc_expected_policy_value(
                eval_policy,
                n_samples_to_approximate=1000,
            )
            policy_value = policy_values[0].item()
        else:
            policy_value = torch.zeros((1,))

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train policy: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "loss": f"{train_losses[i]:.4g}",
                        "policy_value": f"{policy_value:.4g}",
                    }
                )

                train_loss_logs = torch.zeros((n_steps_per_epoch,))

                train_iterator = iter(train_dataloader)

                for j in range(n_steps_per_epoch):
                    try:
                        batch_ = next(train_iterator)
                    except StopIteration:
                        # Reinitialize the iterator if the dataset is exhausted before reaching n_steps_per_epoch
                        train_iterator = iter(train_dataloader)
                        batch_ = next(train_iterator)

                    context_ = to_device(batch_["context"], device=accelerator.device)
                    query_ = to_device(batch_["query"], device=accelerator.device)
                    action_ = to_device(batch_["action"], device=accelerator.device)
                    sentence_ = to_device(batch_["sentence"], device=accelerator.device)
                    reward_ = to_device(batch_["reward"], device=accelerator.device)
                    logging_action_choice_prob_ = to_device(
                        batch_["logging_action_choice_prob"], accelerator.device
                    )
                    logging_pscore_ = logging_action_choice_prob_[
                        to_device(torch.arange(batch_size), device=accelerator.device),
                        action_,
                    ]

                    # importance sampling part
                    if is_two_stage_policy:
                        cluster_centers_ = (
                            self.clustering_policy.retrieve_cluster_centers(
                                context=context_,
                                query=query_,
                                resample_clustering=True,
                            )
                        )
                        cluster_ = self.clustering_policy.retrieve_cluster(
                            context=context_,
                            query=query_,
                            action=action_,
                            sentence=sentence_,
                        )
                        log_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=cluster_,
                            cluster_centers=cluster_centers_,
                            return_cpu_tensor=False,
                            is_log_prob=True,
                            calc_gradient=True,
                        )
                        evaluation_cluster_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=cluster_,
                            cluster_centers=cluster_centers_,
                            return_cpu_tensor=False,
                        )
                        logging_cluster_prob_ = (
                            self.clustering_policy.calc_cluster_choice_prob(
                                context=context_,
                                query=query_,
                                action_choice_prob=logging_action_choice_prob_,
                                cluster=cluster_,
                            )
                        )
                        iw_ = evaluation_cluster_prob_ / logging_cluster_prob_
                        iw_ = torch.nan_to_num(iw_)  # avoid zero division

                        prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=cluster_,
                            cluster_centers=cluster_centers_,
                            return_cpu_tensor=False,
                            is_log_prob=False,
                            calc_gradient=False,
                        )
                        imitation_prob_ = torch.zeros((1,))

                    else:
                        log_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=action_,
                            return_cpu_tensor=False,
                            is_log_prob=True,
                            calc_gradient=True,
                        )
                        evaluation_pscore_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=action_,
                            return_cpu_tensor=False,
                        )
                        iw_ = evaluation_pscore_ / logging_pscore_

                        _, prob_ = model.sample_action_and_output_prob(
                            context=context_,
                            query=query_,
                            return_cpu_tensor=False,
                            is_log_prob=False,
                            calc_gradient=False,
                        )
                        imitation_prob_ = evaluation_pscore_

                    baseline_ = self.prompt_reward_predictor.predict_value(
                        context=context_,
                        query=query_,
                        action=action_,
                        return_cpu_tensor=False,
                    )

                    iw_ = torch.clip(iw_, max=clip_threshold)
                    iw_based_loss_ = -(iw_ * log_prob_ * (reward_ - baseline_)).mean()

                    # model-based part
                    if is_two_stage_policy:
                        cluster_ = model.sample_action(
                            context=context_,
                            query=query_,
                            cluster_centers=cluster_centers_,
                            return_cpu_tensor=False,
                        )
                        eval_log_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            cluster_centers=cluster_centers_,
                            action=cluster_,
                            return_cpu_tensor=False,
                            is_log_prob=True,
                            calc_gradient=True,
                        )
                        candidate_actions_ = (
                            self.clustering_policy.retrieve_candidate_actions(
                                context=context_,
                                query=query_,
                                cluster=cluster_,
                            )
                        )
                        eval_predicted_reward_ = (
                            self.second_stage_policy.predict_policy_value(
                                context=context_,
                                query=query_,
                                candidate_actions=candidate_actions_,
                                return_per_sample=True,
                            )
                        )

                    else:
                        eval_action_ = model.sample_action(
                            context=context_,
                            query=query_,
                            return_cpu_tensor=False,
                        )
                        eval_log_prob_ = model.calc_prob_given_action(
                            context=context_,
                            query=query_,
                            action=eval_action_,
                            return_cpu_tensor=False,
                            is_log_prob=True,
                            calc_gradient=True,
                        )
                        eval_predicted_reward_ = (
                            self.prompt_reward_predictor.predict_value(
                                context=context_,
                                query=query_,
                                action=eval_action_,
                                return_cpu_tensor=False,
                            )
                        )

                    model_based_loss_ = -(
                        eval_predicted_reward_ * eval_log_prob_
                    ).mean()

                    train_loss_ = iw_based_loss_ + model_based_loss_

                    optimizer.zero_grad()
                    accelerator.backward(train_loss_)
                    optimizer.step()

                    train_losses[i + 1] += train_loss_.item() / n_steps_per_epoch
                    train_loss_logs[j] = train_loss_.item()

                if (i + 1) % n_epochs_per_log == 0:
                    if is_two_stage_policy:
                        eval_policy.first_stage_policy = model

                    if self.env is not None:
                        policy_values[
                            (i + 1) // n_epochs_per_log
                        ] = self.env.calc_expected_policy_value(
                            eval_policy,
                            n_samples_to_approximate=1000,
                        )
                        policy_value = policy_values[(i + 1) // n_epochs_per_log].item()
                        # print(policy_value)
                        # print(train_loss_logs.std().item() / batch_size)
                        # print()

                if use_wandb:
                    wandb.log(
                        {
                            "train_loss": train_losses[i + 1],
                            "policy_value": policy_value,
                            "average_iw": iw_.mean(),
                            "average_imitation": imitation_prob_.mean()
                            if is_two_stage_policy
                            else 0,
                            "average_prob": prob_.mean(),
                        }
                    )

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        if return_training_logs:
            output = (model, policy_values)
        else:
            output = model

        return output


@dataclass
class KernelPolicyLearner:
    model: BasePromptPolicyModel
    kernel_marginal_estimator: BaseKernelMarginalDensityModel
    action_list: Sentence
    query_embeddings: Optional[torch.Tensor] = None
    prompt_embeddings: Optional[torch.Tensor] = None
    sentence_reward_predictor: Optional[BaseSentenceRewardModel] = None
    optimizer: Optimizer = Adam
    optimizer_kwargs: Optional[Dict[str, Any]] = None
    env: Optional[SemiSyntheticDataset] = None
    frozen_llm: Optional[BaseFrozenLLM] = None
    query_encoder: Optional[BaseEncoder] = None
    sentence_encoder: Optional[BaseEncoder] = None
    random_state: Optional[int] = None

    """Learner class for action policy.

    Imported as: :class:`off_prompts.opl.PolicyLearner`

    Parameters
    -------
    model: BaseActionPolicyModel
        Single stage policy to train.

    kernel_marginal_estimator: BaseKernelMarginalDensityModel
        Kernel function.

    action_list: Sentence
        Mapping from action id to discrete prompts.

    query_embeddings: torch.Tensor, shape (n_items, dim_query_emb), default=None
        Mapping from item id to its (query) embeddings.

    prompt_embeddings: torch.Tensor, shape (n_items, dim_prompt_emb), default=None
        Mapping from item id to its (prompt) embeddings.

    sentence_reward_oredictor: BaseOutputRewardModel, default=None
        (Pre-trained) output reward predictor. This must be given when using a model-based or hybrid appraoch to train a policy.

    optimizer: torch.optim.Optimizer, default=Adam
        Class of optimizer (not an instance).

    optimizer_kwargs: dict, default=None
        Arguments of optimizer.

    env: SyntheticDataset, default=None
        Online environment for evaluation.

    frozen_llm: BaseFrozenLLM, default=None
        Frozen LLM.

    query_encoder: BaseEncoder, default=None
        Encoder of query.

    sentence_encoder: BaseEncoder, default=None
        Encoder of prompt.

    random_state: int, default=None.
        Random state.
    
    """

    def __post_init__(self):
        self.trained_model = None
        self.trained_action_predictor = None
        self.trained_sample_predictor = None
        self.n_actions = len(self.action_list)

        if self.random_state is None:
            raise ValueError("random_state must be given")

        if self.env is not None:
            if self.action_list is None:
                self.action_list = self.env.action_list
            if self.frozen_llm is None:
                self.frozen_llm = self.env.frozen_llm

        self.prompt_for_frozen_llm = tokenize(
            self.action_list,
            tokenizer=self.frozen_llm.tokenizer,
            tokenizer_kwargs=self.frozen_llm.tokenizer_kwargs,
            device=self.frozen_llm.device,
        )

        if self.optimizer_kwargs is None:
            self.optimizer_kwargs = {
                "lr": 1e-4,
                "momentum": 0.9,
            }

        if self.random_state is None:
            raise ValueError("random_state must be given")

        self.device = self.model.device

    def load(self, path: Path, is_init: bool = False):
        """Load model."""
        if is_init:
            self.model.load_state_dict(torch.load(path))
            model = self.model
        else:
            self.trained_model.load_state_dict(torch.load(path))
            model = self.trained_model

        return model

    def save(self, path):
        """Save model."""
        torch.save(self.trained_model.state_dict(), path)

    def seed(self, random_state: int):
        """Fix seed."""
        random.seed(random_state)
        torch_seed(random_state)

    def _init_policy_with_uniform_random(
        self,
        model: Union[BasePromptPolicyModel, BaseClusterPolicyModel],
        optimizer: Optimizer,
        dataloader: Optional[DataLoader] = None,
        accelerator: Optional[Accelerator] = None,
        batch_size: int = 32,
        n_steps_for_initialization: int = 1000,
        is_two_stage_policy: bool = False,
    ):
        """Initialize policy with a uniform random policy."""
        uniform_prob = 1 / self.n_actions

        if dataloader is not None:
            data_iterator = iter(dataloader)

        for j in range(n_steps_for_initialization):
            if self.env is not None:
                (
                    user_id_,
                    item_id_,
                    context_,
                    query_,
                    query_embeddings_,
                ) = self.env.context_query_loader.sample_context_and_query(
                    n_samples=batch_size,
                    return_query_embeddings=True,
                )

            elif dataloader is not None:
                try:
                    batch_ = next(data_iterator)
                except StopIteration:
                    # Reinitialize the iterator if the dataset is exhausted before reaching n_steps_per_epoch
                    data_iterator = iter(dataloader)
                    batch_ = next(data_iterator)

                context_ = to_device(batch_["context"], device=accelerator.device)
                query_ = to_device(batch_["query"], device=accelerator.device)

            else:
                raise ValueError("dataloader is not give.")

            if is_two_stage_policy:
                cluster_centers_ = self.clustering_policy.retrieve_cluster_centers(
                    context=context_,
                    query=query_,
                    resample_clustering=True,
                )
                cluster_ = model.sample_action(
                    context=context_,
                    query=query_embeddings_,
                    cluster_centers=cluster_centers_,
                    return_cpu_tensor=False,
                )
                prob_ = model.calc_prob_given_action(
                    context=context_,
                    query=query_embeddings_,
                    action=cluster_,
                    cluster_centers=cluster_centers_,
                    return_cpu_tensor=False,
                    is_log_prob=False,
                    calc_gradient=True,
                )
            else:
                action_ = model.sample_action(
                    context=context_,
                    query=query_embeddings_,
                    return_cpu_tensor=False,
                )
                prob_ = model.calc_prob_given_action(
                    context=context_,
                    query=query_embeddings_,
                    action=action_,
                    return_cpu_tensor=False,
                    is_log_prob=False,
                    calc_gradient=True,
                )

            # to get uniform random probability
            loss_ = torch.square(prob_ - uniform_prob).sum()

            optimizer.zero_grad()
            loss_.backward()
            optimizer.step()

        return model

    def importance_sampling_based_policy_gradient(
        self,
        logged_feedback: LoggedDataset,
        clip_threshold: Union[float, int] = 200,
        n_epochs: int = 1000,
        n_steps_per_epoch: int = 10,
        n_epochs_per_log: int = 10,
        batch_size: int = 32,
        make_copy: bool = False,
        return_training_logs: bool = False,
        save_path: Optional[Path] = None,
        random_state: Optional[int] = None,
        use_wandb: bool = False,
        experiment_name: str = "DSO",
    ):
        """Train policy from logged data via the (kernel) importance sampling-based approach.

        Note
        -------
        Kernel importance sampling policy gradient (also referred to as "Direct Sentence Off-policy gradient; DSO") estimates the policy gradient as follows.

        .. math::

            \\nabla_{\\theta} V(\\pi_{\\theta})
            \\approx \\frac{1}{n} \\sum_{i=1}^n \\frac{\\pi_{\\theta}(\\phi(s_i)|x_i)}{\\pi_0(\\phi(s_i)|x_i)} \\nabla_{\\theta} \\log \\pi_{\\theta}(\\phi(s_i)|x_i) \\, r_i
            \\approx \\frac{1}{n} \\sum_{i=1}^n \\mathbb{E}_{(a, s') \\sim \\pi_{\\theta}(a|x_i)p_{\\text{LLM}}(s'|x_i,a)} \\biggl[ \\frac{K(s_i, s'; \\, x_i, \\tau) \\nabla_{\\theta} \\log \\pi_{\\theta}(a | x_i)}{\\pi_{0}(\\phi(s_i)|x_i)} \\biggr] \\, r_i

        where we parametrize the policy as :math:`\\pi_{\\theta}` using some parameters :math:`\\theta \\in \\Theta` (e.g., a neural network).
        :math:`x` is the context, :math:`a` is the action (:math:`a_i` is chosen by the logging policy :math:`pi_0`), :math:`s` is the sentence, and :math:`r` is the reward. :math:`n` is the number of the data sample.
        :math:`\\pi_0(\\phi(s)|x) = \\mathbb{E}_{\\pi_0(s'|x)}[K(s, s'; \\, x, \\tau)]` is the (estimated) logging marginal density. :math:`K(\\cdot)` is a kernel function and :math:`\\tau` is its bandwidthhyperparameter. To see how to estimate :math:`\\pi_0(\\phi(s)|x)`, please also refer to :class:`off_prompts.opl.MarginalDensityLearner`.
        Note that we approximate the expectation (in the numerator) :math:`\\mathbb{E}[\\cdot]` using a single monte-carlo sample for each index of data, :math:`i`.

        References
        -------
        Haruka Kiyohara, Daniel Yiming Cao, Yuta Saito, and Thorsten Joachims.
        "Off-policy learning for prompt-guided text personalization using logged bandit data". 2025.

        Parameters
        -------
        logged_feedback: LoggedDataset
            Logged data, which contains the following keys.

            .. code-block:: python

                    key: [
                        context,          # (n_samples, )
                        query,            # (n_samples, )
                        user_id,          #  -
                        item_id,          # (n_samples, )
                        action,           # (n_samples, )
                        output,           #  -
                        reward,           # (n_samples, )
                        logging_policy,   #  -
                    ]

            See dataset/synthetic.py for the details of each key.

        clip_threshold: float or int, default=200
            Threshold for clipping the importance weight.

        n_epochs: int, default=1000
            Number of epochs to train the model.

        n_steps_per_epoch: int, default=10
            Number of gradient steps within an epoch for training policy.

        n_epochs_per_log: int, default=10
            Number of epochs in the logging interval.

        batch_size: int, default=32
            Batch size.

        make_copy: bool, default=False
            Whether to create copy of the model before training.

        return_training_logs: bool, default=False
            Whether to return (true) policy value at each training epoch.

        save_path: Path, default=None
            Path to save the model.

        random_state: int, default=None
            Random state.

        use_wandb: bool, default=False
            Whether to use wandb to report the training statistics.

        experiment_name: str, default="DSO"
            Experiment name used for wandb reports.

        Return
        -------
        model: Policy
            Trained policy.

        """
        if return_training_logs and self.env is None:
            raise RuntimeError(
                "return_training_logs option is feasible when self.env is given. Please initialize the class with env."
            )

        # check_logged_feedback(logged_feedback)

        if random_state is None:
            random_state = self.random_state
        self.seed(random_state)

        if make_copy:
            model = deepcopy(self.model)
        else:
            model = self.model

        context = logged_feedback["context"]
        query = logged_feedback["query"]
        action = logged_feedback["action"]
        item_id = logged_feedback["item_id"]
        sentence = logged_feedback["sentence"]
        reward = logged_feedback["reward"]
        logging_policy = logged_feedback["logging_policy"]

        logging_marginal_density = (
            self.kernel_marginal_estimator.estimate_marginal_density(
                context=context,
                query=query,
                sentence=sentence,
            )
        )

        train_dataset = TorchLoggedDataset(
            context=context,
            query=query,
            action=action,
            item_id=item_id,
            sentence=sentence,
            reward=reward,
            logging_marginal_density=logging_marginal_density,
            action_list=self.action_list,
            query_embeddings=self.query_embeddings,
            prompt_embeddings=self.prompt_embeddings,
            frozen_llm=self.frozen_llm,
            query_encoder=self.query_encoder,
            sentence_encoder=self.sentence_encoder,
        )
        accelerator = Accelerator()

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        model, train_dataloader = accelerator.prepare(model, train_dataloader)

        optimizer = self.optimizer(model.parameters(), **self.optimizer_kwargs)

        model = self._init_policy_with_uniform_random(
            model=model,
            optimizer=optimizer,
            dataloader=train_dataloader,
            accelerator=accelerator,
            is_two_stage_policy=False,
        )

        train_losses = torch.zeros((n_epochs + 1,))
        policy_values = torch.zeros((n_epochs // n_epochs_per_log + 1,))

        if use_wandb:
            wandb.init(entity="", project=experiment_name)

        if self.env is not None:
            policy_values[0] = self.env.calc_expected_policy_value(
                model,
                n_samples_to_approximate=1000,
            )
            policy_value = policy_values[0].item()
        else:
            policy_value = torch.zeros((1,))

        with tqdm(torch.arange(n_epochs)) as pbar:
            for i, ch in enumerate(pbar):
                pbar.set_description(f"[train policy: Epoch {i}]")
                pbar.set_postfix(
                    {
                        "loss": f"{train_losses[i]:.4g}",
                        "policy_value": f"{policy_value:.4g}",
                    }
                )

                train_loss_logs = torch.zeros((n_steps_per_epoch,))

                train_iterator = iter(train_dataloader)

                for j in range(n_steps_per_epoch):
                    try:
                        batch_ = next(train_iterator)
                    except StopIteration:
                        # Reinitialize the iterator if the dataset is exhausted before reaching n_steps_per_epoch
                        train_iterator = iter(train_dataloader)
                        batch_ = next(train_iterator)

                    context_ = to_device(batch_["context"], device=accelerator.device)
                    query_ = to_device(batch_["query"], device=accelerator.device)
                    action_ = to_device(batch_["action"], device=accelerator.device)
                    sentence_ = to_device(batch_["sentence"], device=accelerator.device)
                    reward_ = to_device(batch_["reward"], device=accelerator.device)
                    logging_marginal_density_ = to_device(
                        batch_["logging_marginal_density"], device=accelerator.device
                    )
                    query_for_frozen_llm_ = to_device(
                        batch_["query_for_frozen_llm"], device=accelerator.device
                    )

                    sampled_action_ = model.sample_action(
                        context=context_,
                        query=query_,
                        return_cpu_tensor=False,
                    )

                    if self.prompt_for_frozen_llm is not None:
                        prompt_for_frozen_llm_ = {}
                        for key in self.prompt_for_frozen_llm:
                            prompt_for_frozen_llm_[key] = self.prompt_for_frozen_llm[
                                key
                            ][sampled_action_].to(accelerator.device)
                    else:
                        prompt_for_frozen_llm_ = list(
                            itemgetter(*sampled_action_)(self.action_list)
                        )

                    sampled_sentence_ = self.frozen_llm.generate_output_sentence(
                        query=query_for_frozen_llm_,
                        prompt=prompt_for_frozen_llm_,
                    )

                    pairwise_weight_ = (
                        self.kernel_marginal_estimator.calc_pairwise_weight(
                            pivot_sentence=sentence_,
                            sampled_sentences=sampled_sentence_,
                            context=context_,
                            query=query_,
                        )
                    )
                    log_prob_ = model.calc_prob_given_action(
                        context=context_,
                        query=query_,
                        action=sampled_action_,
                        return_cpu_tensor=False,
                        is_log_prob=True,
                        calc_gradient=True,
                    )
                    imitation_prob_ = model.calc_prob_given_action(
                        context=context_,
                        query=query_,
                        action=action_,
                        return_cpu_tensor=False,
                        is_log_prob=False,
                        calc_gradient=False,
                    )
                    prob_ = model.calc_prob_given_action(
                        context=context_,
                        query=query_,
                        action=sampled_action_,
                        return_cpu_tensor=False,
                        is_log_prob=False,
                        calc_gradient=False,
                    )

                    marginal_iw_ = pairwise_weight_ / logging_marginal_density_
                    marginal_iw_ = torch.clip(marginal_iw_, max=clip_threshold)

                    policy_loss_ = -(marginal_iw_ * log_prob_ * reward_).sum()

                    optimizer.zero_grad()
                    accelerator.backward(policy_loss_)
                    optimizer.step()

                    train_losses[i + 1] += policy_loss_.item() / n_steps_per_epoch
                    train_loss_logs[j] = policy_loss_.item()

                if (i + 1) % n_epochs_per_log == 0:
                    if self.env is not None:
                        policy_values[
                            (i + 1) // n_epochs_per_log
                        ] = self.env.calc_expected_policy_value(
                            model,
                            n_samples_to_approximate=1000,
                        )
                        policy_value = policy_values[(i + 1) // n_epochs_per_log].item()
                        # print(policy_value)
                        # print(train_loss_logs.std().item() / batch_size)
                        # print()

                if use_wandb:
                    wandb.log(
                        {
                            "train_loss": train_losses[i + 1],
                            "policy_value": policy_value,
                            "average_iw": marginal_iw_.mean(),
                            "average_imitation": imitation_prob_.mean(),
                            "average_prob": prob_.mean(),
                        }
                    )

        self.trained_model = model

        if save_path is not None:
            self.save(save_path)

        if return_training_logs:
            output = (model, policy_values)
        else:
            output = model

        return output
