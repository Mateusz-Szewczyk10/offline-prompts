"""Evaluator class for policies."""
from dataclasses import dataclass
from typing import Union, Optional

import torch

from ..dataset.benchmark import SemiSyntheticDataset
from ..dataset.base import BaseEncoder, BaseFrozenLLM
from ..policy.base import (
    BasePolicy,
    BasePromptPolicyModel,
    BasePromptRewardModel,
    BaseSentenceRewardModel,
    BaseKernelMarginalDensityModel,
    BaseClusteringModel,
)
from ..types import LoggedDataset, Sentence
from ..utils import check_logged_feedback, tokenize, to_device

Policy = Union[BasePolicy, BasePromptPolicyModel]


@dataclass
class PolicyEvaluator:
    """Evaluator class for (action-selection) policy.

    Imported as: :class:`off_prompts.opl.PolicyEvaluator`

    Parameters
    -------
    env: SyntheticDataset, default=None
        Online environment for evaluation.

    prompt_reward_predictor: BasePromptRewardModel, default=None
        (Pre-trained) action reward predictor. This must be given when using a regression-based or hybrid appraoch to evaluate a policy.

    clustering_policy: BaseClusteringModel, default=None
        (Pre-trained) clustering policy that determines the action clustering for each context. This must be given when using OffCEM.

    query_embeddings: torch.Tensor, shape (n_items, dim_query_emb), default=None
        Mapping from item id to its (query) embeddings.

    prompt_embeddings: torch.Tensor, shape (n_items, dim_prompt_emb), default=None
        Mapping from item id to its (prompt) embeddings.

    query_encoder: BaseEncoder, default=None
        Encoder of query.

    frozen_llm: BaseFrozenLLM, default=None
        Frozen LLM.

    random_state: int, default=None
        Random state.

    """

    env: Optional[SemiSyntheticDataset] = None
    prompt_reward_predictor: Optional[BasePromptRewardModel] = None
    clustering_policy: Optional[BaseClusteringModel] = None
    action_list: Optional[Sentence] = None
    query_embeddings: Optional[torch.Tensor] = None
    prompt_embeddings: Optional[torch.Tensor] = None
    query_encoder: Optional[BaseEncoder] = None
    frozen_llm: Optional[BaseFrozenLLM] = None
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.env is not None:
            if self.action_list is None:
                self.action_list = self.env.action_list
            if self.frozen_llm is None:
                self.frozen_llm = self.env.frozen_llm
            if self.query_embeddings is None:
                self.query_embeddings = self.env.query_embeddings
            if self.prompt_embeddings is None:
                self.prompt_embeddings = self.env.prompt_embeddings

        if self.action_list is not None and self.frozen_llm is not None:
            self.prompt_for_frozen_llm = tokenize(
                self.action_list,
                tokenizer=self.frozen_llm.tokenizer,
                tokenizer_kwargs=self.frozen_llm.tokenizer_kwargs,
                device=self.frozen_llm.device,
            )
        else:
            self.prompt_for_frozen_llm = None

        if self.random_state is None:
            raise ValueError("random_state must be given")

    def online_policy_evaluation(
        self,
        eval_policy: Policy,
        n_samples: int = 100,
    ):
        """Evaluate the policy value via rollout in the online environment.

        Note
        -------
        Online policy evaluation approximates the policy value as follows.

        .. math::

            V(\\pi_{\\theta})
            \\approx \\frac{1}{n} \\sum_{i=1}^n \\mathbb{E}_{x_i \\sim p(x), a_i \\sim \\pi_{\\theta}(a | x_i), r_i \\sim p(r|x_i,a_i)}
            \\left[ r_i \\right]

        where we parametrize the policy as :math:`\\pi_{\\theta}` using some parameters :math:`\\theta \\in \\Theta` (e.g., a neural network).
        :math:`x` is the context, :math:`a` is the action, and :math:`r` is the reward.
        :math:`m` is the number of batched samples used in Monte-Carlo sampling.

        References
        -------
        Richard S Sutton and Andrew G Barto.
        "Reinforcement learning: An introduction." 2018.

        Parameters
        -------
        eval_policy: Policy
            Policy to be evaluated.

        n_samples: int, default=100
            Number of samples used for online rollouts.

        Return
        -------
        policy_value: float
            Expected reward approximated by the online rollouts.

        """
        logged_feedback = self.env.sample_dataset(
            policy=eval_policy,
            n_samples=n_samples,
        )
        policy_value = logged_feedback["reward"].mean()
        return policy_value

    def regression_based_policy_evaluation(
        self,
        logged_feedback: LoggedDataset,
        eval_policy: Policy,
        prompt_reward_predictor: Optional[BasePromptRewardModel] = None,
    ):
        """Evaluate the policy value via the regression-based approach.

        Note
        -------
        Direct method (regression-based) estimates the policy value as follows.

        .. math::

            V(\\pi_{\\theta})
            \\approx \\frac{1}{n} \\sum_{i=1}^n \\mathbb{E}_{a \\sim \\pi_{\\theta}(a | x_i)}
            \\left[ \\hat{q}(x_i, a) \\right]

        where we parametrize the policy as :math:`\\pi_{\\theta}` using some parameters :math:`\\theta \\in \\Theta` (e.g., a neural network).
        :math:`x` is the context, :math:`a` is the action, and :math:`r` is the reward.
        :math:`\\hat{q}(x, a) \\approx \\mathbb{E}[r|x,a]` is the predicted reward given context and action. :math:`n` is the number of the data sample.
        Note that we approximate the expectation :math:`\\mathbb{E}[\\cdot]` using a single monte-carlo sample for each index of data, :math:`i`.

        References
        -------
        Alina Beygelzimer and John Langford.
        "The offset tree for learning with partial labels." 2009.

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

        eval_policy: Policy
            Policy to be evaluated.

        prompt_reward_predictor: BasePromptRewardModel, default=None
            (Pre-trained) action reward predictor.

        Return
        -------
        policy_value: float
            Expected reward approximated by the regressed reward.

        """
        if prompt_reward_predictor is None:
            prompt_reward_predictor = self.prompt_reward_predictor

        if prompt_reward_predictor is None:
            raise RuntimeError("prompt_reward_predictor must be given.")

        device = prompt_reward_predictor.device
        context = logged_feedback["context"].to(device)
        item_id = logged_feedback["item_id"].to(device)

        if self.query_embeddings is not None:
            query = self.query_embeddings[item_id].to(device)
        else:
            query = query_encoder.encode(logged_feedback["query"]).to(device)

        action = eval_policy.sample_action(
            context=context,
            query=query,
            return_cpu_tensor=False,
        )
        predicted_reward = prompt_reward_predictor.predict_value(
            context=context,
            query=query,
            action=action,
            return_cpu_tensor=False,
        )

        policy_value = predicted_reward.mean()
        return policy_value

    def importance_sampling_based_policy_evaluation(
        self,
        logged_feedback: LoggedDataset,
        eval_policy: Policy,
        logging_action_choice_prob: Optional[torch.Tensor] = None,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        clip_threshold: int = 200,
    ):
        """Evaluate the policy value via the importance sampling-based approach.

        Note
        -------
        Inverse Propensity Scoring; IPS (importance sampling) estimates the policy value as follows.

        .. math::

            V(\\pi_{\\theta})
            \\approx \\frac{1}{n} \\sum_{i=1}^n \\frac{\\pi_{\\theta}(a_i | x_i)}{\\pi_0(a_i | x_i)} r_i

        where we parametrize the policy as :math:`\\pi_{\\theta}` using some parameters :math:`\\theta \\in \\Theta` (e.g., a neural network).
        :math:`x` is the context, :math:`a` is the action (:math:`a_i` is chosen by the logging policy :math:`pi_0`), and :math:`r` is the reward. :math:`n` is the number of the data sample.

        References
        -------
        Alex Strehl, John Langford, Lihong Li, and Sham M Kakade.
        "Learning from logged implicit exploration data." 2010.

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

        eval_policy: Policy
            Policy to be evaluated.

        logging_action_choice_prob: torch.Tensor, shape (n_samples, n_actions), default=None
            Action choice probability of logging policy to all candidate actions.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions used for the logging policy in the clustering policy.

        clip_threshold: float or int, default=200
            Threshold for clipping the importance weight.

        Return
        -------
        policy_value: float
            Expected reward approximated via importance sampling.

        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        context = logged_feedback["context"].to(device)
        action = logged_feedback["action"].to(device)
        reward = logged_feedback["reward"].to(device)
        item_id = logged_feedback["item_id"].to(device)

        if self.query_embeddings is not None:
            query = self.query_embeddings[item_id].to(device)
        else:
            query = query_encoder.encode(logged_feedback["query"]).to(device)

        logging_policy = logged_feedback["logging_policy"]
        if logging_action_choice_prob is None:
            logging_action_choice_prob = logging_policy.calc_action_choice_probability(
                context=context,
                query=query,
                predicted_reward=logging_predicted_reward,
                return_cpu_tensor=False,
            )

        n_samples = len(context)
        logging_action_prob = logging_action_choice_prob[
            to_device(torch.arange(n_samples), device=device), action
        ]
        eval_action_prob = eval_policy.calc_prob_given_action(
            context=context,
            query=query,
            action=action,
            return_cpu_tensor=False,
        )
        iw = eval_action_prob / logging_action_prob
        iw = torch.nan_to_num(iw)
        iw = torch.clip(iw, max=clip_threshold)

        policy_value = (iw * reward).mean()
        return policy_value

    def hybrid_policy_evaluation(
        self,
        logged_feedback: LoggedDataset,
        eval_policy: Policy,
        prompt_reward_predictor: Optional[BasePromptRewardModel] = None,
        logging_action_choice_prob: Optional[torch.Tensor] = None,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        clip_threshold: int = 200,
    ):
        """Evaluate the policy value via the hybrid approach (i.e., doubly robust; DR).

        Note
        -------
        Doubly robust; DR estimates the policy value as follows.

        .. math::

            V(\\pi_{\\theta})
            \\approx \\frac{1}{n} \\sum_{i=1}^n \\left \\{ \\frac{\\pi_{\\theta}(a_i | x_i)}{\\pi_0(a_i | x_i)}  (r_i - \\hat{q}(x_i, a_i))
            + \\mathbb{E}_{a \\sim \\pi_{\\theta}(a|x_i)}[\\hat{q}(x_i, a)] \\right \\}

        where we parametrize the policy as :math:`\\pi_{\\theta}` using some parameters :math:`\\theta \\in \\Theta` (e.g., a neural network).
        :math:`x` is the context, :math:`a` is the action (:math:`a_i` is chosen by the logging policy :math:`pi_0`), and :math:`r` is the reward.
        :math:`\\hat{q}(x, a) \\approx \\mathbb{E}[r|x,a]` is the predicted reward given context and action. :math:`n` is the number of the data sample.
        Note that we approximate the expectation :math:`\\mathbb{E}[\\cdot]` using a single monte-carlo sample for each index of data, :math:`i`.

        References
        -------
        Miroslav Dudík, John Langford, and Lihong Li.
        "Doubly robust policy evaluation and learning." 2011.

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

        eval_policy: Policy
            Policy to be evaluated.

        prompt_reward_predictor: BasePromptRewardModel, default=None
            (Pre-trained) action reward predictor.

        logging_action_choice_prob: torch.Tensor, shape (n_samples, n_actions), default=None
            Action choice probability of logging policy to all candidate actions.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions used for the logging policy in the clustering policy.

        clip_threshold: float or int, default=200
            Threshold for clipping the importance weight.

        Return
        -------
        policy_value: float
            Expected reward approximated by doubly robust.

        """
        if prompt_reward_predictor is None:
            prompt_reward_predictor = self.prompt_reward_predictor

        if prompt_reward_predictor is None:
            raise RuntimeError("prompt_reward_predictor must be given.")

        device = prompt_reward_predictor.device
        context = logged_feedback["context"].to(device)
        action = logged_feedback["action"].to(device)
        reward = logged_feedback["reward"].to(device)
        item_id = logged_feedback["item_id"].to(device)

        if self.query_embeddings is not None:
            query = self.query_embeddings[item_id].to(device)
        else:
            query = query_encoder.encode(logged_feedback["query"]).to(device)

        logging_policy = logged_feedback["logging_policy"]
        if logging_action_choice_prob is None:
            logging_action_choice_prob = logging_policy.calc_action_choice_probability(
                context=context,
                query=query,
                predicted_reward=logging_predicted_reward,
            )

        n_samples = len(context)
        logging_action_prob = logging_action_choice_prob[
            to_device(torch.arange(n_samples), device=device), action
        ]
        eval_action_prob = eval_policy.calc_prob_given_action(
            context=context,
            query=query,
            action=action,
            return_cpu_tensor=False,
        )
        iw = eval_action_prob / logging_action_prob
        iw = torch.nan_to_num(iw)
        iw = torch.clip(iw, max=clip_threshold)

        predicted_reward = prompt_reward_predictor.predict_value(
            context=context,
            query=query,
            action=action,
            return_cpu_tensor=False,
        )

        eval_action = eval_policy.sample_action(
            context=context,
            query=query,
            return_cpu_tensor=False,
        )
        eval_predicted_reward = prompt_reward_predictor.predict_value(
            context=context,
            query=query,
            action=eval_action,
            return_cpu_tensor=False,
        )

        policy_value = (iw * (reward - predicted_reward) + eval_predicted_reward).mean()
        return policy_value

    def offcem_policy_evaluation(
        self,
        logged_feedback: LoggedDataset,
        eval_policy: Policy,
        prompt_reward_predictor: Optional[BasePromptRewardModel] = None,
        logging_action_choice_prob: Optional[torch.Tensor] = None,
        logging_predicted_reward: Optional[torch.Tensor] = None,
        clustering_policy: Optional[BaseClusteringModel] = None,
        clip_threshold: int = 200,
    ):
        """Evaluate the policy value via OffCEM.

        Note
        -------
        OffCEM estimates the policy value as follows.

        .. math::

            V(\\pi_{\\theta})
            \\approx \\frac{1}{n} \\sum_{i=1}^n \\left \\{ \\frac{\\pi_{\\theta}(c(a_i)|x_i)}{\\pi_0(c(a_i)|x_i)} (r_i - \\hat{q}(x_i, a_i))
            + \\mathbb{E}_{a \\sim \\pi_{\\theta}(a|x_i)}[\\hat{q}(x_i, a)] \\right \\}

        where we parametrize the policy as :math:`\\pi_{\\theta}` using some parameters :math:`\\theta \\in \\Theta` (e.g., a neural network).
        :math:`x` is the context, :math:`a` is the action (:math:`a_i` is chosen by the logging policy :math:`pi_0`), and :math:`r` is the reward.
        :math:`\\hat{q}(x, a) \\approx \\mathbb{E}[r|x,a]` is the predicted reward given context and action. :math:`n` is the number of the data sample.
        :math:`\\pi(c(a)|x) = \\sum_{a' \\in \\mathcal{A}, c(a')=c(a)} \\pi(a|x)` is the probability of choosing cluster :math:`c` under policy :math:`\pi`, but because this can be calculated for arbitrary clustering,
        this estimator is applicable to arbitrary policy which do not have two-stage structure.
        Note that we approximate the expectation :math:`\\mathbb{E}[\\cdot]` using a single monte-carlo sample for each index of data, :math:`i`.

        References
        -------
        Yuta Saito, Qingyang Ren, and Thorsten Joachims.
        "Off-policy evaluation for large action spaces via conjunct effect modeling." 2023.

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

        eval_policy: Policy
            Policy to be evaluated.

        prompt_reward_predictor: BasePromptRewardModel, default=None
            (Pre-trained) action reward predictor.

        logging_action_choice_prob: torch.Tensor, shape (n_samples, n_actions), default=None
            Action choice probability of logging policy to all candidate actions.

        logging_predicted_reward: torch.Tensor, shape (n_samples, n_actions), default=None
            Predicted reward for all candidate actions used for the logging policy in the clustering policy.

        clustering_policy: BaseClusteringModel, default=None
            (Pre-trained) clustering policy that determines the action clustering for each context.

        clip_threshold: float or int, default=200
            Threshold for clipping the importance weight.

        Return
        -------
        policy_value: float
            Expected reward approximated by OffCEM.

        """
        if prompt_reward_predictor is None:
            prompt_reward_predictor = self.prompt_reward_predictor

        if prompt_reward_predictor is None:
            raise RuntimeError("prompt_reward_predictor must be given.")

        if clustering_policy is None:
            clustering_policy = self.clustering_policy

        if clustering_policy is None:
            raise RuntimeError("clustering_policy must be given.")

        device = prompt_reward_predictor.device
        context = logged_feedback["context"].to(device)
        action = logged_feedback["action"].to(device)
        sentence = logged_feedback["sentence"].to(device)
        reward = logged_feedback["reward"].to(device)
        item_id = logged_feedback["item_id"].to(device)

        if self.query_embeddings is not None:
            query = self.query_embeddings[item_id].to(device)
        else:
            query = query_encoder.encode(logged_feedback["query"]).to(device)

        logging_policy = logged_feedback["logging_policy"]
        if logging_action_choice_prob is None:
            logging_action_choice_prob = logging_policy.calc_action_choice_probability(
                context=context,
                query=query,
                predicted_reward=logging_predicted_reward,
                return_cpu_tensor=False,
            )
        eval_action_choice_prob = eval_policy.calc_action_choice_probability(
            context=context,
            query=query,
        )

        cluster = self.clustering_policy.retrieve_cluster(
            context=context,
            query=query,
            action=action,
            sentence=sentence,
            resample_clustering=True,
            return_cpu_tensor=False,
        )

        logging_cluster_prob = self.clustering_policy.calc_cluster_choice_prob(
            context=context,
            query=query,
            cluster=cluster,
            action_choice_prob=logging_action_choice_prob,
            return_cpu_tensor=False,
        )
        eval_cluster_prob = self.clustering_policy.calc_cluster_choice_prob(
            context=context,
            query=query,
            cluster=cluster,
            action_choice_prob=eval_action_choice_prob,
            return_cpu_tensor=False,
        )

        iw = eval_cluster_prob / logging_cluster_prob
        iw = torch.nan_to_num(iw)
        iw = torch.clip(iw, max=clip_threshold)

        predicted_reward = prompt_reward_predictor.predict_value(
            context=context,
            query=query,
            action=action,
            return_cpu_tensor=False,
        )

        eval_action = eval_policy.sample_action(
            context=context,
            query=query,
            return_cpu_tensor=False,
        )
        eval_predicted_reward = prompt_reward_predictor.predict_value(
            context=context,
            query=query,
            action=eval_action,
            return_cpu_tensor=False,
        )

        policy_value = (iw * (reward - predicted_reward) + eval_predicted_reward).mean()
        return policy_value

    def kernel_IS_based_policy_evaluation(
        self,
        logged_feedback: LoggedDataset,
        eval_policy: Policy,
        logging_kernel_density_model: BaseKernelMarginalDensityModel,
        action_list: Optional[action_list] = None,
        frozen_llm: Optional[BaseFrozenLLM] = None,
        n_samples_to_approximate: int = 5,
        clip_threshold: int = 200,
    ):
        """Evaluate the policy value via kernel IS (i.e., OPE estimator corresponds to DSO).

        Note
        -------
        Kernel importance sampling estimates the policy value as follows.

        .. math::

            V(\\pi_{\\theta})
            \\approx \\frac{1}{n} \\sum_{i=1}^n \\frac{\\pi_{\\theta}(\\phi(s_i)|x_i)}{\\pi_0(\\phi(s_i)|x_i)} r_i

        where we parametrize the policy as :math:`\\pi_{\\theta}` using some parameters :math:`\\theta \\in \\Theta` (e.g., a neural network).
        :math:`x` is the context, :math:`a` is the action (:math:`a_i` is chosen by the logging policy :math:`pi_0`), :math:`s` is the sentence, and :math:`r` is the reward. :math:`n` is the number of the data sample.
        :math:`\\pi_0(\\phi(s)|x) = \\mathbb{E}_{\\pi_0(s'|x)}[K(s, s'; \\, x, \\tau)]` is the (estimated) logging marginal density. :math:`K(\\cdot)` is a kernel function and :math:`\\tau` is its bandwidthhyperparameter. To see how to estimate :math:`\\pi_0(\\phi(s)|x)`, please also refer to :class:`off_prompts.opl.MarginalDensityLearner`.
        For the marginal density of the policy of interest (i.e., :math:`\\pi_{\\theta}`), we approximate the expectation :math:`\\pi(\\phi(s)|x) = \\mathbb{E}_{\\pi(s'|x)}[K(s, s'; \\, x, \\tau)]` using # (n_samples_to_approximate) of samples.

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
                        output,           # (n_samples, )
                        reward,           # (n_samples, )
                        logging_policy,   #  -
                    ]

            See dataset/synthetic.py for the details of each key.

        eval_policy: Policy
            Policy to be evaluated.

        logging_marginal_density_model: BaseKernelMarginalDensityModel
            Model to estimate the logging policy's marginal density of a given sentence.

        action_list: Sentence, default=None
            Mapping from action id to discrete prompts.

        frozen_llm: BaseFrozenLLM, default=None
            Frozen LLM.

        n_samples_to_approximate: int, default=5
            Number of samples to approximate the expectation in the numerator of the DSO estimator.

        clip_threshold: float or int, default=200
            Threshold for clipping the importance weight.

        Return
        -------
        policy_value: float
            Expected reward approximated by the kernel (marginal) IS.

        """
        if action_list is None:
            action_list = self.action_list
        if action_list is None:
            raise ValueError("action_list must be given.")

        if frozen_llm is None:
            frozen_llm = self.frozen_llm
        if frozen_llm is None:
            raise RuntimeError("frozen_llm must be given.")

        if logging_kernel_density_model is None:
            raise RuntimeError("logging_kernel_density_model must be given.")

        if self.prompt_for_frozen_llm is None:
            prompt_for_frozen_llm = tokenize(
                self.action_list,
                tokenizer=self.frozen_llm.tokenizer,
                tokenizer_kwargs=self.frozen_llm.tokenizer_kwargs,
                device=self.frozen_llm.device,
            )
        else:
            prompt_for_frozen_llm = self.prompt_for_frozen_llm

        device = frozen_llm.device
        context = logged_feedback["context"].to(device)
        reward = logged_feedback["reward"].to(device)
        item_id = logged_feedback["item_id"].to(device)
        sentence = logged_feedback["sentence"]
        n_samples = len(context)

        if self.query_embeddings is not None:
            query = self.query_embeddings[item_id].to(device)
        else:
            query = query_encoder.encode(logged_feedback["query"]).to(device)

        logging_marginal_density = (
            logging_kernel_density_model.estimate_marginal_density(
                context=context,
                query=query,
                sentence=sentence,
                return_cpu_tensor=False,
            )
        )

        eval_marginal_density = torch.zeros(
            (n_samples, n_samples_to_approximate), device=device
        )
        for i in range(n_samples_to_approximate):
            sampled_action_ = model.sample_action(
                context=context_,
                query=query_,
                return_cpu_tensor=False,
            )

            prompt_for_frozen_llm_ = {}
            for key in self.prompt_for_frozen_llm:
                prompt_for_frozen_llm_[key] = self.prompt_for_frozen_llm[key][
                    sampled_action_
                ].to(device)

            sampled_sentence_ = self.frozen_llm.generate_output_sentence(
                query=query_for_frozen_llm_,
                prompt=prompt_for_frozen_llm_,
            )

            eval_marginal_density[
                :, i
            ] = self.kernel_marginal_estimator.calc_pairwise_weight(
                pivot_sentence=sentence_,
                sampled_sentences=sampled_sentence_,
                context=context_,
                query=query_,
                return_cpu_tensor=False,
            )

        iw = eval_marginal_density.mean(dim=1) / logging_marginal_density
        iw = torch.nan_to_num(iw)
        iw = torch.clip(iw, max=clip_threshold)

        policy_value = (iw * reward).mean()
        return policy_value
