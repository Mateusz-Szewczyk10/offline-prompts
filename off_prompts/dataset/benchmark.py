"""Class to handle synthetic dataset generation."""
from dataclasses import dataclass
from typing import Optional, Union, Any
from operator import itemgetter

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from sklearn.utils import check_scalar

from .base import (
    BasePromptFormatter,
    BaseContextQueryLoader,
    BaseCandidateActionsLoader,
    BaseFrozenLLM,
    BaseRewardSimulator,
)
from .function import (
    DefaultContextQueryLoader,
    DefaultCandidateActionsLoader,
)
from .frozen_llm import AutoFrozenLLM
from .reward_simulator import TransformerRewardSimulator
from .prompt_formatter import MovielensPromptFormatter

# from ..policy.base import (
#     BasePolicy,
#     BasePromptPolicyModel,
#     BaseClusterPolicyModel,
# )
from ..types import Sentence, Tokens
from ..utils import torch_seed, to_device

# Policy = Union[BasePolicy, BasePromptPolicyModel, BaseClusterPolicyModel]
Policy = Any  # avoid circular import


@dataclass
class SemiSyntheticDataset:
    """Base class for logged dataset.

    Imported as: :class:`off_prompts.dataset.SemiSyntheticDataset`

    Parameters
    -------
    context_query_loader: ContextQueryLoader, default=None
        Data loader (generator) of context and query.

    candidate_actions_loader: CandidateActionsLoader, default=None
        Data loader (generator) of candidate actions.

    frozen_llm: BaseFrozenLLM, default=None
        Frozen LLM that generate sentences from query and prompt.

    reward_simulator: BaseRewardSimulator, default=None
        Reward simulator that determines expected reward given context, query, and generated sentence.

    frozen_llm_prompt_formatter: BasePromptFormatter
        Prompt formatter that is used in the frozen LLM.

    path_to_user_embeddings: str, default="assets/movielens_transformer_user_embeddings.pt"
        Path to user embeddings (.pt). Required only when context_query_loader is not given. (Optional)

    path_to_queries: str, default="assets/movielens_query.csv"
        Path to query (.csv). Required only when context_query_loader is not given. (Optional)

    path_to_query_embeddings: str, default="assets/movielens_query_embs.pt"
        Path to query embeddings (.pt). Required only when context_query_loader is not given. (Optional)

    path_to_interaction_data: str, default="assets/movielens_preprocessed_data.csv"
        Path to (user, item) collaborative filtering data. Required only when context_query_loader is not given. (Optional)

    path_to_candidate_prompts: str, default="assets/movielens_benchmark_prompts.csv"
        Path to the csv file that specifies the vocabulary used in simulation (.csv). Required only when candidate_actions_loader is not given. (Optional)

    path_to_prompt_embeddings: str, default="assets/movielens_prompt_embs.pt"
        Path to prompt embeddings (.pt). Required only when candidate_actions_loader is not given. (Optional)

    path_to_finetuned_params: str, default="assets/movielens_distilbert_reward_simulator.pt"
        Path to the .pt file containing the (fine-tuned) parameters of the reward simulator (.pt). Required only when reward_simulator is not given. (Optional)

    frozen_llm_base_model_id: str, default="mistralai/Mistral-7B-Instruct-v0.2"
        Base model of the frozen LLM (i.e., huggingface's model name). Required only when frozen_llm is not given. (Optional)

    frozen_llm_base_tokenizer_id: str, default="mistralai/Mistral-7B-Instruct-v0.2"
        Tokenizer of the frozen LLM (i.e., huggingface's model name). Required only when frozen_llm is not given. (Optional)

    frozen_llm_use_tokenizer_fast: bool, default=False.
        Whether to use TokenizerFast. Required only when frozen_llm is not given.

    frozen_llm_pattern: str, default=None
        System prompt for a frozen llm. (Optional)

    reward_simulator_base_model_id: str, default="distilbert-base-uncased"
        Base model of the reward simulator (i.e., huggingface's model name). Required only when frozen_llm is not given. (Optional)

    reward_simulator_base_tokenizer_id: str, default="distilbert-base-uncased"
        Tokenizer of the reward simulator (i.e., huggingface's model name). Required only when frozen_llm is not given. (Optional)

    reward_simulator_use_tokenizer_fast: bool, default=False.
        Whether to use TokenizerFast. Required only when frozen_llm is not given.

    reward_type: {"binary", "continuous"}, default="binary"
        Whether to sample binary reward or continuous reward.

    reward_std: float, default=1.0 (> 0.0)
        Standard deviation of the reward. Required only when reward_type=="continuous".

    device: str, default="cuda"
        Device of the model.

    random_state: int, default=None.
        Random state.

    """

    context_query_loader: Optional[BaseContextQueryLoader] = None
    candidate_actions_loader: Optional[BaseCandidateActionsLoader] = None
    frozen_llm: Optional[BaseFrozenLLM] = None
    reward_simulator: Optional[BaseRewardSimulator] = None
    frozen_llm_prompt_formatter: Optional[BasePromptFormatter] = None
    path_to_user_embeddings: str = "assets/movielens_transformer_user_embeddings.pt"
    path_to_queries: str = "assets/movielens_query.csv"
    path_to_query_embeddings: str = "assets/movielens_query_embs.pt"
    path_to_interaction_data: str = "assets/movielens_preprocessed_data.csv"
    path_to_candidate_prompts: str = "assets/movielens_benchmark_prompts.csv"
    path_to_prompt_embeddings: str = "assets/movielens_prompt_embs.pt"
    path_to_finetuned_params: str = "assets/movielens_distilbert_reward_simulator.pt"
    frozen_llm_base_model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"
    frozen_llm_base_tokenizer_id: str = "mistralai/Mistral-7B-Instruct-v0.2"
    frozen_llm_use_tokenizer_fast: bool = False
    frozen_llm_pattern: str = None
    reward_simulator_base_model_id: str = "distilbert-base-uncased"
    reward_simulator_base_tokenizer_id: str = "distilbert-base-uncased"
    reward_simulator_use_tokenizer_fast: bool = False
    reward_type: str = "binary"
    reward_std: float = 1.0
    device: str = "cuda"
    random_state: Optional[int] = None

    def __post_init__(self):
        check_scalar(self.reward_std, "reward_std", target_type=float, min_val=0.0)

        if self.random_state is not None:
            torch_seed(self.random_state, device=self.device)

        if self.context_query_loader is None:
            self.context_query_loader = DefaultContextQueryLoader(
                path_to_user_embeddings=self.path_to_user_embeddings,
                path_to_queries=self.path_to_queries,
                path_to_query_embeddings=self.path_to_query_embeddings,
                path_to_interaction_data=self.path_to_interaction_data,
                device=self.device,
                random_state=self.random_state,
            )

        if self.candidate_actions_loader is None:
            self.candidate_actions_loader = DefaultCandidateActionsLoader(
                n_actions=1000,
                path_to_candidate_prompts=self.path_to_candidate_prompts,
                path_to_prompt_embeddings=self.path_to_prompt_embeddings,
                random_state=self.random_state,
            )

        if self.frozen_llm is None:
            frozen_llm_tokenizer = AutoTokenizer.from_pretrained(
                self.frozen_llm_base_tokenizer_id,
                truncation=True,
                do_lower_case=True,
                use_fast=self.frozen_llm_use_tokenizer_fast,
            )

            frozen_llm_model = AutoModelForCausalLM.from_pretrained(
                self.frozen_llm_base_model_id,
            )

            frozen_llm_tokenizer_kwargs = {
                "add_special_tokens": True,
                "padding": True,
                "truncation": True,
                "max_length": 20,
                "return_tensors": "pt",
            }

            if self.frozen_llm_prompt_formatter is None:
                self.frozen_llm_prompt_formatter = MovielensPromptFormatter(
                    tokenizer=frozen_llm_tokenizer,
                    tokenizer_kwargs=frozen_llm_tokenizer_kwargs,
                    device=self.device,
                )

            if self.frozen_llm_pattern is None:
                self.frozen_llm_pattern = (
                    r"Broadly describe in a sentence the genres of the movie without including the name or any specifics of.*?\n\n",
                )[0]

            frozen_llm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            frozen_llm_model.resize_token_embeddings(len(frozen_llm_tokenizer))
            frozen_llm_model.to(self.device)

            self.frozen_llm = AutoFrozenLLM(
                prompt_formatter=self.frozen_llm_prompt_formatter,
                model=frozen_llm_model,
                tokenizer=frozen_llm_tokenizer,
                tokenizer_kwargs=frozen_llm_tokenizer_kwargs,
                pattern=self.frozen_llm_pattern,
                device=self.device,
                random_state=self.random_state,
            )

        if self.reward_simulator is None:
            reward_simulator_tokenizer = AutoTokenizer.from_pretrained(
                self.reward_simulator_base_tokenizer_id,
                truncation=True,
                do_lower_case=True,
                use_fast=self.reward_simulator_use_tokenizer_fast,
            )

            reward_simulator_base_model = AutoModel.from_pretrained(
                self.reward_simulator_base_model_id,
            )

            reward_simulator_tokenizer_kwargs = {
                "add_special_tokens": True,
                "padding": True,
                "truncation": True,
                "max_length": 20,
                "return_tensors": "pt",
            }

            reward_simulator_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            reward_simulator_base_model.resize_token_embeddings(
                len(reward_simulator_tokenizer)
            )
            reward_simulator_base_model.to(self.device)

            self.reward_simulator = TransformerRewardSimulator(
                n_users=self.context_query_loader.n_users,
                n_items=self.context_query_loader.n_queries,
                base_model=reward_simulator_base_model,
                tokenizer=reward_simulator_tokenizer,
                tokenizer_kwargs=reward_simulator_tokenizer_kwargs,
                device=self.device,
                random_state=self.random_state,
            )
            self.reward_simulator.load_state_dict(
                torch.load(self.path_to_finetuned_params),
            )

        self.action_list = self.candidate_actions_loader.action_list
        self.n_actions = self.candidate_actions_loader.n_actions
        self.dim_context = self.context_query_loader.dim_context

        self.user_embeddings = self.context_query_loader.user_embeddings
        self.query_embeddings = self.context_query_loader.query_embeddings
        self.prompt_embeddings = self.candidate_actions_loader.prompt_embeddings

    def _sample_reward(self, expected_reward: torch.Tensor):
        """Sample reward.

        Parameters
        -------
        expected_reward: torch.Tensor, shape (n_samples, )
            Expected reward predicted by some frozen LLMs (i.e., reward simulator).

        Return
        -------
        reward: torch.Tensor, shape (n_samples, )
            Either binary or continuous reward.

        """
        if self.reward_type == "binary":
            reward = torch.bernoulli(expected_reward)
        elif self.reward_type == "continuous":
            reward_std = torch.full_like(expected_reward, self.reward_std)
            reward = torch.normal(expected_reward, reward_std)
        return reward

    def sample_dataset(
        self,
        policy: Policy,
        n_samples: int = 10000,
        is_test: bool = False,
        return_user_id: bool = True,
        return_item_id: bool = True,
        return_context: bool = True,
        return_query: bool = True,
        return_action_choice_prob: bool = False,
        return_meta_data: bool = False,
    ):
        """Sample dataset given data collection policy.

        Parameters
        -------
        policy: Policy
            Policy that chooses discrete prompt (index) as action.

        n_samples: int, default=1000 (> 0)
            Number of samples.

        is_test: bool, default=False
            Whether to use test samples.

        return_user_id: bool, default=True
            Whether to return the user id.

        return_item_id: bool, default=True
            Whether to return the item id.

        return_context: bool, default=True
            Whether to return the context.

        return_query: bool, default=True
            Whether to return the query.

        return_action_choice_prob: bool, default=False
            Whether to record action choice probability of the data collection policy.

        return_meta_data: bool, default=False
            Whether to record meta data including data size, type of action, etc.

        Return
        -------
        logged_dataset: dict
            Dictionary containing dataset with the following keys:

            .. code-block:: python

                key: [
                    user_id,
                    item_id,
                    context,
                    query,
                    action,
                    action_choice_prob,
                    prompt,
                    sentence,
                    expected_reward,
                    reward,
                    logging_policy,
                ]

            user_id: torch.Tensor, shape (n_samples, )
                User id (corresponding to the index of context).

            item_id: torch.Tensor, shape (n_samples, )
                Item id (corresponding to the index of query).

            context: torch.Tensor, shape (n_samples, dim_context)
                Context of each user.

            query: Sentence, shape (n_samples, )
                Query given by users (e.g., movie title).

            action: torch.Tensor, shape (n_samples, )
                Discrete actions (index) chosen by the given policy.

            action_choice_prob: torch.Tensor, shape (n_samples, )
                Action choice probability of the sampled actions.

            sentence: Sentence, shape (n_samples, )
                Sentence generated by some frozen LLMs using user-specified query and prompts chosen by a policy.

            expected_reward: torch.Tensor, shape (n_samples, )
                Expected reward predicted by some frozen LLMs (i.e., reward simulator).

            reward: torch.Tensor, shape (n_samples, )
                Either binary or continuous reward.

            logging_policy: BasePolicy
                Policy that chooses either discrete or continuous (soft) prompt as action.

        meta_data: dict
            Dictionary containing setting of simulation with the following keys:

            .. code-block:: python

                key: [
                    size,
                    reward_type,
                    reward_std,
                    candidate_prompts,
                    user_embeddings,
                    item_embeddings,
                    prompt_embeddings,
                ]

            size: int
                Data size (i.e., number of samples).

            reward_type: {"binary", "continuous"}
                Whether to sample binary or continuous rewards.

            reward_std: float
                Noise level of reward. Required only when `reward_type=="continuous"`.

            candidate_prompts: Sentence, shape (n_actions, )
                Mapping from discrete action to prompts.

            user_embeddings: Sentence, shape (n_users, dim_context)
                Mapping from discrete user id to user embeddings.

            item_embeddings: Sentence, shape (n_items, dim_query)
                Mapping from discrete item (query) id to item embeddings.

            prompt_embeddings: Sentence, shape (n_actions, dim_action)
                Mapping from discrete action to prompt embeddings.

        """

        # self._check_policy(policy)

        (
            user_id,
            item_id,
            context,
            query,
        ) = self.context_query_loader.sample_context_and_query(
            n_samples=n_samples,
            is_test=is_test,
        )

        action = policy.sample_action(
            user_id=user_id,
            item_id=item_id,
            context=context,
            query=query,
        )
        prompt = list(itemgetter(*action)(self.action_list))

        action_ = to_device(action, device=self.device)

        output_sentence = self.frozen_llm.generate_output_sentence(
            query=query,
            prompt=prompt,
        )
        baseline_sentence = self.frozen_llm.generate_output_sentence(
            query=query,
            prompt=None,
        )

        expected_reward = self.reward_simulator.calc_expected_reward(
            user_id=user_id,
            item_id=item_id,
            context=context,
            query=query,
            action=action,
            sentence=output_sentence,
            baseline_sentence=baseline_sentence,
        )
        reward = self._sample_reward(expected_reward)

        if return_action_choice_prob:
            action_choice_prob = policy.calc_prob_given_action(
                user_id=user_id,
                item_id=item_id,
                context=context,
                query=query,
                action=action,
            )
        else:
            action_choice_prob = None

        logged_dataset = {
            "user_id": user_id,
            "item_id": item_id,
            "context": context,
            "query": query,
            "action": action,
            "action_choice_prob": action_choice_prob,
            "prompt": prompt,
            "sentence": output_sentence,
            "expected_reward": expected_reward,
            "reward": reward,
            "logging_policy": policy,
        }

        if return_meta_data:
            meta_data = {
                "size": n_samples,
                "reward_type": self.reward_type,
                "reward_std": self.reward_std,
                "candidate_prompts": self.candidate_actions_loader.prompts,
                "user_embeddings": self.user_embeddings,
                "item_embeddings": self.query_embeddings,
                "prompt_embeddings": self.prompt_embeddings,
            }
            output = (logged_dataset, meta_data)

        else:
            output = logged_dataset

        return output

    def calc_expected_reward_given_action(
        self,
        action: Optional[torch.Tensor],
        user_id: Optional[torch.Tensor] = None,
        item_id: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        query: Optional[Union[Sentence, Tokens]] = None,
        return_cpu_tensor: bool = True,
    ):
        """Calculate expected reward given action chosen by a policy.

        Parameters
        -------
        action: torch.Tensor, shape (n_samples, )
            Discrete actions chosen by the given policy.

        user_id: torch.Tensor, shape (n_samples, ), default=None
            User id. Either user_id or context must be given, depending on the choice of reward simulator.

        item_id: torch.Tensor, shape (n_samples, ), default=None
            Item id. Either item_id or query must be given, depending on the choice of reward simulator.

        context: torch.Tensor, shape (n_samples, dim_context), default=None
            Context of each user. Either user_id or context must be given, depending on the choice of reward simulator.

        query: Sentence or Tokens, shape (n_samples, ), default=None
            Query given by users. Either item_id or query must be given, depending on the choice of reward simulator.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

        Return
        -------
        expected_reward: torch.Tensor, shape (n_samples, )
            Expected reward given context, query, and action.

        """
        if action is not None:
            prompt = list(itemgetter(*action)(self.action_list))
        else:
            prompt = None

        output_sentence = self.frozen_llm.generate_output_sentence(
            query=query,
            prompt=prompt,
        )
        baseline_sentence = self.frozen_llm.generate_output_sentence(
            query=query,
            prompt=None,
        )

        expected_reward = self.reward_simulator.calc_expected_reward(
            user_id=user_id,
            item_id=item_id,
            context=context,
            query=query,
            action=action,
            sentence=output_sentence,
            baseline_sentence=baseline_sentence,
            return_cpu_tensor=return_cpu_tensor,
        )
        return expected_reward

    def sample_reward_given_action(
        self,
        action: Optional[torch.Tensor],
        user_id: Optional[torch.Tensor] = None,
        item_id: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        query: Optional[Union[Sentence, Tokens, torch.Tensor]] = None,
        return_cpu_tensor: bool = True,
    ):
        """Sample reward given action chosen by a policy.

        Parameters
        -------
        action: torch.Tensor, shape (n_samples, )
            Discrete actions chosen by the given policy.

        user_id: torch.Tensor, shape (n_samples, ), default=None
            User id. Either user_id or context must be given, depending on the choice of reward simulator.

        item_id: torch.Tensor, shape (n_samples, ), default=None
            Item id. Either item_id or query must be given, depending on the choice of reward simulator.

        context: torch.Tensor, shape (n_samples, dim_context), default=None
            Context of each user. Either user_id or context must be given, depending on the choice of reward simulator.

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, dim_query), default=None
            Query given by users. Either item_id or query must be given, depending on the choice of reward simulator.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

        Return
        -------
        reward: torch.Tensor, shape (n_samples, )
            Sampled reward.

        """
        expected_reward = self.calc_expected_reward_given_action(
            action=action,
            user_id=user_id,
            item_id=item_id,
            context=context,
            query=query,
            return_cpu_tensor=return_cpu_tensor,
        )
        return self._sample_reward(expected_reward)

    def calc_expected_reward_given_output(
        self,
        sentence: Union[Sentence, Tokens, torch.Tensor],
        user_id: Optional[torch.Tensor] = None,
        item_id: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        query: Optional[Union[Sentence, Tokens]] = None,
        action: Optional[torch.Tensor] = None,
        return_cpu_tensor: bool = True,
    ):
        """Culculate expected reward given action chosen by a policy.

        Parameters
        -------
        sentence: Sentence or Tokens or torch.Tensor, shape (n_samples, dim_sentence)
            Output sentence generated by query and action using frozen LLMs.

        user_id: torch.Tensor, shape (n_samples, ), default=None
            User id. Either user_id or context must be given, depending on the choice of reward simulator.

        item_id: torch.Tensor, shape (n_samples, ), default=None
            Item id. Either item_id or query must be given, depending on the choice of reward simulator.

        context: torch.Tensor, shape (n_samples, dim_context), default=None
            Context of each user. Either user_id or context must be given, depending on the choice of reward simulator.

        query: Sentence or Tokens, shape (n_samples, ), default=None
            Query given by users. Either item_id or query must be given, depending on the choice of reward simulator.

        action: torch.Tensor, shape (n_samples, ), default=None
            Discrete actions chosen by the given policy. Required depending on the choice of reward simulator.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

        Return
        -------
        expected_reward: torch.Tensor, shape (n_samples, )
            Expected reward given content, query, and sentence.

        """
        baseline_sentence = self.frozen_llm.generate_output_sentence(
            query=query,
            prompt=None,
        )
        expected_reward = self.reward_simulator.calc_expected_reward(
            user_id=user_id,
            item_id=item_id,
            context=context,
            query=query,
            action=action,
            sentence=sentence,
            baseline_sentence=sentence,
            return_cpu_tensor=return_cpu_tensor,
        )
        return expected_reward

    def sample_reward_given_output(
        self,
        sentence: Union[Sentence, Tokens, torch.Tensor],
        user_id: Optional[torch.Tensor] = None,
        item_id: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        query: Optional[Union[Sentence, Tokens, torch.Tensor]] = None,
        action: Optional[torch.Tensor] = None,
        return_cpu_tensor: bool = True,
    ):
        """Sample reward given action chosen by a policy.

        Parameters
        -------
        sentence: Sentence or Tokens, shape (n_samples, dim_sentence)
            Output sentence generated by query and action using frozen LLMs.

        user_id: torch.Tensor, shape (n_samples, ), default=None
            User id. Either user_id or context must be given, depending on the choice of reward simulator.

        item_id: torch.Tensor, shape (n_samples, ), default=None
            Item id. Either item_id or query must be given, depending on the choice of reward simulator.

        context: torch.Tensor, shape (n_samples, dim_context), default=None
            Context of each user. Either user_id or context must be given, depending on the choice of reward simulator.

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, dim_query), default=None
            Query given by users. Either item_id or query must be given, depending on the choice of reward simulator.

        action: torch.Tensor, shape (n_samples, ), default=None
            Discrete actions chosen by the given policy. Required depending on the choice of reward simulator.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

        Return
        -------
        reward: torch.Tensor, shape (n_samples, )
            Sampled reward.

        """
        expected_reward = self.calc_expected_reward_given_output(
            sentence=sentence,
            user_id=user_id,
            item_id=item_id,
            context=context,
            query=query,
            action=action,
            return_cpu_tensor=return_cpu_tensor,
        )
        return self._sample_reward(expected_reward)

    def calc_expected_policy_value(
        self,
        policy: Optional[Policy],
        n_samples_to_approximate: int = 10000,
    ):
        """Sample dataset given data collection policy.

        Parameters
        -------
        policy: Policy
            Policy that chooses discrete prompt (index) as action. If None is given, the function returns the no-prompt-baseline's performance (i.e., generate sentences without prompts).

        n_samples_to_approximate: int, default = 10000 (> 0).
            Number of samples to approximate the policy value.

        Return
        -------
        policy_value: float
            Approximated policy value.

        """
        (
            user_id,
            item_id,
            context,
            query,
        ) = self.context_query_loader.sample_context_and_query(
            n_samples=n_samples_to_approximate,
            is_test=True,
        )

        if policy is not None:
            action = policy.sample_action(
                user_id=user_id,
                item_id=item_id,
                context=context,
                query=query,
            )
        else:
            action = None

        expected_reward = self.calc_expected_reward_given_action(
            action=action,
            user_id=user_id,
            item_id=item_id,
            context=context,
            query=query,
            return_cpu_tensor=False,
        )
        return expected_reward.mean().item()

    def calc_skyline_policy_value(
        self,
        n_samples_to_approximate: int = 100,
    ):
        """Sample dataset given data collection policy.

        Parameters
        -------
        n_samples_to_approximate: int, default=100 (> 0)
            Number of samples to approximate the policy value. (The default value is small because this operation takes time to compute.)

        Return
        -------
        policy_value: float
            Approximated policy value.

        """
        (
            user_id,
            item_id,
            context,
            query,
        ) = self.context_query_loader.sample_context_and_query(
            n_samples=n_samples_to_approximate,
            is_test=True,
        )
        baseline_sentence = self.frozen_llm.generate_output_sentence(
            query=query,
            prompt=None,
        )
        # baseline_sentence = None
        expected_reward = self.reward_simulator.calc_skyline_expected_reward(
            user_id=user_id,
            item_id=item_id,
            context=context,
            query=query,
            baseline_sentence=baseline_sentence,
            frozen_llm=self.frozen_llm,
            action_list=self.action_list,
        )
        return expected_reward.mean().item()
