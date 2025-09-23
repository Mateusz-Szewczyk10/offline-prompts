"""Implementations of the (fine-tuned) reward simulator."""
from typing import Optional, Union, Any, Dict
from operator import itemgetter
from tqdm.auto import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator

from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedModel,
    AutoTokenizer,
    AutoModel,
)

from .base import BaseRewardSimulator, BaseFrozenLLM, BaseEncoder
from ..utils import torch_seed, tokenize, to_device, RewardSimulatorDataset
from ..types import Sentence, Tokens


class CollaborativeFilteringRewardSimulator(BaseRewardSimulator):
    """Simple collaborative filtering based reward simulator.

    Parameters
    -------
    n_users: int
        Number of users.

    n_items: int
        Number of items.

    dim_emb: int, default=10 (> 0)
        Dimension of user and item embeddings.

    hidden_dim: int, default=100 (> 0)
        Dimension of the hidden layer of MLP.

    device: str, default="cuda"
        Device.

    random_state: int, default=None
        Random state.

    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim_emb: int = 10,
        hidden_dim: int = 100,
        device: str = "cuda",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.device = device

        if random_state is not None:
            torch_seed(random_state, device=self.device)

        self.user_embedding = nn.Embedding(n_users, dim_emb).to(device)
        self.item_embedding = nn.Embedding(n_items, dim_emb).to(device)

        self.l1 = nn.Linear(2 * dim_emb, hidden_dim).to(device)
        self.l2 = nn.Linear(hidden_dim, 1).to(device)
        self.relu = nn.ReLU()

        self.user_bias = nn.Embedding(n_users, 1).to(device)
        self.item_bias = nn.Embedding(n_items, 1).to(device)

        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.l1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.l2.weight, nonlinearity="relu")

    def forward(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        **kwargs,
    ):
        """Produce logit values using user and item id embeddings.

        Parameters
        -------
        user_id: torch.Tensor, shape (n_samples, )
            User id.

        item_id: torch.Tensor, shape (n_samples, )
            Item id.

        Return
        -------
        logits: torch.Tensor, shape (n_samples, )
            Logit value.

        """
        user_id = to_device(user_id, device=self.device)
        item_id = to_device(item_id, device=self.device)

        user_embedding = self.user_embedding(user_id)
        item_embedding = self.item_embedding(item_id)

        x = torch.cat((user_embedding, item_embedding), dim=1)
        x = self.relu(self.l1(x))
        interaction = self.l2(x).squeeze()

        user_bias = self.user_bias(user_id).squeeze(1)
        item_bias = self.item_bias(item_id).squeeze(1)

        logits = interaction + user_bias + item_bias
        return self.sigmoid(logits)

    def calc_expected_reward(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        return_cpu_tensor: bool = True,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Calculate expeected reward.

        Parameters
        -------
        user_id: torch.Tensor, shape (n_sample, )
            User id.

        item_id: torch.Tensor, shape (n_sample, )
            Item id.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        expected_reward: torch.Tensor, shape (n_samples, )
            Expected reward.

        """
        if calc_gradient:
            expected_reward = self(user_id, item_id)
        else:
            with torch.no_grad():
                expected_reward = self(user_id, item_id)

        return expected_reward


class TransformerRewardSimulator(BaseRewardSimulator):
    """Transformer-based reward simulator.

    Bases: :class:`off_prompts.dataset.BaseRewardSimulator`

    Imported as: :class:`off_prompts.dataset.TransformerRewardSimulator`

    Parameters
    -------
    n_users: int
        Number of users.

    n_items: int
        Number of items.

    dim_emb: int, default=10 (> 0)
        Dimension of feature embeddings.

    hidden_dim: int, default=100 (> 0)
        Dimension of the hidden layer of MLP.

    base_model: PreTrainedModel, default=None
        Base transformer model. When `None` is given, "distilbert-base-uncased" will be used.

    tokenizer: PreTrainedTokenizer or PreTrainedTokenizerFast, default=None
        Tokenizer. When `None` is given, "distilbert-base-uncased" will be used.

    tokenizer_kwargs: dict of params, default=None.
        Dictionary containing args of tokenizer. (e.g., `max_length`)

    device: str, default="cuda"
        Device.

    random_state: int, default=None
        Random state.

    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim_emb: int = 10,
        hidden_dim: int = 100,
        base_model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        self.device = device

        if random_state is not None:
            torch_seed(random_state, device=self.device)

        if base_model is None:
            base_model = AutoModel.from_pretrained(
                "distilbert-base-uncased",
            )
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "distilbert-base-uncased",
                truncation=True,
                do_lower_case=True,
                use_fast=True,
            )
        if tokenizer_kwargs is None:
            self.tokenizer_kwargs = {
                "add_special_tokens": True,
                "padding": True,
                "truncation": True,
                "max_length": 20,
                "return_tensors": "pt",
            }

        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        self.base_model.to(device)

        self.user_embedding = nn.Embedding(n_users, dim_emb).to(device)
        self.item_encoder = nn.Linear(self.base_model.config.hidden_size, dim_emb).to(
            device
        )

        self.l1 = nn.Linear(2 * dim_emb, hidden_dim).to(device)
        self.l2 = nn.Linear(hidden_dim, 1).to(device)
        self.relu = nn.ReLU()

        self.user_bias = nn.Embedding(n_users, 1).to(device)
        self.item_bias = nn.Embedding(n_items, 1).to(device)

        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_normal_(self.l1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.l2.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.item_encoder.weight, nonlinearity="relu")

    def _reward_function(
        self,
        expected_reward: torch.Tensor,
        baseline_reward: torch.Tensor,
    ):
        """Reward function (normalized by "no-prompt baseline").

        Parameters
        -------
        expected_reward: torch.Tensor, shape (n_samples, )
            Expected reward of sentence generated with some prompt.

        baseline_reward: torch.Tensor, shape (n_samples, )
            Expected reward of sentence generated without any prompts (i.e., "no-prompt baseline").

        Parameters
        -------
        normalized_reward: torch.Tensor, shape (n_samples, )
            Expected reward of sentence generated with some prompt normalized by the no-prompt baseline.

        """
        expected_reward = (expected_reward - baseline_reward) * 10
        return expected_reward.float()

    def forward(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        item_tokens: Tokens,
        is_finetuning: bool = False,
        **kwargs,
    ):
        """Produce logit values using a Transformer model.

        Parameters
        -------
        user_id: torch.Tensor, shape (n_samples, )
            User id.

        item_id: torch.Tensor, shape (n_samples, )
            Item id.

        item_tokens: Tokens, shape (n_samples, )
            Tokens of item description.

        is_finetuning: bool, default=False.
            Whether to finetune the base model (or to use the frozen model).

        Return
        -------
        logits: torch.Tensor, shape (n_samples, )
            Logit value.

        """
        user_id = to_device(user_id, device=self.device)
        item_id = to_device(item_id, device=self.device)
        item_tokens = to_device(item_tokens, device=self.device)

        user_embedding = self.user_embedding(user_id)

        user_bias = self.user_bias(user_id).squeeze()
        item_bias = self.item_bias(item_id).squeeze()

        if is_finetuning:
            item_embedding = self.base_model(**item_tokens)
            item_embedding = item_embedding[0][:, 0]
        else:
            with torch.no_grad():
                item_embedding = self.base_model(**item_tokens)
                item_embedding = item_embedding[0][:, 0]

        item_embedding = self.relu(self.item_encoder(item_embedding))

        x = torch.cat((user_embedding, item_embedding), dim=1)
        x = self.relu(self.l1(x))
        interaction = self.l2(x).squeeze()

        logits = interaction + user_bias + item_bias
        return self.sigmoid(logits)

    def calc_expected_reward(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        sentence: Union[Sentence, Tokens],
        baseline_sentence: Union[Sentence, Tokens],
        batch_size: int = 10000,
        return_cpu_tensor: bool = False,
        calc_gradient: bool = False,
        is_finetuning: bool = False,
        **kwargs,
    ):
        """Calculate expeected reward.

        Parameters
        -------
        user_id: torch.Tensor, shape (n_sample, )
            User id.

        item_id: torch.Tensor, shape (n_sample, )
            Item id.

        sentence: Sentence or Tokens, shape (n_samples, )
            Sentence generated by frozen LLMs.

        baseline_sentence: Sentence or Tokens, shape (n_samples, )
            Baseline sentence generated by frozen LLMs (without prompts).

        batch_size: int, default=10000 (> 0)
            Batch size.

        return_cpu_tensor: bool, default=False
            Whether to return output as a cpu tensor.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        is_finetuning: bool, default=False.
            Whether to finetune the base model (or to use the frozen model).

        Return
        -------
        expected_reward: torch.Tensor, shape (n_samples, )
            Expected reward.

        """
        if isinstance(sentence, list):
            tokens = tokenize(
                sentence,
                tokenizer=self.tokenizer,
                tokenizer_kwargs=self.tokenizer_kwargs,
                device=self.device,
            )
        else:
            tokens = sentence

        if isinstance(baseline_sentence, list):
            baseline_tokens = tokenize(
                baseline_sentence,
                tokenizer=self.tokenizer,
                tokenizer_kwargs=self.tokenizer_kwargs,
                device=self.device,
            )
        else:
            baseline_tokens = baseline_sentence

        n_samples = len(tokens[list(tokens.keys())[0]])
        if n_samples > batch_size:
            reward_simulator_dataset = RewardSimulatorDataset(
                user_id=user_id,
                item_id=item_id,
                sentence_tokens=tokens,
            )
            dataloader = DataLoader(
                reward_simulator_dataset, batch_size=batch_size, shuffle=False
            )

            # accelerator = Accelerator()
            # model, dataloader = accelerator.prepare(self, dataloader)

            expected_rewards = []
            for batch_ in tqdm(dataloader, desc="Inference batches of the frozen LLM"):
                # batch_ = to_device(batch_, device=accelerator.device)
                user_id_ = batch_["user_id"]
                item_id_ = batch_["item_id"]
                tokens_ = batch_["sentence_tokens"]

                if calc_gradient:
                    expected_reward = self(
                        user_id_,
                        item_id_,
                        tokens_,
                        return_cpu_tensor=return_cpu_tensor,
                        is_finetuning=is_finetuning,
                    )
                else:
                    with torch.no_grad():
                        expected_reward = self(
                            user_id_,
                            item_id_,
                            tokens_,
                            return_cpu_tensor=return_cpu_tensor,
                            is_finetuning=is_finetuning,
                        )
                expected_rewards.append(expected_reward)

            baseline_dataset = RewardSimulatorDataset(
                user_id=user_id,
                item_id=item_id,
                sentence_tokens=baseline_tokens,
            )
            dataloader = DataLoader(
                baseline_dataset, batch_size=batch_size, shuffle=False
            )

            # accelerator = Accelerator()
            # model, dataloader = accelerator.prepare(self, baseline_dataloader)

            baseline_rewards = []
            for batch_ in tqdm(dataloader, desc="Inference batches of the frozen LLM"):
                # batch_ = to_device(batch_, device=accelerator.device)
                user_id_ = batch_["user_id"]
                item_id_ = batch_["item_id"]
                baseline_tokens_ = batch_["sentence_tokens"]

                with torch.no_grad():
                    baseline_reward = self(
                        user_id_,
                        item_id_,
                        baseline_tokens_,
                        return_cpu_tensor=return_cpu_tensor,
                        is_finetuning=is_finetuning,
                    )
                baseline_rewards.append(baseline_reward)

            expected_reward = torch.cat(expected_rewards, dim=0)
            baseline_reward = torch.cat(baseline_rewards, dim=0)

        else:
            if calc_gradient:
                expected_reward = self(
                    user_id,
                    item_id,
                    tokens,
                    return_cpu_tensor=return_cpu_tensor,
                    is_finetuning=is_finetuning,
                )
            else:
                with torch.no_grad():
                    expected_reward = self(
                        user_id,
                        item_id,
                        tokens,
                        return_cpu_tensor=return_cpu_tensor,
                        is_finetuning=is_finetuning,
                    )

            with torch.no_grad():
                baseline_reward = self(
                    user_id,
                    item_id,
                    baseline_tokens,
                    return_cpu_tensor=return_cpu_tensor,
                    is_finetuning=is_finetuning,
                )

        expected_reward = self._reward_function(
            expected_reward=expected_reward,
            baseline_reward=baseline_reward,
        )
        return expected_reward

    def calc_skyline_expected_reward(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        query: Sentence,
        frozen_llm: BaseFrozenLLM,
        action_list: Sentence,
        return_cpu_tensor: bool = False,
        calc_gradient: bool = False,
        is_finetuning: bool = False,
        **kwargs,
    ):
        """Calculate expeected reward.

        Parameters
        -------
        user_id: torch.Tensor, shape (n_sample, )
            User id.

        item_id: torch.Tensor, shape (n_sample, )
            Item id.

        query: Sentence, shape (n_samples, )
            Query (e.g., movie title).

        frozen_llm: BaseFrozenLLM
            Frozen llm to simulate sentence generation.

        action_list: Sentence, shape (n_actions, )
            Mapping from discrete action to embedding.

        return_cpu_tensor: bool, default=False
            Whether to return output as a cpu tensor.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        is_finetuning: bool, default=False.
            Whether to finetune the base model (or to use the frozen model).

        Return
        -------
        expected_reward: torch.Tensor, shape (n_samples, )
            Expected reward.

        """
        n_samples = len(query)
        n_actions = len(action_list)
        assert n_samples < 1000

        device = "cpu" if return_cpu_tensor else self.device

        # skyline
        expected_rewards = torch.zeros((n_samples, n_actions), device=device)
        for a_ in range(n_actions):
            action_ = torch.full((n_samples,), a_)
            prompt_ = list(itemgetter(*action_)(action_list))

            sentence_ = frozen_llm.generate_output_sentence(
                query=query,
                prompt=prompt_,
            )
            tokens_ = tokenize(
                sentence_,
                tokenizer=self.tokenizer,
                tokenizer_kwargs=self.tokenizer_kwargs,
                device=self.device,
            )

            with torch.no_grad():
                expected_rewards[:, a_] = self(
                    user_id,
                    item_id,
                    tokens_,
                    return_cpu_tensor=return_cpu_tensor,
                    is_finetuning=False,
                )

        skyline_reward = expected_rewards.max(dim=1)[0]

        # baseline
        baseline_sentence = frozen_llm.generate_output_sentence(
            query=query,
            prompt=None,
        )
        baseline_tokens = tokenize(
            baseline_sentence,
            tokenizer=self.tokenizer,
            tokenizer_kwargs=self.tokenizer_kwargs,
            device=self.device,
        )

        with torch.no_grad():
            baseline_reward = self(
                user_id,
                item_id,
                baseline_tokens,
                return_cpu_tensor=return_cpu_tensor,
                is_finetuning=False,
            )

        expected_reward = self._reward_function(
            expected_reward=skyline_reward,
            baseline_reward=baseline_reward,
        )
        return expected_reward


class PromptCossimRewardSimulator(BaseRewardSimulator):
    """Simple reward simulator base on the cosine similarity between (users, queries) and prompts.

    Note
    -------
    user_embeddings, query_embeddings, and prompt_embeddings should have the same feature dimension to use this class.

    Parameters
    -------
    user_embeddings: torch.Tensor, shape (n_users, dim_context)
        Embeddings of users.

    query_embeddings: torch.Tensor, shape (n_items, dim_query)
        Embeddings of queries.

    prompt_embeddings: torch.Tensor, shape (n_actions, dim_prompt)
        Embeddings of prompts.

    device: str, default="cuda"
        Device.

    random_state: int, default=None
        Random state.

    """

    def __init__(
        self,
        user_embeddings: torch.Tensor,
        query_embeddings: torch.Tensor,
        prompt_embeddings: torch.Tensor,
        device: str = "cuda",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.device = device

        if random_state is not None:
            torch_seed(random_state, device=self.device)

        self.user_embedding = user_embeddings.to(device)
        self.item_embedding = query_embeddings.to(device)
        self.prompt_embedding = prompt_embeddings.to(device)

        self.cossim = nn.CosineSimilarity()

    def forward(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        action: torch.Tensor,
        **kwargs,
    ):
        """Produce logit values using user and item id embeddings.

        Parameters
        -------
        user_id: torch.Tensor, shape (n_samples, )
            User id.

        item_id: torch.Tensor, shape (n_samples, )
            Item id.

        action: torch.Tensor, shape (n_samples, )
            Index of prompts.

        Return
        -------
        logits: torch.Tensor, shape (n_samples, )
            Logit value.

        """
        user_id = to_device(user_id, device=self.device)
        item_id = to_device(item_id, device=self.device)
        action = to_device(action, device=self.device)

        user_embedding = self.user_embedding[user_id]
        item_embedding = self.item_embedding[item_id]
        prompt_embedding = self.prompt_embedding[action]

        logits = self.cossim(user_embedding, prompt_embedding)
        return logits * 100

    def calc_expected_reward(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        action: torch.Tensor,
        return_cpu_tensor: bool = True,
        **kwargs,
    ):
        """Calculate expeected reward.

        Parameters
        -------
        user_id: torch.Tensor, shape (n_sample, )
            User id.

        item_id: torch.Tensor, shape (n_sample, )
            Item id.

        action: torch.Tensor, shape (n_samples, )
            Action.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

        Return
        -------
        expected_reward: torch.Tensor, shape (n_samples, )
            Expected reward.

        """
        with torch.no_grad():
            expected_reward = self(user_id, item_id, action)

        if return_cpu_tensor:
            expected_reward = expected_reward.to("cpu")

        return expected_reward

    def calc_skyline_expected_reward(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        query: Sentence,
        action_list: Sentence,
        return_cpu_tensor: bool = False,
        calc_gradient: bool = False,
        is_finetuning: bool = False,
        **kwargs,
    ):
        """Calculate expeected reward.

        Parameters
        -------
        user_id: torch.Tensor, shape (n_sample, )
            User id.

        item_id: torch.Tensor, shape (n_sample, )
            Item id.

        action_list: Sentence, shape (n_actions, )
            Mapping from discrete action to embedding.

        return_cpu_tensor: bool, default=False
            Whether to return output as a cpu tensor.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        is_finetuning: bool, default=False.
            Whether to finetune the base model (or to use the frozen model).

        Return
        -------
        expected_reward: torch.Tensor, shape (n_samples, )
            Expected reward.

        """
        n_samples = len(user_id)
        n_actions = len(action_list)

        expected_rewards = torch.zeros((n_samples, n_actions), device=self.device)
        for a_ in range(n_actions):
            with torch.no_grad():
                action_ = torch.full((n_samples,), a_, device=self.device)
                expected_rewards[:, a_] = self(user_id, item_id, action_)

        if return_cpu_tensor:
            expected_rewards = expected_rewards.to("cpu")

        skyline = expected_rewards.max(dim=1)[0]
        return skyline


class SentenceCossimRewardSimulator(BaseRewardSimulator):
    """Simple reward simulator base on the cosine similarity between users and sentences.

    Note
    -------
    user_embeddings, query_embeddings, and prompt_embeddings should have the same feature dimension to use this class.

    This class only support context-independent embeddings of sentences.

    Parameters
    -------
    user_embeddings: torch.Tensor, shape (n_users, dim_context)
        Embeddings of users.

    query_embeddings: torch.Tensor, shape (n_users, dim_query)
        Embeddings of queries.

    sentence_encoder: BaseEncoder
        Encoder of sentences.

    device: str, default="cuda"
        Device.

    random_state: int, default=None
        Random state.

    """

    def __init__(
        self,
        user_embeddings: torch.Tensor,
        query_embeddings: torch.Tensor,
        sentence_encoder: BaseEncoder,
        device: str = "cuda",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.device = device

        if random_state is not None:
            torch_seed(random_state, device=self.device)

        self.user_embedding = user_embeddings.to(device)
        self.item_embedding = query_embeddings.to(device)
        self.sentence_encoder = sentence_encoder

        self.cossim = nn.CosineSimilarity()

    def forward(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        sentence: Union[Sentence, Tokens],
        **kwargs,
    ):
        """Produce logit values using user and item id embeddings.

        Parameters
        -------
        user_id: torch.Tensor, shape (n_samples, )
            User id.

        item_id: torch.Tensor, shape (n_samples, )
            Item id.

        sentence: Sentence or Tokens, shape (n_samples, )
            Sentence.

        Return
        -------
        logits: torch.Tensor, shape (n_samples, )
            Logit value.

        """
        user_id = to_device(user_id, device=self.device)
        user_embedding = self.user_embedding[user_id]
        item_embedding = self.item_embedding[item_id]
        sentence_embedding = self.sentence_encoder.encode(sentence)

        sentence_logits = self.cossim(user_embedding, sentence_embedding)
        item_logits = self.cossim(user_embedding, item_embedding)
        return (sentence_logits - item_logits) * 100

    def calc_expected_reward(
        self,
        user_id: torch.Tensor,
        sentence: Union[Sentence, Tokens],
        return_cpu_tensor: bool = True,
        **kwargs,
    ):
        """Calculate expeected reward.

        Parameters
        -------
        user_id: torch.Tensor, shape (n_sample, )
            User id.

        sentence: Sentence or Tokens, shape (n_samples, )
            Sentence.

        return_cpu_tensor: bool, default=True
            Whether to return output as a cpu tensor.

        Return
        -------
        expected_reward: torch.Tensor, shape (n_samples, )
            Expected reward.

        """
        with torch.no_grad():
            expected_reward = self(user_id, sentence)

        if return_cpu_tensor:
            expected_reward = expected_reward.to("cpu")

        return expected_reward

    def calc_skyline_expected_reward(
        self,
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        query: Union[Sentence, Tokens],
        frozen_llm: BaseFrozenLLM,
        action_list: Sentence,
        return_cpu_tensor: bool = False,
        calc_gradient: bool = False,
        is_finetuning: bool = False,
        **kwargs,
    ):
        """Calculate expeected reward.

        Parameters
        -------
        user_id: torch.Tensor, shape (n_sample, )
            User id.

        item_id: torch.Tensor, shape (n_sample, )
            Item id.

        query: Sentence or Tokens, shape (n_samples, )
            Query.

        frozen_llm: BaseFrozenLLM
            Frozen llm to simulate sentence generation.

        action_list: Sentence
            Mapping from discrete action to embedding.

        return_cpu_tensor: bool, default=False.
            Whether to return output as a cpu tensor.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        is_finetuning: bool, default=False.
            Whether to finetune the base model (or to use the frozen model).

        Return
        -------
        expected_reward: torch.Tensor, shape (n_samples, )
            Expected reward.

        """
        n_samples = len(user_id)
        n_actions = len(action_list)

        expected_rewards = torch.zeros((n_samples, n_actions), device=self.device)
        for a_ in range(n_actions):
            with torch.no_grad():
                action_ = torch.full((n_samples,), a_)
                prompt_ = list(itemgetter(*action_)(action_list))
                sentence_ = frozen_llm.generate_output_sentence(
                    query=query,
                    prompt=prompt_,
                )
                expected_rewards[:, a_] = self(user_id, item_id, sentence_)

        if return_cpu_tensor:
            expected_rewards = expected_rewards.to("cpu")

        skyline = expected_rewards.max(dim=1)[0]
        return skyline
