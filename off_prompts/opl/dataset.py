"""Dataset class for policy learning."""
from typing import Optional
from operator import itemgetter

import torch
from torch.utils.data import Dataset

from ..dataset.base import BaseFrozenLLM, BaseEncoder
from ..types import Sentence
from ..utils import tokenize


class TorchLoggedDataset(Dataset):
    """Formatter of logged dataset to fit torch dataloader.

    Bases: :class:`torch.utils.data.Dataset`

    Imported as: :class:`off_prompts.opl.dataset.TorchLoggedDataset`

    Parameters
    -------
    context: torch.Tensor, shape (n_samples, dim_context), default=None
        User context.

    query: Sentence, shape (n_samples, ), default=None
        Item query.

    user_id: torch.Tensor, shape (n_samples, ), default=None
        User id.

    item_id: torch.Tensor, shape (n_samples, ), default=None
        Item id.

    action: torch.Tensor, shape (n_samples, ), default=None
        Action (i.e., index of prompt).

    sentence: Sentence, shape (n_samples, ), default=None
        Generated sentence description of item.

    reward: torch.Tensor, shape (n_samples, ), default=None
        Reward.

    predicted_reward: torch.Tensor, shape (n_samples, ), default=None
        Predicted reward.

    logging_action_choice_prob: torch.Tensor, shape (n_samples, ), default=None
        Action choice probability of the logging policy.

    logging_marginal_density: torch.Tensor, shape (n_samples, ), default=None
        Marginalized action choice probability of the logging policy.

    action_list: Sentence, shape (n_actions, ), default=None
        List of prompts or their embeddings.

    query_embeddings: torch.Tensor, shape (n_items, dim_query_emb), default=None
        Mapping from item id to its (query) embeddings.

    prompt_embeddings: torch.Tensor, shape (n_items, dim_prompt_emb), default=None
        Mapping from item id to its (prompt) embeddings.

    query_encoder: BaseEncoder, default=None
        Encoder of queries.

    prompt_encoder: BaseEncoder, default=None
        Encoder of prompts.

    sentence_encoder: BaseEncoder, default=None
        Encoder of sentences.

    frozen_llm: BaseFrozenLLM, default=None
        Frozen LLM.

    """

    def __init__(
        self,
        context: Optional[torch.Tensor] = None,
        query: Optional[Sentence] = None,
        user_id: Optional[torch.Tensor] = None,
        item_id: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        sentence: Optional[Sentence] = None,
        reward: Optional[torch.Tensor] = None,
        predicted_reward: Optional[torch.Tensor] = None,
        logging_action_choice_prob: Optional[torch.Tensor] = None,
        logging_marginal_density: Optional[torch.Tensor] = None,
        action_list: Optional[Sentence] = None,
        prompt_embeddings: Optional[torch.Tensor] = None,
        query_embeddings: Optional[torch.Tensor] = None,
        query_encoder: Optional[BaseEncoder] = None,
        prompt_encoder: Optional[BaseEncoder] = None,
        sentence_encoder: Optional[BaseEncoder] = None,
        frozen_llm: Optional[BaseFrozenLLM] = None,
    ):
        self.context = context
        self.action = action
        self.reward = reward
        self.predicted_reward = predicted_reward
        self.logging_action_choice_prob = logging_action_choice_prob
        self.logging_marginal_density = logging_marginal_density

        self.query = query
        self.sentence = sentence

        self.prompt = action_list
        self.prompt_for_frozen_llm = action_list
        self.query_for_frozen_llm = query

        if prompt_embeddings is not None:
            self.prompt = prompt_embeddings

        elif action_list is not None:
            if prompt_encoder is not None:
                self.prompt = prompt_encoder.encode(self.prompt)

        if action_list is not None:
            if frozen_llm is not None:
                self.prompt_for_frozen_llm = tokenize(
                    self.prompt_for_frozen_llm,
                    tokenizer=frozen_llm.tokenizer,
                    tokenizer_kwargs=frozen_llm.tokenizer_kwargs,
                    device=frozen_llm.device,
                )

        if query_embeddings is not None and item_id is not None:
            self.query = query_embeddings[item_id]

        elif query is not None:
            if query_encoder is not None:
                self.query = query_encoder.encode(self.query)

        if query is not None:
            if frozen_llm is not None:
                self.query_for_frozen_llm = tokenize(
                    self.query_for_frozen_llm,
                    tokenizer=frozen_llm.tokenizer,
                    tokenizer_kwargs=frozen_llm.tokenizer_kwargs,
                    device=frozen_llm.device,
                )

        if sentence is not None and context is not None and query is not None:
            if sentence_encoder is not None:
                self.sentence = sentence_encoder.encode(self.sentence, self.context, self.query)

    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        batch_context = torch.nan
        batch_action = torch.nan
        batch_reward = torch.nan
        batch_predicted_reward = torch.nan
        batch_logging_action_choice_prob = torch.nan
        batch_logging_marginal_density = torch.nan
        batch_prompt = torch.nan
        batch_prompt_for_frozen_llm = torch.nan
        batch_query = torch.nan
        batch_query_for_frozen_llm = torch.nan
        batch_sentence = torch.nan

        if self.context is not None:
            batch_context = self.context[idx]

        if self.action is not None:
            batch_action = self.action[idx]

        if self.reward is not None:
            batch_reward = self.reward[idx]

        if self.predicted_reward is not None:
            batch_predicted_reward = self.predicted_reward[idx]

        if self.logging_action_choice_prob is not None:
            batch_logging_action_choice_prob = self.logging_action_choice_prob[idx]

        if self.logging_marginal_density is not None:
            batch_logging_marginal_density = self.logging_marginal_density[idx]

        if self.prompt is not None and self.action is not None:
            batch_prompt = self.prompt[self.action[idx]]

            if isinstance(self.prompt_for_frozen_llm, list):
                batch_prompt_for_frozen_llm = self.prompt_for_frozen_llm[self.action[idx]]
            else:
                batch_prompt_for_frozen_llm = {}
                for key in self.prompt_for_frozen_llm:
                    batch_prompt_for_frozen_llm[key] = self.prompt_for_frozen_llm[key][
                        self.action[idx]
                    ]

        if self.query is not None:
            batch_query = self.query[idx]

            if isinstance(self.query_for_frozen_llm, list):
                batch_query_for_frozen_llm = self.query_for_frozen_llm[idx]
            else:
                batch_query_for_frozen_llm = {}
                for key in self.query_for_frozen_llm:
                    batch_query_for_frozen_llm[key] = self.query_for_frozen_llm[key][
                        idx
                    ]

        if self.sentence is not None:
            batch_sentence = self.sentence[idx]

        batch = {
            "context": batch_context,
            "action": batch_action,
            "reward": batch_reward,
            "predicted_reward": batch_predicted_reward,
            "logging_action_choice_prob": batch_logging_action_choice_prob,
            "logging_marginal_density": batch_logging_marginal_density,
            "prompt": batch_prompt,  # Sentence or embeddings
            "prompt_for_frozen_llm": batch_prompt_for_frozen_llm,  # Sentence or Token
            "query": batch_query,  # Sentence or embeddings
            "query_for_frozen_llm": batch_query_for_frozen_llm,  # Sentence or Token
            "sentence": batch_sentence,  # embeddings
        }
        return batch
