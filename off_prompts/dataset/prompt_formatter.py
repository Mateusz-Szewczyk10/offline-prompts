"""Implementations of the class to format prompts."""
from typing import Union, Optional, Any, Dict
from dataclasses import dataclass

import torch

from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoTokenizer,
)

from .base import BasePromptFormatter
from ..types import Tokens
from ..utils import tokenize


@dataclass
class MovielensPromptFormatter(BasePromptFormatter):
    """Class for formatting prompt and queries to input to the frozen LLM.
    
    Bases: :class:`off_prompts.dataset.BasePromptFormatter`

    Imported as: :class:`off_prompts.dataset.MovielensPromptFormatter`

    Note
    -------
    By default, given ``query`` and ``prompt``, this class format the prompt as follows.

    f"Broadly describe in a sentence the genres of the movie without including the name or any specifics of the movie.\n
      Title: {query}\n
      Keyword: {prompt}\n
      Movie description: "

    Parameters
    -------
    tokenizer: PreTrainedTokenizer or PretrainedTokenizerFast, default=None
        Tokenizer of the (transformers') frozen LLM.

    tokenizer_kwargs: dict, default=None
        Kwargs of the tokenizers.

    prefix_prompt: str, default="Broadly describe in a sentence the genres of the movie without including the name or any specifics of the movie.\nTitle: "
        System prompt to attach to the given keyword prompt. The prompt is formatted as f"{prefix_prompt}{query}{mid_prompt}{query}{postfix_prompt}".

    mid_prompt: str, default="\nKeyword: "
        System prompt to attach to the given keyword prompt. The prompt is formatted as f"{prefix_prompt}{query}{mid_prompt}{query}{postfix_prompt}".

    postfix_prompt: str, default="\nMovie description: "
        System prompt to attach to the given keyword prompt. The prompt is formatted as f"{prefix_prompt}{query}{mid_prompt}{query}{postfix_prompt}".

    device: str, default="cuda"
        Device.
    
    """
    tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None
    tokenizer_kwargs: Optional[Dict[str, Any]] = None
    prefix_prompt: str = "Broadly describe in a sentence the genres of the movie without including the name or any specifics of the movie.\nTitle: "
    mid_prompt: str = "\nKeyword: "
    postfix_prompt: str = "\nMovie description: "
    device: str = "cuda"

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                truncation=True,
                do_lower_case=True,
                use_fast=True,
            )
        if self.tokenizer_kwargs is None:
            self.tokenizer_kwargs = {
                "add_special_tokens": True,
                "padding": True,
                "truncation": True,
                "max_length": 20,
                "return_tensors": "pt",
            }

        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        self.tokenizer_kwargs["max_length"] = 22
        self.prefix_prompt_tokens = tokenize(
            self.prefix_prompt,
            tokenizer=self.tokenizer,
            tokenizer_kwargs=self.tokenizer_kwargs,
            device=self.device,
        )

        self.tokenizer_kwargs["max_length"] = 2
        self.mid_prompt_tokens = tokenize(
            self.mid_prompt,
            tokenizer=self.tokenizer,
            tokenizer_kwargs=self.tokenizer_kwargs,
            device=self.device,
        )

        self.tokenizer_kwargs["max_length"] = 3
        self.postfix_prompt_tokens = tokenize(
            self.postfix_prompt,
            tokenizer=self.tokenizer,
            tokenizer_kwargs=self.tokenizer_kwargs,
            device=self.device,
        )

    def format_tokens(
        self, query_tokens: Tokens, prompt_tokens: Optional[Tokens],
    ):
        """Format prompt tokens.
        
        Parameters
        -------
        query_tokens: Tokens
            Tokens of the query.

        prompt_tokens: Tokens
            Tokens of the prompt.

        Return
        -------
        concat_tokens: Tokens
            Tokens of the whole prompts (which is combined with the system prompt).

        """
        concat_tokens = {}

        if prompt_tokens is None:  # generate sentence without keyword prompts
            for key in query_tokens:
                n_samples = len(query_tokens[key])
                concat_tokens[key] = torch.cat(
                    [
                        self.prefix_prompt_tokens[key].expand(n_samples, -1),
                        query_tokens[key],
                        self.postfix_prompt_tokens[key].expand(n_samples, -1),
                    ],
                    dim=1,
                )
        else:
            for key in query_tokens:
                n_samples = len(query_tokens[key])
                concat_tokens[key] = torch.cat(
                    [
                        self.prefix_prompt_tokens[key].expand(n_samples, -1),
                        query_tokens[key],
                        self.mid_prompt_tokens[key].expand(n_samples, -1),
                        prompt_tokens[key],
                        self.postfix_prompt_tokens[key].expand(n_samples, -1),
                    ],
                    dim=1,
                )
        
        return concat_tokens
