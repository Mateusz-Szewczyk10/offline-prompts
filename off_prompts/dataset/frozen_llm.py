"""Implementation of the Frozen LLMs-based sentence generator."""
from dataclasses import dataclass
from typing import Optional, Any, Dict, Union
from tqdm.auto import tqdm
import re

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from .base import BaseFrozenLLM, BasePromptFormatter
from .prompt_formatter import MovielensPromptFormatter
from ..utils import torch_seed, tokenize, to_device, FrozenLLMDataset
from ..types import Sentence, Tokens


@dataclass
class AutoFrozenLLM(BaseFrozenLLM):
    """Frozen LLMs for the recipe generation task.

    Bases: :class:`off_prompts.dataset.BaseFrozenLLM`

    Imported as: :class:`off_prompts.dataset.AutoFrozenLLM`

    Note
    -------
    This model is available only on GPUs.

    Parameters
    -------
    prompt_formatter: BasePromptFormatter
        Prompt formatter.

    model: PreTrainedModel, default=None
        Base model of the (frozen) LLM.

    tokenizer: PreTrainedTokenizer or PreTrainedTokenizerFast
        Tokenizer of the (frozen) LLM.

    tokenizer_kwargs: dict
        Tokenizer kwargs.

    frozen_llm_pattern: str, default=""
        System prompt for a frozen llm. If the system prompt is repeated in the output, this pattern will be cut off to avoid redundancy.

    device: str, default="cuda"
        Device.

    random_state: int, default=None
        Random state.

    """

    prompt_formatter: BasePromptFormatter
    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None
    tokenizer_kwargs: Optional[Dict[str, Any]] = None
    pattern: str = ""
    device: str = "cuda"
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.random_state is not None:
            torch_seed(self.random_state, device=self.device)

        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
            )
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
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)

    def generate_output_sentence(
        self,
        query: Union[Sentence, Tokens],
        prompt: Optional[Union[Sentence, Tokens]],
        max_new_tokens: int = 30,
        batch_size: int = 128,
    ):
        """Generate output sentence given query and prompt.

        Parameters
        -------
        query: Sentence or Tokens, shape (n_samples, )
            Qurey specified by users.

        prompt: Sentence or Tokens, shape (n_samples, ) or (n_samples, n_actions, ) or (n_actions, )
            Discrete prompts for each given user context and query.

        max_new_tokens: int, default=30 (> 0)
            Maximum length of the output sentence.

        batch_size: int, default=128 (> 0)
            Batch size.

        Return
        -------
        sentence: list of str, shape (n_samples, )
            Sampled sentence.

        """
        if isinstance(query, list):
            query_tokens = tokenize(
                query,
                tokenizer=self.tokenizer,
                tokenizer_kwargs=self.tokenizer_kwargs,
                device=self.device,
            )
        else:
            query_tokens = query

        if isinstance(prompt, list):
            prompt_tokens = tokenize(
                prompt,
                tokenizer=self.tokenizer,
                tokenizer_kwargs=self.tokenizer_kwargs,
                device=self.device,
            )
        else:
            prompt_tokens = prompt

        inputs = self.prompt_formatter.format_tokens(
            query_tokens=query_tokens,
            prompt_tokens=prompt_tokens,
        )

        n_samples = len(inputs[list(inputs.keys())[0]])
        if n_samples > batch_size:
            frozen_llm_dataset = FrozenLLMDataset(inputs)
            dataloader = DataLoader(
                frozen_llm_dataset, batch_size=batch_size, shuffle=False
            )

            accelerator = Accelerator()
            model, dataloader = accelerator.prepare(self.model, dataloader)

            output_sentence = []
            for batch_ in tqdm(dataloader, desc="Inference batches of the frozen LLM"):
                batch_ = to_device(batch_, device=accelerator.device)

                with torch.no_grad():
                    encoded_ = model.generate(
                        **batch_,
                        max_new_tokens=max_new_tokens,
                        num_return_sequences=1,
                    )
                    decoded_ = self.tokenizer.batch_decode(
                        encoded_,
                        skip_special_tokens=True,
                    )
                    output_sentence_ = list(
                        map(lambda text: re.split(self.pattern, text)[-1], decoded_)
                    )
                    output_sentence.extend(output_sentence_)

        else:
            with torch.no_grad():
                encoded = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=1,
                )
                decoded = self.tokenizer.batch_decode(
                    encoded,
                    skip_special_tokens=True,
                )
                output_sentence = list(
                    map(lambda text: re.split(self.pattern, text)[-1], decoded)
                )

        return output_sentence
