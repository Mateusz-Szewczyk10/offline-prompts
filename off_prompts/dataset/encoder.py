"""Implementations of the encoders of texts, e.g, sentence, query, and prompt."""
from tqdm.auto import tqdm
from typing import Any, Union, Optional, Dict, List, Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator

from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedModel,
)
from transformers import (
    AutoTokenizer,
    AutoModel,
)

from .base import BaseEncoder
from ..types import Sentence, Tokens
from ..utils import torch_seed, tokenize, to_device


class BatchDataset(Dataset):
    """Dataset class used in the encoder models.

    Parameters
    -------
    input: Sentence or Tokens, shape (n_samples, )
        Textual data.

    """

    def __init__(
        self,
        input: Union[Sentence, Tokens],
    ):
        self.input = input

    def __len__(self):
        if isinstance(self.input, list):
            length_ = len(self.input)
        else:
            length_ = len(self.input["input_ids"])

        return length_

    def __getitem__(self, idx):
        if isinstance(self.input, list):
            input_ = self.input[idx]
        else:
            input_ = {}
            for key in self.input:
                input_[key] = self.input[key][idx]

        return input_


class TransformerEncoder(BaseEncoder):
    """Transformer based encoder.

    Bases: :class:`off_prompts.policy.BaseEncoder`

    Imported as: :class:`off_prompts.policy.TransformerEncoder`

    Parameters
    -------
    base_model: PreTrainedModel or Pretrained, default=None
        Base transformer model. When `None` is given, "mistralai/Mistral-7B-Instruct-v0.2" will be used.

    tokenizer: PreTrainedTokenizer or PreTrainedTokenizerFast, default=None
        Tokenizer. When `None` is given, "mistralai/Mistral-7B-Instruct-v0.2" will be used.

    tokenizer_kwargs: dict of params, default=None
        Dictionary containing args of tokenizer. (e.g., `max_length`)

    is_causal_lm: bool, default=False
        Whether the model is for causal LM.

    prefix_prompt: str, default=" "
        System prompt to attach to the given movie title/prompt. The format is "{prefix_prompt}{input_prompt}{postfix_prompt}".

    postfix_prompt: str, default=" "
        System prompt to attach to the given movie title/prompt. The format is "{prefix_prompt}{input_prompt}{postfix_prompt}".

    prefix_tokens_max_length: int, default=1 (> 0)
        Length of the prefix prompt. (Longer prompts will be cutted off.)

    postfix_tokens_max_length: int, default=1 (> 0)
        Length of the postfix prompt. (Longer prompts will be cutted off.)

    max_length: int, default=20 (> 0)
        Max length of the (entire) input tokens. Required only when tokenizer_kwargs is not given.

    dim_emb: int, default=10 (> 0)
        Dimension of embeddings.

    device: str, default="cuda"
        Device.

    random_state: int, default=None
        Random state.

    """

    def __init__(
        self,
        base_model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        is_causal_lm: bool = False,
        prefix_prompt: str = " ",
        postfix_prompt: str = " ",
        prefix_tokens_max_length: int = 1,
        postfix_tokens_max_length: int = 1,
        max_length: int = 20,
        dim_emb: int = 10,
        device: str = "cuda",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        self.is_causal_lm = is_causal_lm
        self.max_length = max_length
        self.dim_emb = dim_emb
        self.device = device

        if random_state is not None:
            torch_seed(random_state, device=self.device)

        if base_model is None:
            self.base_model = AutoModel.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
            )
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-Instruct-v0.2",
                truncation=True,
                do_lower_case=True,
                use_fast=True,
            )
        if tokenizer_kwargs is None:
            self.tokenizer_kwargs = {
                "add_special_tokens": True,
                "padding": True,
                "truncation": True,
                "max_length": self.max_length,
                "return_tensors": "pt",
            }

        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        self.base_model.to(self.device)

        self.tokenizer_kwargs["max_length"] = prefix_tokens_max_length
        self.prefix_tokens = tokenize(
            prefix_prompt,
            tokenizer=self.tokenizer,
            tokenizer_kwargs=self.tokenizer_kwargs,
            device=self.device,
        )

        self.tokenizer_kwargs["max_length"] = postfix_tokens_max_length
        self.postfix_tokens = tokenize(
            postfix_prompt,
            tokenizer=self.tokenizer,
            tokenizer_kwargs=self.tokenizer_kwargs,
            device=self.device,
        )

        self.tokenizer_kwargs["max_length"] = self.max_length
        self.pca_matrix = None

    def _original_embeddings(
        self,
        inputs: Union[Sentence, Tokens],
        batch_size: int = 128,
    ):
        """Get original (high-dimensional) embeddings.

        Parameters
        -------
        inputs: Sentence or Tokens, shape (n_samples, )
            Input texts, such as sentence, query, and prompt.

        batch_size: int, default=128 (> 0)
            Batch size.

        Return
        -------
        embeddings: torch.Tensor, shape (n_samples, dim_emb)
            Original (high dimensional) embeddings of the input texts.

        """
        if isinstance(inputs, list):
            inputs = tokenize(
                inputs,
                tokenizer=self.tokenizer,
                tokenizer_kwargs=self.tokenizer_kwargs,
                device=self.device,
            )

        inputs = self.format_tokens(inputs)

        batch_dataset = BatchDataset(inputs)
        batch_dataloader = DataLoader(
            batch_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        accelerator = Accelerator()
        model, batch_dataloader = accelerator.prepare(self.base_model, batch_dataloader)

        embs = []
        for inputs_ in tqdm(
            batch_dataloader, desc="Encoder: Inference batches of the frozen LLM"
        ):
            inputs_ = to_device(inputs_, device=accelerator.device)

            with torch.no_grad():
                if self.is_causal_lm:
                    output_ = model(**inputs_, output_hidden_states=True)
                    emb_ = output_.hidden_states[-1]
                else:
                    output_ = model(**inputs_)
                    emb_ = output_.last_hidden_state

            embs.append(emb_.mean(dim=1))  # mean pooling

        embs = torch.cat(embs, dim=0)
        return embs

    def fit_pca(
        self,
        inputs: Union[Sentence, Tokens],
        batch_size: int = 128,
    ):
        """Fit PCA to map high-dimensional features to a low-dimensional vector.

        Parameters
        -------
        inputs: Sentence or Tokens, shape (n_samples, )
            Input texts, such as sentence, query, and prompt.

        batch_size: int, default=128 (> 0)
            Batch size.

        """
        emb = self._original_embeddings(inputs, batch_size=batch_size)
        U, S, V = torch.pca_lowrank(emb, q=self.dim_emb)
        self.pca_matrix = V[:, : self.dim_emb]

    def encode(
        self,
        inputs: Union[Sentence, Tokens],
        context: Optional[torch.Tensor] = None,
        query: Optional[Union[Sentence, Tokens, torch.Tensor]] = None,
        batch_size: int = 128,
        **kwargs,
    ):
        """Encode input sentence to a low-dimensional vector.

        Parameters
        -------
        inputs: Sentence or Tokens, shape (n_samples, )
            Input texts, such as sentence, query, and prompt.

        context: torch.Tensor, shape (n_samples, dim_context), default=None
            For API consistency.

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, dim_sentence), default=None
            For API consistency.

        batch_size: int, default=128 (> 0)
            Batch size.

        Return
        -------
        embeddings: torch.Tensor, shape (n_samples, dim_emb)
            Low dimensional embeddings of the input texts after applying PCA.

        """
        if self.pca_matrix is None:
            raise RuntimeError("pca is not fitted. Please call `fit_pca` first.")

        emb = self._original_embeddings(inputs, batch_size=batch_size)
        emb = torch.matmul(emb, self.pca_matrix)
        return emb

    def format_tokens(
        self,
        input_tokens: Tokens,
    ):
        """Format prompt tokens.

        Parameters
        -------
        input_tokens: Tokens, shape (n_samples, )
            Tokens of the query.

        Return
        -------
        concat_tokens: Tokens, shape (n_samples, )
            Tokens of the whole prompts (which is combined with the system prompt).

        """
        concat_tokens = {}
        for key in input_tokens:
            n_samples = len(input_tokens[key])

            concat_tokens[key] = torch.cat(
                [
                    self.prefix_tokens[key].expand(n_samples, -1),
                    input_tokens[key],
                    self.postfix_tokens[key].expand(n_samples, -1),
                ],
                dim=1,
            )

        return concat_tokens


class NNSentenceEncoder(nn.Module):
    """NN-based sentence encoder that is used on top of the transformer encoder.

    Bases: :class:`off_prompts.dataset.BaseEncoder

    Imported as: :class:`off_prompts.dataset.NNSentenceEncoder

    Parameters
    -------
    base_transformer_encoder: TransformerEncoder
        Transformer encoder model (of sentence).

    query_encoder: BaseEncoder
        Encoder of the queries.

    is_context_dependent: bool, default=False
        Whether to use context dependent encoder.

    dim_context: int, default=20 (> 0)
        Dimension of the context.

    dim_emb: int, default=20 (> 0)
        Dimention of the sentence embeddings.

    hidden_dim: int, default=100 (> 0)
        Dimension of the hidden layer of MLP.

    device: str, default="cuda"
        Device of the model.

    random_state: int, default=None.
        Random state.

    """

    def __init__(
        self,
        base_transformer_encoder: TransformerEncoder,
        query_encoder: Optional[BaseEncoder] = None,
        is_context_dependent: bool = False,
        dim_context: int = 20,
        dim_emb: int = 20,
        hidden_dim: int = 100,
        device: str = "cuda",
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.transformer_encoder = base_transformer_encoder
        self.query_encoder = query_encoder
        self.is_context_dependent = is_context_dependent
        self.dim_emb = dim_emb
        self.device = device

        if random_state is not None:
            torch_seed(random_state, device=self.device)

        if is_context_dependent:
            l2_input_dim = hidden_dim + dim_context + query_encoder.dim_emb
            l2_output_dim = hidden_dim
        else:
            l2_input_dim = hidden_dim
            l2_output_dim = dim_emb

        self.l1 = nn.Linear(
            self.transformer_encoder.pca_matrix.shape[0], hidden_dim
        ).to(self.device)
        self.l2 = nn.Linear(l2_input_dim, l2_output_dim).to(self.device)
        self.l3 = nn.Linear(hidden_dim, dim_emb).to(self.device)
        self.relu = nn.ReLU()

    def forward(
        self,
        emb: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        query: Optional[Union[Sentence, Tokens, torch.Tensor]] = None,
    ):
        """Produce logit values using a Transformer model.

        Parameters
        -------
        emb: torch.Tensor, shape (n_samples, dim_emb)
            Embeddings of the sentence.

        context: torch.Tensor, shape (n_samples, dim_context), default=None
            Context of each user.

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, dim_sentence), default=None
            Query given by the users.

        Return
        -------
        output_embeddings: torch.Tensor, shape (n_samples, dim_output_emb)
            Transformed embeddings.

        """
        emb = to_device(emb, device=self.device)
        context = to_device(context, device=self.device)
        query = to_device(query, device=self.device)

        if self.is_context_dependent:
            if context is None:
                raise ValueError(
                    "context must be given when is_context_dependent is True."
                )

            if query is None:
                raise ValueError(
                    "query must be given when is_context_dependent is True."
                )

            if not isinstance(query, torch.Tensor):
                query = self.query_encoder.encode(query)

        x = self.relu(self.l1(emb))

        if self.is_context_dependent:
            x = torch.cat([context, query, x], dim=-1)
            x = self.relu(self.l2(x))
            output = self.l3(x)

        else:
            output = self.l2(x)

        return output

    def encode(
        self,
        inputs: Union[Sentence, Tokens],
        context: Optional[torch.Tensor] = None,
        query: Optional[Union[Sentence, Tokens, torch.Tensor]] = None,
        batch_size: int = 128,
        calc_gradient: bool = False,
        **kwargs,
    ):
        """Encode input sentence to a low-dimensional vector.

        Parameters
        -------
        inputs: Sentence or Tokens, shape (n_samples, )
            Input texts, such as sentence, query, and prompt.

        context: torch.Tensor, shape (n_samples, dim_context), default=None
            Context of each user.

        query: Sentence or Tokens or torch.Tensor, shape (n_samples, dim_sentence), default=None
            Query given by the users.

        batch_size: int, default=128 (> 0)
            Batch size.

        calc_gradient: bool, default=False.
            Whether to calculate the gradient or not.

        Return
        -------
        embeddings: torch.Tensor, shape (n_samples, dim_emb)
            Low dimensional embeddings of the input texts after passing through an MLP model.

        """
        with torch.no_grad():
            # emb = self.transformer_encoder.encode(inputs, batch_size=batch_size,)
            emb = self.transformer_encoder._original_embeddings(
                inputs,
                batch_size=batch_size,
            )

        if calc_gradient:
            emb = self(emb, context, query)

        else:
            with torch.no_grad():
                emb = self(emb, context, query)

        return emb

    def load(self, path: Path):
        """Load model."""
        self.load_state_dict(torch.load(path))

    def save(self, path: Path):
        """Save model."""
        torch.save(self.state_dict(), path)
