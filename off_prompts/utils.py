"""Useful tools."""
from collections import defaultdict
from typing import DefaultDict, Dict, Union, Optional, Any
import math

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedModel
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from .types import LoggedDataset, Sentence, Tokens


class FrozenLLMDataset(Dataset):
    """Torch dataset class for frozen LLMs.

    Bases: :class:`torch.utils.data.Dataset`

    Imported as: :class:`src.utils.FrozenLLMDataset`
    
    Parameters
    -------
    tokens: Tokens, shape (n_samples, )
        Input tokens.
    
    """
    def __init__(
        self, tokens: Tokens,
    ):
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens[list(self.tokens.keys())[0]])

    def __getitem__(self, idx):
        batch_tokens = {}
        for key in self.tokens:
            batch_tokens[key] = self.tokens[key][idx]

        return batch_tokens


class RewardSimulatorDataset(Dataset):
    """Torch dataset class for the reward simulator.
    
    Bases: :class:`torch.utils.data.Dataset`

    Imported as: :class:`src.utils.RewardSimulatorDataset`

    Parameters
    -------
    user_id: torch.Tensor, shape (n_samples, )
        User id.

    item_id: torch.Tensor, shape (n_samples, )
        Item id.

    sentence_tokens: Tokens, shape (n_samples, )
        Sentence tokens.
    
    """
    def __init__(
        self, 
        user_id: torch.Tensor,
        item_id: torch.Tensor,
        sentence_tokens: Tokens,
    ):
        self.user_id = user_id
        self.item_id = item_id
        self.sentence_tokens = sentence_tokens

    def __len__(self):
        return len(self.user_id)

    def __getitem__(self, idx):
        batch_tokens = {}
        for key in self.sentence_tokens:
            batch_tokens[key] = self.sentence_tokens[key][idx]

        batch = {
            "user_id": self.user_id[idx],
            "item_id": self.item_id[idx],
            "sentence_tokens": batch_tokens,
        }
        return batch


def to_device(
    inputs: Union[torch.Tensor, Dict[str, Any]], device: str = "cuda",
):
    """Transfer tensor to device.
    
    Parameters
    -------
    inputs: dict or dict of torch.Tensor
        Inputs.

    device: str, default="cuda"
        Device.

    Return
    -------
    outputs: dict or dict of torch.Tensor
        Tensors loaded on the device.
    
    """
    if isinstance(inputs, torch.Tensor):
        outputs = inputs.to(device)
    elif isinstance(inputs, dict):
        outputs = {key: value.to(device) for key, value in inputs.items()}
    else:
        outputs = inputs
    return outputs


def tokenize(
    inputs: Sentence,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
):
    """Tokenize sentences.
    
    Parameters
    -------
    inputs: Sentence, shape (n_samples, )
        Input sentences.

    tokenizer: PreTrainedTokenizer or PreTrainedTokenizerFast
        Tokenizer of the (frozen) LLM.

    tokenizer_kwargs: dict
        Tokenizer kwargs.

    device: str, default="cuda"
        Device.

    Return
    -------
    tokens: Tokens, shape (n_samples, )
        Tokenized sentences.
    
    """
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {"return_tensors": "pt"}

    tokens = tokenizer(inputs, **tokenizer_kwargs)
    tokens = {key: value.to(device) for key, value in tokens.items()}

    return tokens


def gaussian_kernel(distance: torch.Tensor, tau: float = 1.0):
    """Gaussian kernel.
    
    Parameters
    -------
    distance: torch.Tensor, shape (n_samples, )
        Distance between two inputs (i.e., sentences).

    tau: float, default=1.0
        Bandwidth hyperparameter of the gaussian kernel.

    Return
    -------
    weight: torch.Tensor (n_samples, )
        Weight of the gaussian kernel.

    """
    if tau <= 0:
        raise ValueError("tau should be a positive value, but found False.")

    return torch.exp(-(distance ** 2) / (2 * tau ** 2)) / math.sqrt(
        2 * math.pi * tau ** 2
    )


def uniform_kernel(distance: torch.Tensor, tau: float = 1.0):
    """Uniform kernel.
    
    Parameters
    -------
    distance: torch.Tensor, shape (n_samples, )
        Distance between two inputs (i.e., sentences).

    tau: float, default=1.0
        Bandwidth hyperparameter of the uniform kernel.

    Return
    -------
    weight: torch.Tensor (n_samples, )
        Weight of the uniform kernel.
    
    """
    if tau <= 0:
        raise ValueError("tau should be a positive value, but found False.")

    return (distance <= tau) / (2 * tau)


def torch_seed(random_state: int, device: str = "cuda:0"):
    """Set seeds of pytorch.
    
    Parameters
    -------
    random_state: int
        Random state.

    device: str, default="cuda"
        Device.
    
    """
    if device != "cpu":
        torch.cuda.manual_seed(random_state)

    torch.manual_seed(random_state)


def defaultdict_to_dict(dict_: Union[Dict[Any, Any], DefaultDict[Any, Any]]):
    """Transform a defaultdict into a corresponding dict.
    
    Parameters
    -------
    dict_: defaultdict
        Input dict (formatted as defaultdict).

    Return
    -------
    dict_: dict
        Transformed dict (formatted as dict).

    """
    if isinstance(dict_, defaultdict):
        dict_ = {key: defaultdict_to_dict(value) for key, value in dict_.items()}
    return dict_


def check_tensor(
    tensor: torch.Tensor,
    name: str,
    expected_dim: int = 1,
    expected_dtype: Optional[type] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> ValueError:
    """Input validation on tensor.

    Parameters
    -------
    tensor: object
        Input tensor to check.

    name: str
        Name of the input tensor.

    expected_dim: int, default=1
        Expected dimension of the input tensor.

    expected_dtype: {type, tuple of type}, default=None
        Expected dtype of the input tensor.

    min_val: float, default=None
        Minimum value allowed in the input tensor.

    max_val: float, default=None
        Maximum value allowed in the input tensor.

    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(
            f"{name} must be {expected_dim}D tensor, but got {type(tensor)}"
        )
    if tensor.ndim != expected_dim:
        raise ValueError(
            f"{name} must be {expected_dim}D tensor, but got {tensor.ndim}D tensor"
        )
    if expected_dtype is not None:
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"The elements of {name} must be {expected_dtype}, but got {tensor.dtype}"
            )
    if min_val is not None:
        if tensor.min() < min_val:
            raise ValueError(
                f"The elements of {name} must be larger than {min_val}, but got minimum value {tensor.min()}"
            )
    if max_val is not None:
        if tensor.max() > max_val:
            raise ValueError(
                f"The elements of {name} must be smaller than {max_val}, but got maximum value {tensor.max()}"
            )


def check_logged_feedback(logged_feedback: LoggedDataset):
    """Check keys of logged feedback.

    Parameters
    -------
    logged_feedback: LoggedDataset
        Logged data, which contains the following keys.

        .. code-block:: python

                key: [
                    context,
                    query,
                    action,
                    sentence,
                    expected_reward,
                    reward,
                    logging_policy,
                ]

            context: torch.Tensor, shape (n_samples, dim_context)
                Feature vector of each user.

            query: Sentence, shape (n_samples, )
                Propmpts or embeddings of the prompts specified by users.

            action: torch.Tensor, shape (n_samples, ) or (n_samples, dim_action)
                Discrete or continuous (soft) prompts chosen by the given policy.

            sentence: Sentence, shape (n_samples, )
                Sentence generated by some frozen LLMs using user-specified query and prompts chosen by a policy.

            expected_reward: torch.Tensor, shape (n_samples, )
                Expected reward predicted by some frozen LLMs (i.e., reward simulator).

            reward: torch.Tensor, shape (n_samples, )
                Either binary or continuous reward.

            logging_policy: BasePolicy
                Policy that chooses either discrete or continuous (soft) prompt as action.

    """
    for key in ["context", "query", "action", "sentence", "reward"]:
        if key not in logged_feedback.keys():
            raise ValueError(
                f"logged_logged_feedback must contrain {key}, but {key} is not found"
            )

    context = logged_feedback["context"]
    query = logged_feedback["query"]
    action = logged_feedback["action"]
    sentence = logged_feedback["sentence"]
    reward = logged_feedback["reward"]

    if not (len(context) == len(query) == len(action) == len(sentence) == len(reward)):
        raise ValueError(
            "content, query, action, sentence, reward must have the same data size, but found False."
        )
