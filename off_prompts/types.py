"""Types."""
from typing import Dict, List, Tuple, Any, Union
import torch


LoggedDataset = Dict[str, Any]
Sentence = Union[List[str], Tuple[str]]
Tokens = Dict[str, torch.Tensor]
