"""Implementations of the classes to define function in the data generation process."""
from dataclasses import dataclass
from typing import Optional
from operator import itemgetter
import gc

import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from .base import BaseCandidateActionsLoader
from .base import BaseContextQueryLoader
from ..utils import torch_seed


@dataclass
class DefaultCandidateActionsLoader(BaseCandidateActionsLoader):
    """Generator of candidate actions.

    Bases: :class:`off_prompts.dataset.BaseCandidateActionsLoader`

    Imported as: :class:`off_prompts.dataset.DefaultCandidateActionsLoader`

    Parameters
    -------
    n_actions: int, default=1000 (> 0)
        Number of discrete actions.

    path_to_candidate_prompts: str, default="assets/movielens_benchmark_prompts.csv"
        Path to the csv file specifying vocabulary used in simulation.

    path_to_prompt_embeddings: str, default=None
        Path to the pt file specifying prompt embeddings used in simulation.

    random_state: int, default=None
        Random state.

    """

    n_actions: int = 1000
    path_to_candidate_prompts: str = "assets/movielens_benchmark_prompts.csv"
    path_to_prompt_embeddings: Optional[str] = None
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.random_state is not None:
            torch_seed(self.random_state)

        df = pd.read_csv(self.path_to_candidate_prompts)
        if len(df.columns) != 1:
            raise ValueError("DataFrame should have a single column, but found False.")
        if self.n_actions > len(df):
            raise ValueError("n_actions must be smaller than the length of the prompt dataframe, but found False.")

        idx = torch.multinomial(torch.ones((len(df),)), num_samples=self.n_actions,)
        self.prompts = df.loc[idx].values[:, 0].tolist()
        self.action_list = self.prompts

        if self.path_to_prompt_embeddings is not None:
            prompt_embeddings = torch.load(self.path_to_prompt_embeddings)
            if len(df) != len(prompt_embeddings):
                raise ValueError("The number of prompts and that of prompt embeddings must be the same, but found False.")
            
            self.prompt_embeddings = prompt_embeddings[idx]

        else:
            self.prompt_embeddings = None


@dataclass
class DefaultContextQueryLoader(BaseContextQueryLoader):
    """Generator of context and query.

    Bases: :class:`off_prompts.dataset.BaseContextQueryLoader`

    Imported as: :class:`off_prompts.dataset.DefaultContextQueryLoader`

    Parameters
    -------
    path_to_user_embeddings: str, default="assets/movielens_transformer_user_embeddings.pt"
        Path to the csv file specifying the user context.

    path_to_queries: str, default="assets/movielens_query.csv"
        Path to the csv file specifying the query.

    path_to_query_embeddings: str, default=None
        Path to the csv file specifying the query embeddings.

    path_to_interaction_data: str, default=None
        Path to the csv file specifying the logs of (user, item) data.

    test_ratio: float, default=0.5 (> 0.0)
        Ratio of the test data.

    shuffle: bool, False
        Whether to shuffle data when splitting train/test data.

    device: str, default="cuda"
        Device of the tensor.

    random_state: int, default=None
        Random state.

    """

    path_to_user_embeddings: str = "assets/movielens_transformer_user_embeddings.pt"
    path_to_queries: str = "assets/movielens_query.csv"
    path_to_query_embeddings: Optional[str] = None
    path_to_interaction_data: Optional[str] = None
    test_ratio: float = 0.5
    shuffle: bool = False
    device: str = "cuda:0"
    random_state: Optional[int] = None

    def __post_init__(self):
        if self.random_state is not None:
            torch_seed(self.random_state, device=self.device)

        self.user_embeddings = torch.load(self.path_to_user_embeddings)
        self.queries = pd.read_csv(self.path_to_queries)
        self.queries = self.queries["query"].to_list()

        if self.path_to_query_embeddings is not None:
            self.query_embeddings = torch.load(self.path_to_query_embeddings)
            if len(self.queries) != len(self.query_embeddings):
                raise ValueError("The number of queries and that of query embeddings must be the same, but found False.")

            self.query_embeddings_dict = {
                query: self.query_embeddings[i] for i, query in enumerate(self.queries)
            }

        else:
            self.query_embeddings = None

        if self.path_to_interaction_data is not None:
            df = pd.read_csv(self.path_to_interaction_data)
            self.user_ids = torch.LongTensor(df["user_id"].to_numpy())
            self.item_ids = torch.LongTensor(df["item_id"].to_numpy())
            del df
            gc.collect()

        if self.path_to_interaction_data is None:
            if self.test_ratio == 0:
                self.train_user_embeddings = (
                    self.test_user_embeddings
                ) = self.user_embeddings
            else:
                self.train_user_embeddings, self.test_user_embeddings = train_test_split(
                    self.user_embeddings,
                    test_size=self.test_ratio,
                    shuffle=self.shuffle,
                    random_state=self.random_state,
                )
            self.n_train_users = len(self.train_user_embeddings)
            self.n_test_users = len(self.test_user_embeddings)
            
        else:
            if self.test_ratio == 0:
                self.train_data_id = self.test_data_id = torch.arange(len(self.user_ids))
            else:
                self.train_data_id, self.test_data_id = train_test_split(
                    torch.arange(len(self.user_ids)),
                    test_size=self.test_ratio,
                    shuffle=self.shuffle,
                    random_state=self.random_state,
                )
            self.n_train_data = len(self.train_data_id)
            self.n_test_data = len(self.test_data_id)

        self.n_users, self.dim_context = self.user_embeddings.shape
        self.n_queries = len(self.queries)

    def sample_context_and_query(
        self, 
        n_samples: int, 
        is_test: bool = False, 
        return_query_embeddings: bool = False,
    ):
        """Sample context and query.

        Parameters
        -------
        n_samples: int
            Number of samples.

        is_test: bool, default=False
            Whether to use test samples.

        return_query_embeddings: bool, default=False
            Whether to return query with its embeddings.

        Return
        -------
        user_id: torch.Tensor, shape (n_samples, )
            User id.

        item_id: torch.Tensor, shape (n_samples, )
            Item id.

        context: torch.Tensor, shape (n_samples, dim_context)
            Context vector of each user.

        query: Sentence, shape (n_samples, )
            Query sentence given by users.

        query_embeddings: torch.Tensor, shape (n_samples, dim_query)
            Query vector given by users.

        """
        if return_query_embeddings and self.query_embeddings is None:
            raise RuntimeError("query_embeddings is not given. Please initialize the class with query embeddings.")

        if self.path_to_interaction_data is None:
            if is_test:
                user_id = torch.randint(self.n_test_users, (n_samples,))
                context = self.test_user_embeddings[user_id]
            else:
                user_id = torch.randint(self.n_train_users, (n_samples,))
                context = self.train_user_embeddings[user_id]

            item_id = torch.randint(self.n_queries, (n_samples,))
            query = list(itemgetter(*item_id)(self.queries))

        else:
            if is_test:
                data_id = torch.randint(self.n_test_data, (n_samples, ))
            else:
                data_id = torch.randint(self.n_train_data, (n_samples, ))

            user_id = self.user_ids[data_id]
            item_id = self.item_ids[data_id]

            context = self.user_embeddings[user_id]
            query = list(itemgetter(*item_id)(self.queries))

        if return_query_embeddings:
            query_embs = self.query_embeddings[item_id]
            output = (user_id, item_id, context, query, query_embs)
        else:
            output = (user_id, item_id, context, query)

        return output
