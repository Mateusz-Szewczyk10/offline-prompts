"""Scripts for preprocessing datasets to train the reward simulator."""
from typing import Optional, Dict, Union
from pathlib import Path
from tqdm.auto import tqdm
import random
import time
import re

import hydra
from omegaconf import DictConfig

import torch
import pandas as pd
from sklearn.utils import shuffle

from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers import set_seed
from torch.utils.data import DataLoader
from accelerate import Accelerator

from off_prompts.types import Tokens
from off_prompts.utils import FrozenLLMDataset, tokenize, to_device
from utils import format_runtime


def filter_data(df, min_val: int = 10):
    # Filter users and items with at least 10 positive and 10 negative ratings
    filtered_users = df.groupby("user_id")["reward"].apply(
        lambda x: (x == 1).sum() >= min_val and (x == 0).sum() >= min_val
    )
    filtered_items = df.groupby("item_id")["reward"].apply(
        lambda x: (x == 1).sum() >= min_val and (x == 0).sum() >= min_val
    )
    return df[
        df["user_id"].isin(filtered_users[filtered_users].index)
        & df["item_id"].isin(filtered_items[filtered_items].index)
    ]


def generate_movie_description(
    model: PreTrainedModel,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    prefix_tokens: Tokens,
    postfix_tokens: Tokens,
    title_tokens: Tokens,
    max_new_tokens: int = 30,
    batch_size: int = 128,
):
    cat_tokens = {}
    for key in title_tokens.keys():
        n_samples = title_tokens[key].shape[0]
        cat_tokens[key] = torch.cat(
            [
                prefix_tokens[key].expand(n_samples, -1),
                title_tokens[key],
                postfix_tokens[key].expand(n_samples, -1),
            ],
            dim=1,
        )

    frozen_llm_dataset = FrozenLLMDataset(cat_tokens)
    dataloader = DataLoader(frozen_llm_dataset, batch_size=batch_size, shuffle=False)

    accelerator = Accelerator()
    model, dataloader = accelerator.prepare(model, dataloader)

    description = []
    for batch_tokens_ in tqdm(dataloader, desc="Inference batches of the frozen LLM"):
        batch_tokens_ = to_device(batch_tokens_, device=accelerator.device)

        # generate sentences
        with torch.no_grad():
            output_ = model.generate(
                **batch_tokens_, max_new_tokens=max_new_tokens, num_return_sequences=1
            )

        # Decode the generated text
        decoded_ = tokenizer.batch_decode(output_, skip_special_tokens=True)

        pattern = r"Broadly describe in a sentence the genres of the movie without including the name or any specifics of.*?\n\n"
        description_ = list(map(lambda text: re.split(pattern, text)[-1], decoded_))

        description.extend(description_)

    return description


def process(
    dataset_dir: str,
    model_id: str,
    tokenizer_id: str,
    use_tokenizer_fast: bool = False,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    if random_state is not None:
        random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed(random_state)
        set_seed(random_state)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print("processing rating data..")

    # load the rating data
    df = pd.read_csv(
        f"{dataset_dir}/ratings.dat",
        sep="::",
        names=["user_id", "item_id", "rating"],
        engine="python",
        usecols=[0, 1, 2],
    )
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)

    # exclude the rating of 4
    df = df[df["rating"] != 4]
    # Modify rating: 1 for 5 stars (positive) and 0 for 1-3 stars (negative)
    df["reward"] = (df["rating"] == 5).astype(int)

    # iteratively filter data until no further changes occur
    prev_len = len(df) + 1
    while len(df) < prev_len:
        prev_len = len(df)
        df = filter_data(df)

    # sample balanced data for each user
    balanced_data = []
    for user_id in df["user_id"].unique():
        user_data = df[df["user_id"] == user_id]
        pos_samples = user_data[user_data["reward"] == 1]
        neg_samples = user_data[user_data["reward"] == 0]

        # use the same numbers of positive/negative samples
        min_samples = min(len(pos_samples), len(neg_samples))
        balanced_data.extend(
            pos_samples.sample(n=min_samples, random_state=random_state).to_dict(
                "records"
            )
        )
        balanced_data.extend(
            neg_samples.sample(n=min_samples, random_state=random_state).to_dict(
                "records"
            )
        )

    balanced_df = pd.DataFrame(balanced_data)
    balanced_df = balanced_df.drop(columns=["rating"])
    balanced_df = shuffle(balanced_df, random_state=random_state)

    print("retrieving item info..")

    # load the item metadata
    # depending on the dataset version, the delimiter might be '|' or '\t'
    items_df = pd.read_csv(
        f"{dataset_dir}/movies.dat",
        sep="::",
        encoding="latin-1",
        names=["item_id", "title"],
        usecols=["item_id", "title"],
    )

    all_item_id = items_df.item_id.values
    all_title = items_df.title.values

    # using only the items observed in the rating data
    filtered_item_id = []
    filtered_title = []
    unique_item_ids = list(balanced_df["item_id"].unique())

    for item_id, title in zip(all_item_id, all_title):
        if item_id in unique_item_ids:
            filtered_item_id.append(item_id)
            filtered_title.append(title)

    print("generating item description with frozen llm..")

    # generate description of movies
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=use_tokenizer_fast)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    tokenizer_kwargs = {
        "add_special_tokens": True,
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
    }

    prefix_prompt = "Broadly describe in a sentence the genres of the movie without including the name or any specifics of the movie. \nTitle"
    postfix_prompt = "\nMovie description: "

    tokenizer_kwargs["max_length"] = 22
    prefix_tokens = tokenize(
        prefix_prompt,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        device=device,
    )

    tokenizer_kwargs["max_length"] = 3
    postfix_tokens = tokenize(
        postfix_prompt,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        device=device,
    )

    tokenizer_kwargs["max_length"] = 5
    title_tokens = tokenize(
        filtered_title,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        device=device,
    )

    description = generate_movie_description(
        model=model,
        tokenizer=tokenizer,
        prefix_tokens=prefix_tokens,
        postfix_tokens=postfix_tokens,
        title_tokens=title_tokens,
    )

    # dictionary to map item_id to title and description
    item_id_to_title = {}
    item_id_to_description = {}
    for i in range(len(filtered_item_id)):
        item_id_to_title[filtered_item_id[i]] = filtered_title[i]
        item_id_to_description[filtered_item_id[i]] = description[i]

    balanced_df["title"] = balanced_df["item_id"].map(item_id_to_title)
    balanced_df["description"] = balanced_df["item_id"].map(item_id_to_description)

    # re-index dataset
    unique_user_ids = list(balanced_df["user_id"].unique())
    unique_item_ids = list(balanced_df["item_id"].unique())

    user_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_user_ids)}
    item_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_item_ids)}

    balanced_df["user_id"] = balanced_df["user_id"].map(user_id_mapping)
    balanced_df["item_id"] = balanced_df["item_id"].map(item_id_mapping)
    balanced_df = balanced_df.sample(frac=1.0)

    balanced_df.to_csv("movielens_preprocessed_data.csv", index=False)
    print(f"data size: {len(balanced_df)}")
    print(f"# of unique users: {balanced_df['user_id'].nunique()}")
    print(f"# of unique items: {balanced_df['item_id'].nunique()}")


@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    print(f"The current working directory is {Path().cwd()}")
    print(f"The original working directory is {hydra.utils.get_original_cwd()}")
    print()

    process(
        dataset_dir=cfg.setting.dataset_dir,
        model_id=cfg.setting.model_id,
        tokenizer_id=cfg.setting.tokenizer_id,
        use_tokenizer_fast=cfg.setting.use_tokenizer_fast,
        random_state=cfg.setting.random_state,
    )


if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()
    print("total runtime:", format_runtime(start, finish))
