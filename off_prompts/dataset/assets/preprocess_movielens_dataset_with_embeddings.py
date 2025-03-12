"""Scripts for preprocessing user and query information."""
from typing import Optional
from pathlib import Path
import random
import time
import gc

import hydra
from omegaconf import DictConfig

import pandas as pd

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed

from off_prompts.dataset import TransformerEncoder

from utils import format_runtime


def process(
    setting: str,
    query_path: str,
    prompt_path: str,
    sentence_path: str,
    base_model_id: str,
    tokenizer_id: str,
    dim_emb: int,
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

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id, use_fast=use_tokenizer_fast
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.to(device)

    tokenizer_kwargs = {
        "add_special_tokens": True,
        "padding": True,
        "truncation": True,
        "max_length": 20,
        "return_tensors": "pt",
    }

    # query
    query_df = pd.read_csv(query_path)
    queries = query_df["query"].values.tolist()
    n_querys = len(queries)

    query_prefix_prompt = "Broadly describe in a sentence the genres of the movie without including the name or any specifics of the movie.\nTitle: "
    query_postfix_prompt = " "
    query_prefix_max_length = 22
    query_postfix_max_length = 1

    query_encoder = TransformerEncoder(
        base_model=base_model,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        is_causal_lm=True,
        prefix_prompt=query_prefix_prompt,
        postfix_prompt=query_postfix_prompt,
        prefix_tokens_max_length=query_prefix_max_length,
        postfix_tokens_max_length=query_postfix_max_length,
        max_length=5,
        dim_emb=dim_emb,
        device=device,
        random_state=random_state,
    )

    query_encoder.fit_pca(queries[:1000])
    query_embs = query_encoder.encode(queries)

    query_encoder.save(f"{setting}_query_pca_matrix.pt")
    torch.save(query_embs, f"{setting}_query_embs.pt")

    del query_df, queries, query_embs, query_encoder
    gc.collect()

    # prompt
    prompt_df = pd.read_csv(prompt_path)
    prompts = prompt_df["vocab"].values.tolist()
    n_prompts = len(prompts)

    prompt_prefix_prompt = "Associate the word - "
    prompt_postfix_prompt = " - in the context of movie genres"
    prompt_prefix_max_length = 4
    prompt_postfix_max_length = 7

    prompt_encoder = TransformerEncoder(
        base_model=base_model,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        is_causal_lm=True,
        prefix_prompt=prompt_prefix_prompt,
        postfix_prompt=prompt_postfix_prompt,
        prefix_tokens_max_length=prompt_prefix_max_length,
        postfix_tokens_max_length=prompt_postfix_max_length,
        max_length=2,
        dim_emb=dim_emb,
        device=device,
        random_state=random_state,
    )

    prompt_encoder.fit_pca(prompts)
    prompt_embs = prompt_encoder.encode(prompts)

    prompt_encoder.save(f"{setting}_prompt_pca_matrix.pt")
    torch.save(prompt_embs, f"{setting}_prompt_embs.pt")

    del prompt_df, prompts, prompt_embs, prompt_encoder
    gc.collect()

    # sentence
    sentence_df = pd.read_csv(sentence_path)
    sentence_df = sentence_df[:1000]

    sentences = sentence_df["description"].values.tolist()

    sentence_prefix_prompt = " "
    sentence_postfix_prompt = " "
    sentence_prefix_max_length = 1
    sentence_postfix_max_length = 1

    sentence_encoder = TransformerEncoder(
        base_model=base_model,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        is_causal_lm=True,
        prefix_prompt=prompt_prefix_prompt,
        postfix_prompt=prompt_postfix_prompt,
        prefix_tokens_max_length=prompt_prefix_max_length,
        postfix_tokens_max_length=prompt_postfix_max_length,
        max_length=20,
        dim_emb=dim_emb,
        device=device,
        random_state=random_state,
    )

    sentence_encoder.fit_pca(sentences)
    sentence_encoder.save(f"{setting}_sentence_pca_matrix.pt")


@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    print(f"The current working directory is {Path().cwd()}")
    print(f"The original working directory is {hydra.utils.get_original_cwd()}")
    print()

    process(
        setting=cfg.encoding.setting,
        query_path=cfg.encoding.query_path,
        prompt_path=cfg.encoding.prompt_path,
        sentence_path=cfg.encoding.sentence_path,
        base_model_id=cfg.encoding.base_model_id,
        tokenizer_id=cfg.encoding.tokenizer_id,
        dim_emb=cfg.encoding.dim_emb,
        use_tokenizer_fast=cfg.encoding.use_tokenizer_fast,
        random_state=cfg.encoding.random_state,
    )


if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()
    print("total runtime:", format_runtime(start, finish))
