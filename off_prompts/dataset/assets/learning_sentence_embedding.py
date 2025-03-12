"""Scripts for learning sentence embeddings."""
from typing import Optional, Union, Any, Dict
from pathlib import Path
import time

import hydra
from omegaconf import DictConfig

import torch
from torch.optim import Adam
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from off_prompts.dataset import SemiSyntheticDataset
from off_prompts.dataset.base import BaseEncoder
from off_prompts.dataset.encoder import TransformerEncoder, NNSentenceEncoder
from off_prompts.dataset.prompt_formatter import MovielensPromptFormatter
from off_prompts.dataset.function import DefaultContextQueryLoader
from off_prompts.dataset.function import DefaultCandidateActionsLoader
from off_prompts.dataset.frozen_llm import AutoFrozenLLM
from off_prompts.dataset.reward_simulator import TransformerRewardSimulator

from sentence_embedding_learner import SentenceEmbeddingLearner

from utils import format_runtime


def process(
    setting: str,
    base_frozen_llm_id: str,
    frozen_llm_tokenizer_id: str,
    base_reward_model_id: str,
    reward_model_tokenizer_id: str,
    path_to_user_embeddings: str,
    path_to_finetuned_params: str,
    path_to_queries: str,
    path_to_query_embeddings: str,
    path_to_candidate_prompts: str,
    path_to_prompt_embeddings: str,
    path_to_query_encoder: str,
    path_to_prompt_encoder: str,
    path_to_sentence_encoder: str,
    use_tokenizer_fast: bool = False,
    is_sentence_dependent: bool = True,
    n_actions: int = 1632,  # all
    reward_type: str = "continuous",
    reward_std: float = 0.0,
    dim_query: int = 20,
    dim_prompt: int = 20,
    dim_sentence: int = 20,
    device: str = "cuda",
    random_state: Optional[int] = None,
):
    # load dataset
    context_query_loader = DefaultContextQueryLoader(
        path_to_user_embeddings=path_to_user_embeddings,
        path_to_queries=path_to_queries,
        path_to_query_embeddings=path_to_query_embeddings,
        device=device,
        random_state=random_state,
    )
    candidate_actions_loader = DefaultCandidateActionsLoader(
        n_actions=n_actions,
        path_to_candidate_prompts=path_to_candidate_prompts,
        path_to_prompt_embeddings=path_to_prompt_embeddings,
        random_state=random_state,
    )

    frozen_llm_model = AutoModelForCausalLM.from_pretrained(
        base_frozen_llm_id,
    )
    frozen_llm_tokenizer = AutoTokenizer.from_pretrained(
        frozen_llm_tokenizer_id,
        truncation=True,
        do_lower_case=True,
        use_fast=True,
    )
    frozen_llm_tokenizer_kwargs = {
        "add_special_tokens": True,
        "truncation": True,
        "max_length": 20,
        "pad_to_max_length": True,
        "return_tensors": "pt",
    }
    frozen_llm_prompt_formatter = MovielensPromptFormatter(
        tokenizer=frozen_llm_tokenizer,
        tokenizer_kwargs=frozen_llm_tokenizer_kwargs,
        device=device,
    )
    pattern = (
        r"Broadly describe in a sentence the genres of the movie without including the name or any specifics of.*?\n\n",
    )[0]

    frozen_llm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    frozen_llm_model.resize_token_embeddings(len(frozen_llm_tokenizer))
    frozen_llm_model.to(device)

    frozen_llm = AutoFrozenLLM(
        prompt_formatter=frozen_llm_prompt_formatter,
        model=frozen_llm_model,
        tokenizer=frozen_llm_tokenizer,
        tokenizer_kwargs=frozen_llm_tokenizer_kwargs,
        pattern=pattern,
        device=device,
        random_state=random_state,
    )

    reward_simulator_base_model = AutoModel.from_pretrained(base_reward_model_id,)
    reward_simulator_tokenizer = AutoTokenizer.from_pretrained(
        reward_model_tokenizer_id, truncation=True, do_lower_case=True, use_fast=True,
    )
    reward_simulator_tokenizer_kwargs = {
        "add_special_tokens": True,
        "truncation": True,
        "max_length": 20,
        "pad_to_max_length": True,
        "return_tensors": "pt",
    }

    reward_simulator_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    reward_simulator_base_model.resize_token_embeddings(len(reward_simulator_tokenizer))
    reward_simulator_base_model.to(device)

    reward_simulator = TransformerRewardSimulator(
        n_users=context_query_loader.n_users,
        n_items=context_query_loader.n_queries,
        base_model=reward_simulator_base_model,
        tokenizer=reward_simulator_tokenizer,
        tokenizer_kwargs=reward_simulator_tokenizer_kwargs,
        dim_emb=20,
        device=device,
        random_state=random_state,
    )
    reward_simulator.load_state_dict(torch.load(path_to_finetuned_params))

    dataset = SemiSyntheticDataset(
        context_query_loader=context_query_loader,
        candidate_actions_loader=candidate_actions_loader,
        frozen_llm=frozen_llm,
        reward_simulator=reward_simulator,
        frozen_llm_prompt_formatter=frozen_llm_prompt_formatter,
        reward_type=reward_type,
        reward_std=reward_std,
        device=device,
        random_state=random_state,
    )

    # load base encoders
    base_model = dataset.frozen_llm.model
    tokenizer = dataset.frozen_llm.tokenizer
    tokenizer_kwargs = dataset.frozen_llm.tokenizer_kwargs

    query_prefix_prompt = "Broadly describe in a sentence the genres of the movie without including the name or any specifics of the movie.\nTitle: "
    query_postfix_prompt = " "
    query_prefix_max_length = 22
    query_postfix_max_length = 1

    prompt_prefix_prompt = "Associate the word - "
    prompt_postfix_prompt = " - in the context of movie genres"
    prompt_prefix_max_length = 4
    prompt_postfix_max_length = 7

    sentence_prefix_prompt = " "
    sentence_postfix_prompt = " "
    sentence_prefix_max_length = 1
    sentence_postfix_max_length = 1

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
        dim_emb=dim_query,
        device=device,
        random_state=random_state,
    )
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
        dim_emb=dim_prompt,
        device=device,
        random_state=random_state,
    )
    sentence_encoder = TransformerEncoder(
        base_model=base_model,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        is_causal_lm=True,
        prefix_prompt=sentence_prefix_prompt,
        postfix_prompt=sentence_postfix_prompt,
        prefix_tokens_max_length=sentence_prefix_max_length,
        postfix_tokens_max_length=sentence_postfix_max_length,
        max_length=20,
        dim_emb=dim_sentence,
        device=device,
        random_state=random_state,
    )

    query_encoder.load(path_to_query_encoder)
    prompt_encoder.load(path_to_prompt_encoder)
    sentence_encoder.load(path_to_sentence_encoder)

    context_independent_indicator = "_ind" if not is_context_dependent else ""

    # fit NN sentence encoder
    nn_sentence_encoder = NNSentenceEncoder(
        base_transformer_encoder=sentence_encoder,
        query_encoder=query_encoder,
        is_context_dependent=is_context_dependent,
        dim_context=dataset.dim_context,
        dim_emb=dim_sentence,
        device=device,
        random_state=random_state,
    )
    sentence_embedding_learner = SentenceEmbeddingLearner(
        model=nn_sentence_encoder,
        env=dataset,
        optimizer=Adam,
        optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.0},
        random_state=random_state,
    )
    nn_sentence_encoder = sentence_embedding_learner.fit_online(
        n_epochs=3,
        save_path=f"{setting}_nn_sentence_encoder{context_independent_indicator}.pt",
        random_state=random_state,
    )


@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    print(f"The current working directory is {Path().cwd()}")
    print(f"The original working directory is {hydra.utils.get_original_cwd()}")
    print()

    process(
        setting=cfg.sentence_embedding.setting,
        path_to_user_embeddings=cfg.sentence_embedding.path_to_user_embeddings,
        path_to_queries=cfg.sentence_embedding.path_to_queries,
        path_to_query_embeddings=cfg.sentence_embedding.path_to_query_embeddings,
        path_to_candidate_prompts=cfg.sentence_embedding.path_to_candidate_prompts,
        path_to_prompt_embeddings=cfg.sentence_embedding.path_to_prompt_embeddings,
        path_to_query_encoder=cfg.sentence_embedding.path_to_query_encoder,
        path_to_prompt_encoder=cfg.sentence_embedding.path_to_prompt_encoder,
        path_to_sentence_encoder=cfg.sentence_embedding.path_to_sentence_encoder,
        base_frozen_llm_id=cfg.sentence_embedding.base_frozen_llm_id,
        frozen_llm_tokenizer_id=cfg.sentence_embedding.frozen_llm_tokenizer_id,
        base_reward_model_id=cfg.sentence_embedding.base_reward_model_id,
        reward_model_tokenizer_id=cfg.sentence_embedding.reward_model_tokenizer_id,
        path_to_finetuned_params=cfg.sentence_embedding.path_to_finetuned_params,
        device=cfg.sentence_embedding.device,
        random_state=cfg.sentence_embedding.random_state,
    )


if __name__ == "__main__":
    start = time.time()
    main()
    finish = time.time()
    print("total runtime:", format_runtime(start, finish))
