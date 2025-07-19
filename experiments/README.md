# Experiments with OfflinePrompts

This directory includes the code to replicate the experiments done in the following paper.

Haruka Kiyohara, Daniel Yiming Cao, Yuta Saito, Thorsten Joachims.<br>
**Off-Policy Learning for Prompt-guided Text Personalization Using Logged Bandit Data**<br>
[link]()


(Table of contents)
- [Dependencies](#dependencies)
- [Synthetic setting](#synthetic-setting)
- [Full-LLM (movie description) setting](#full-llm-movie-description-setting)
- [Citation](#citation)

## Dependencies
This repository supports Python 3.9 or newer. The code corresponds to `offline-prompts==0.1.0`

- scikit-learn==1.0.2
- torch==2.1.0
- transformers==4.28.1
- sentence-transformers==2.2.2
- pandas==2.0.2
- seaborn==0.12.2
- matplotlib==3.7.1
- hydra-core==1.3.2
- wandb==0.19.8

Also, the full-LLM task requires access to [`mistralai/Mistral-7B-Instruct-v0.2`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).

## Synthetic setting
To conduct the synthetic experiment, run the following commands. Note that, make sure that the path is connected to this repository and `experiments/synthetic` directory.

(configurations)

```bash
setting={data_size/candidate_actions/reward_noise}
```

(i) Define the logging policy.

```bash
python logging_reward_regression.py setting={setting}
```

(ii) Train the regression model.

```bash
python3 data_collection_and_reward_regression.py setting={setting}
```

(iii) Train a single stage policy.

```bash
python off_policy_learning.py setting={setting} setting.gradient_type={regression-based/IS-based/hybrid}
```

(iv) Train a two stage policy.

```bash
python off_policy_learning.py setting={setting} setting.gradient_type=hybrid setting.clustering_type=fixed-action setting.is_two_stage_policy=True
```

(v) Train DSO (our proposal).

```bash
python off_policy_learning.py setting={setting} setting.gradient_type=IS-based setting.is_dso=True
```

(vi) Evaluate policies.

```bash
python evaluate_policy_performance.py setting={setting}
```

(ablations)

```bash
setting={data_size/candidate_actions/reward_noise}
kernel_type={gaussian/uniform}
is_monte_carlo_estimation={True/False}
```

```bash
python off_policy_learning.py setting={setting} setting.gradient_type=IS-based setting.is_dso=True setting.kernel_type={kernel_type} setting.is_monte_carlo_estimation={is_monte_carlo_estimation}
```

## Full-LLM (movie description) setting

To set up the full-LLM environment, download the pre-trained simulators and datasets [here](). For the detailed instructions, please also refer to [this file](). Then, make sure that the path is connected to this repository and `experiments/movie_description` directory. The code exection is confirmed with a single or multiple GPUs with 64GB memory.

(i) Train a base online policy.

```bash
python online_policy_learning.py \
  setting.path_to_user_embeddings={path_to_data_directory}/movielens_naive_cf_user_embeddings.pt \
  setting.path_to_queries={path_to_data_directory}/movielens_query.csv \
  setting.path_to_query_embeddings={path_to_data_directory}/movielens_query_embs.pt \
  setting.path_to_interaction_data={path_to_data_directory}/movielens_preprocessed_data.csv \
  setting.path_to_candidate_prompts={path_to_data_directory}/movielens_benchmark_prompts.csv \
  setting.path_to_prompt_embeddings={path_to_data_directory}/movielens_prompt_embs.pt \
  setting.path_to_finetuned_params=/{path_to_data_directory}/movielens_distilbert_reward_simulator.pt \
  setting.path_to_query_pca_matrix={path_to_data_directory}/movielens_query_pca_matrix.pt \
  setting.path_to_prompt_pca_matrix={path_to_data_directory}/movielens_prompt_pca_matrix.pt \
  setting.path_to_sentence_pca_matrix={path_to_data_directory}/movielens_sentence_pca_matrix.pt
```

(ii) Collect logged data and train regression and marginal density models.

```bash
python data_collection_and_reward_regression.py \
  setting.path_to_user_embeddings={path_to_data_directory}/movielens_naive_cf_user_embeddings.pt \
  setting.path_to_queries={path_to_data_directory}/movielens_query.csv \
  setting.path_to_query_embeddings={path_to_data_directory}/movielens_query_embs.pt \
  setting.path_to_interaction_data={path_to_data_directory}/movielens_preprocessed_data.csv \
  setting.path_to_candidate_prompts={path_to_data_directory}/movielens_benchmark_prompts.csv \
  setting.path_to_prompt_embeddings={path_to_data_directory}/movielens_prompt_embs.pt \
  setting.path_to_finetuned_params=/{path_to_data_directory}/movielens_distilbert_reward_simulator.pt \
  setting.path_to_query_pca_matrix={path_to_data_directory}/movielens_query_pca_matrix.pt \
  setting.path_to_prompt_pca_matrix={path_to_data_directory}/movielens_prompt_pca_matrix.pt \
  setting.path_to_sentence_pca_matrix={path_to_data_directory}/movielens_sentence_pca_matrix.pt
```

(iii) Train a single stage policy.

```bash
python off_policy_learning.py offline.gradient_type={regression-based/IS-based/hybrid} \
  setting.path_to_user_embeddings={path_to_data_directory}/movielens_naive_cf_user_embeddings.pt \
  setting.path_to_queries={path_to_data_directory}/movielens_query.csv \
  setting.path_to_query_embeddings={path_to_data_directory}/movielens_query_embs.pt \
  setting.path_to_interaction_data={path_to_data_directory}/movielens_preprocessed_data.csv \
  setting.path_to_candidate_prompts={path_to_data_directory}/movielens_benchmark_prompts.csv \
  setting.path_to_prompt_embeddings={path_to_data_directory}/movielens_prompt_embs.pt \
  setting.path_to_finetuned_params=/{path_to_data_directory}/movielens_distilbert_reward_simulator.pt \
  setting.path_to_query_pca_matrix={path_to_data_directory}/movielens_query_pca_matrix.pt \
  setting.path_to_prompt_pca_matrix={path_to_data_directory}/movielens_prompt_pca_matrix.pt \
  setting.path_to_sentence_pca_matrix={path_to_data_directory}/movielens_sentence_pca_matrix.pt
```

(iv) Train a two stage policy.

```bash
python off_policy_learning.py offline.is_two_stage_policy=True offline.gradient_type=hybrid \
  setting.path_to_user_embeddings={path_to_data_directory}/movielens_naive_cf_user_embeddings.pt \
  setting.path_to_queries={path_to_data_directory}/movielens_query.csv \
  setting.path_to_query_embeddings={path_to_data_directory}/movielens_query_embs.pt \
  setting.path_to_interaction_data={path_to_data_directory}/movielens_preprocessed_data.csv \
  setting.path_to_candidate_prompts={path_to_data_directory}/movielens_benchmark_prompts.csv \
  setting.path_to_prompt_embeddings={path_to_data_directory}/movielens_prompt_embs.pt \
  setting.path_to_finetuned_params=/{path_to_data_directory}/movielens_distilbert_reward_simulator.pt \
  setting.path_to_query_pca_matrix={path_to_data_directory}/movielens_query_pca_matrix.pt \
  setting.path_to_prompt_pca_matrix={path_to_data_directory}/movielens_prompt_pca_matrix.pt \
  setting.path_to_sentence_pca_matrix={path_to_data_directory}/movielens_sentence_pca_matrix.pt
```

(v) Train DSO (our proposal).

```bash
python off_policy_learning.py offline.is_dso=True offline.gradient_type=IS-based \
  setting.path_to_user_embeddings={path_to_data_directory}/movielens_naive_cf_user_embeddings.pt \
  setting.path_to_queries={path_to_data_directory}/movielens_query.csv \
  setting.path_to_query_embeddings={path_to_data_directory}/movielens_query_embs.pt \
  setting.path_to_interaction_data={path_to_data_directory}/movielens_preprocessed_data.csv \
  setting.path_to_candidate_prompts={path_to_data_directory}/movielens_benchmark_prompts.csv \
  setting.path_to_prompt_embeddings={path_to_data_directory}/movielens_prompt_embs.pt \
  setting.path_to_finetuned_params=/{path_to_data_directory}/movielens_distilbert_reward_simulator.pt \
  setting.path_to_query_pca_matrix={path_to_data_directory}/movielens_query_pca_matrix.pt \
  setting.path_to_prompt_pca_matrix={path_to_data_directory}/movielens_prompt_pca_matrix.pt \
  setting.path_to_sentence_pca_matrix={path_to_data_directory}/movielens_sentence_pca_matrix.pt
```

(vi) Evaluate policies.

```bash
python evaluate_policy_performance.py \
  setting.path_to_user_embeddings={path_to_data_directory}/movielens_naive_cf_user_embeddings.pt \
  setting.path_to_queries={path_to_data_directory}/movielens_query.csv \
  setting.path_to_query_embeddings={path_to_data_directory}/movielens_query_embs.pt \
  setting.path_to_interaction_data={path_to_data_directory}/movielens_preprocessed_data.csv \
  setting.path_to_candidate_prompts={path_to_data_directory}/movielens_benchmark_prompts.csv \
  setting.path_to_prompt_embeddings={path_to_data_directory}/movielens_prompt_embs.pt \
  setting.path_to_finetuned_params=/{path_to_data_directory}/movielens_distilbert_reward_simulator.pt \
  setting.path_to_query_pca_matrix={path_to_data_directory}/movielens_query_pca_matrix.pt \
  setting.path_to_prompt_pca_matrix={path_to_data_directory}/movielens_prompt_pca_matrix.pt \
  setting.path_to_sentence_pca_matrix={path_to_data_directory}/movielens_sentence_pca_matrix.pt
```

## Citation

If you use this simulator in your project or find this resource useful, please cite the following papers.

(benchmarking experiment)

Haruka Kiyohara, Daniel Yiming Cao, Yuta Saito, Thorsten Joachims.<br>
**An Off-Policy Learning Approach for Steering Sentence Generation towards Personalization**<br>
[link]()

```
@article{kiyohara2025off,
  title = {An Off-Policy Learning Approach for Steering Sentence Generation towards Personalization},
  author = {Kiyohara, Haruka and Cao, Daniel Yiming and Saito, Yuta and Joachims, Thorsten},
  journal = {xxx},
  pages = {xxx--xxx},
  year = {2025},
}
```

(package and documentation)

Haruka Kiyohara, Daniel Yiming Cao, Yuta Saito, Thorsten Joachims.<br>
**OfflinePrompts: Benchmark Suites for Prompt-guided Text Personalization using Logged Data**<br>
[link]()

```
@article{kiyohara2025offline,
  title = {OfflinePrompts: Benchmark Suites for Prompt-guided Text Personalization using Logged Data},
  author = {Kiyohara, Haruka and Cao, Daniel Yiming and Saito, Yuta and Joachims, Thorsten},
  journal = {xxx},
  pages = {xxx--xxx},
  year = {2025},
}
```
