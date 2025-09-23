:html_theme.sidebar_secondary.remove:

==========
Quickstart
==========

Full-LLM benchmark
~~~~~~~~~~
We first provide the example of running OPL on the movie-description generation task.

Setting up a semi-synthetic simulation
^^^^^^^^^^

To set up the default movie description benchmark, users can follow the following code:

.. code:: python

    from off_prompts.dataset import SemiSyntheticDataset
    dataset = SemiSyntheticDataset(
        path_to_user_embeddings="assets/movielens_naive_cf_user_embeddings.pt",
        path_to_queries="assets/movielens_query.csv",
        path_to_query_embeddings="assets/movielens_query_embs.pt",
        path_to_interaction_data="assets/movielens_preprocessed_data.csv",
        path_to_candidate_prompts="assets/movielens_benchmark_prompts.csv",
        path_to_prompt_embeddings="assets/movielens_prompt_embs.pt",
        path_to_finetuned_params= "assets/movielens_distilbert_reward_simulator.pt",
        random_state=12345,
    )

The default datasets, candidate prompts, and finetuned parameters, PCA matrices are stored in `off_prompts/dataset/assets/` in the OfflinePrompts repository. 
Please also refer to this page: `dataset/assets/README.md <https://github.com/aiueola/offline-prompts/tree/main/off_prompts/dataset/assets/README.md>`_, for the training process to obtain the default parameters.

To customize the benchmark setting, it is also possible to use configurable submodules: `ContextQueryLoader`, `CandidateActionsLoader`, `FrozenLLM`, and `RewardSimulator`. 
Specifically, users can first create customized instances of these submodules and then pass them to `SemiSyntheticDataset` as exemplified in the following codes:

.. code:: python

    from off_prompts.dataset import (
        ContextQueryLoader,
        CandidateActionsLoader,
        AutoFrozenLLM,
        TransformerRewardSimulator,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # load contexts and queries
    context_query_loader = DefaultContextQueryLoader(
        path_to_user_embeddings="assets/movielens_naive_cf_user_embeddings.pt",
        path_to_queries="assets/movielens_query.csv",
        path_to_query_embeddings="assets/movielens_query_embs.pt",
        path_to_interaction_data="assets/movielens_preprocessed_data.csv",
        device="cuda",
        random_state=12345,
    )

    # load candidate prompts
    candidate_actions_loader = CandidateActionsLoader(
        n_actions=1000,
        path_to_candidate_prompts="assets/movielens_benchmark_prompts.csv",
        path_to_prompt_embeddings="assets/movielens_prompt_embs.pt",
        random_state=12345,
    )

    # load frozen llm
    frozen_llm_tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        truncation=True,
        do_lower_case=True,
        use_fast=True,
    )
    frozen_llm_model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
    )
    frozen_llm_tokenizer_kwargs = {
        "add_special_tokens": True,
        "padding": True,
        "truncation": True,
        "max_length": 20,
        "return_tensors": "pt",
    }
    self.frozen_llm_prompt_formatter = MovielensPromptFormatter(
            tokenizer=frozen_llm_tokenizer,
            tokenizer_kwargs=frozen_llm_tokenizer_kwargs,
            device="cuda",
    )
    pattern = r"Broadly describe in a sentence the genres of the movie without including the name or any specifics of.*?\n\n"

    frozen_llm_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    frozen_llm_model.resize_token_embeddings(len(frozen_llm_tokenizer))
    frozen_llm_model.to("cuda")

    frozen_llm = AutoFrozenLLM(
        prompt_formatter=frozen_llm_prompt_formatter,
        model=frozen_llm_model,
        tokenizer=frozen_llm_tokenizer,
        tokenizer_kwargs=frozen_llm_tokenizer_kwargs,
        pattern=pattern,
        device="cuda",
        random_state=12345,
    )

    # load reward simulator
    reward_simulator_tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased",
        truncation=True,
        do_lower_case=True,
        use_fast=True,
    )
    reward_simulator_base_model = AutoModel.from_pretrained(
        "distilbert-base-uncased",
    )
    reward_simulator_tokenizer_kwargs = {
        "add_special_tokens": True,
        "padding": True,
        "truncation": True,
        "max_length": 20,
        "return_tensors": "pt",
    }

    reward_simulator_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    reward_simulator_base_model.resize_token_embeddings(
        len(reward_simulator_tokenizer)
    )
    reward_simulator_base_model.to("cuda")

    reward_simulator = TransformerRewardSimulator(
        n_users=context_query_loader.n_users,
        n_items=context_query_loader.n_queries,
        base_model=reward_simulator_base_model,
        tokenizer=reward_simulator_tokenizer,
        tokenizer_kwargs=reward_simulator_tokenizer_kwargs,
        device="cuda",
        random_state=12345,
    )
    reward_simulator.load_state_dict(
        torch.load("assets/movielens_distilbert_reward_simulator.pt")
    )

    # create a custom environment with customized modules
    dataset = SemiSyntheticDataset(
        context_query_loader=context_query_loader,
        candidate_actions_loader=candidate_actions_loader,
        frozen_llm=frozen_llm,
        reward_simulator=reward_simulator,
        frozen_llm_prompt_formatter=frozen_llm_prompt_formatter,
        reward_type="continuous",
        device="cuda",
        random_state=12345,
    )

Logging policy
^^^^^^^^^^

After setting up the simulator, the next step is to define a logging policy to collect logged feedback. 

For this step, we first load the dimension reduction model to obtain low dimensional embeddings of `query`, `prompt`, and `sentence`. 
These encoders are used across various models, e.g., to define the logging policy and to define a reward preditor, etc. 

.. code:: python

    from off_prompts.dataset import TransformerEncoder

    # define and fit encoders
    query_encoder = TransformerEncoder(
        prefix_prompt="Broadly describe in a sentence the genres of the movie without including the name or any specifics of the movie.\nTitle: ",
        postfix_prompt=" ",
        prefix_tokens_max_length=22,
        postfix_tokens_max_length=1,
        max_length=5,  # max length of query
        dim_emb=10,
        device="cuda",
        random_state=12345,
    )
    prompt_encoder = TransformerEncoder(
        prefix_prompt="Associate the word - ",
        postfix_prompt=" - in the context of movie genres",
        prefix_tokens_max_length=4,
        postfix_tokens_max_length=7,
        max_length=2,  # max length of prompt
        dim_emb=10,
        device="cuda",
        random_state=12345,
    )
    sentence_encoder = TransformerEncoder(
        prefix_prompt=" ",
        postfix_prompt=" ",
        prefix_tokens_max_length=1,
        postfix_tokens_max_length=1,
        max_length=20,  # max length of prompt
        dim_emb=10,
        device="cuda",
        random_state=12345,
    )

    # load fitted PCA matrix for dimension reduction
    query_encoder.load("assets/movielens_query_pca_matrix.pt")
    prompt_encoder.load("assets/movielens_prompt_pca_matrix.pt")
    sentence_encoder.load("assets/movielens_sentence_pca_matrix.pt")

Then, we train an online policy, which is used to define a softmax logging policy as follows. 


.. code:: python

    from off_prompts.opl import PromptRewardLearner
    from off_prompts.policy import PromptRewardPredictor
    from off_prompts.policy import PromptPolicy

    # train policy online using the learner class
    prompt_reward_predictor = PromptRewardPredictor(
        dim_context=dataset.dim_context,
        action_list=dataset.action_list,
        query_encoder=query_encoder,
        prompt_encoder=prompt_encoder,
        device="cuda",
        random_state=12345,
    )
    base_policy = PromptPolicy(
        n_actions=dataset.n_actions,
        dim_context=dataset.dim_context,
        query_encoder=query_encoder,
        device="cuda",
        random_state=12345,
    )
    policy_learner = PolicyLearner(
        model=policy,
        action_list=dataset.action_list,
        query_embeddings=dataset.query_embeddings,
        prompt_embeddings=dataset.prompt_embeddings,
        prompt_reward_predictor=prompt_reward_predictor,
        query_encoder=query_encoder,
        sentence_encoder=sentence_encoder,
        env=dataset,
        random_state=12345,
    )
    online_policy = policy_learner.online_policy_gradient(
        save_path="logs/online=policy.pt",
    )

Finally, we also collect the logged data using the above softmax logging policy.

.. code:: python

    from off_prompts.policy import SoftmaxPolicy

    # softmax logging policy on top of the online policy
    logging_policy = SoftmaxPolicy(
        action_list=dataset.action_list,
        base_model=base_policy,
        beta=0.2,
        device="cuda",
        random_state=12345,
    )

    # collect logged dataset
    logged_feedback = dataset.sample_dataset(
        policy=logging_policy, 
        n_samples=10000,
    )

The outputs, including `logged_feedback` and `meta_data`, contain the following keys.

* `logged_feedback``:
  * { `user_id`, `item_id`, `context`, `query`, `action`, `action_choice_probability`*, `sentence`, `expected_reward`*, `reward` }
* `meta_data`*: 
  * { `size`, `reward_type`, `reward_std`, `action_list`` }

Note that the keys with an asterisk (*) are optional outputs, and action is returned by index. 
`reward_type` indicates whether the reward is binary or continuous, and `action_list` contains the list of candidate prompts, corresponding to each action index. 

Regressions
^^^^^^^^^^

After obtaining the logged data, we regress the reward as follows.

.. code:: python

    from off_prompts.opl import  PromptRewardLearner
    from off_prompts.policy import  PromptRewardPredictor

    # train regression models
    prompt_reward_predictor = PromptRewardPredictor(
        dim_context=dataset.dim_context,
        action_list=dataset.action_list,
        query_encoder=query_encoder,
        prompt_encoder=prompt_encoder,
        device="cuda",
        random_state=12345,
    )
    prompt_reward_learner = PromptRewardLearner(
        model=prompt_reward_predictor,
        action_list=dataset.action_list,
        query_embeddings=dataset.query_embeddings,
        prompt_embeddings=dataset.prompt_embeddings,
        frozen_llm=dataset.frozen_llm,
        query_encoder=query_encoder,
        prompt_encoder=prompt_encoder,
        optimizer_kwargs={"lr": lr, "weight_decay": 0.0},
        env=dataset,
        random_state=12345,
    )
    prompt_reward_predictor = prompt_reward_learner.offline_training(
        logged_feedback=logged_feedback,
        save_path="logs/reward_predictor.pt",
    )

`prompt_reward_predictor` is used by regression-based, hybrid PG, and POTEC.

Similarly, we train a logging marginal density model as follows. 

.. code:: python

    from off_prompts.opl import MarginalDensityLearner
    from off_prompts.policy import KernelMarginalDensityEstimator
    from off_prompts.utils import gaussian_kernel

    # learning a marginal density model
    kernel_marginal_estimator = KernelMarginalDensityEstimator(
        action_list=dataset.action_list,
        dim_context=dataset.dim_context,
        frozen_llm=dataset.frozen_llm,
        query_encoder=query_encoder,
        sentence_encoder=sentence_encoder,
        kernel_function=gaussian=kernel,
        kernel_kwargs={"tau": 1.0},
        device="cuda",
        random_state=12345,
    )
    marginal_density_learner = MarginalDensityLearner(
        model=kernel_marginal_estimator,
        action_list=dataset.action_list,
        query_embeddings=dataset.query_embeddings,
        prompt_embeddings=dataset.prompt_embeddings,
        frozen_llm=dataset.frozen_llm,
        optimizer_kwargs={"lr": lr, "weight_decay": 0.0},
        random_state=12345,
    )
    kernel_marginal_estimator = marginal_density_learner.simulation_training(
        logged_feedback=logged_feedback,
        save_path="logs/logging_marginal_density.pt",
    )

`marginal_density_model` is used by DSO (our proposal).


Baseline policy gradients
^^^^^^^^^^

The following code shows the example codes to run naive PGs, including regression-based, IS-based, and hybrid ones (Please refer to :doc:`implementations` for the details about baseline methods). 

.. code:: python

    from off_prompts.opl import PolicyLearner
    from off_prompts.policy import PromptPolicy

    policy = PromptPolicy(
        n_actions=dataset.n_actions,
        dim_context=dataset.dim_context,
        query_encoder=query_encoder,
        device="cuda",
        random_state=12345,
    )
    policy_learner = PolicyLearner(
        model=policy,
        action_list=dataset.action_list,
        prompt_reward_predictor=prompt_reward_predictor,
        query_encoder=query_encoder,
        sentence_encoder=sentence_encoder,
        optimizer_kwargs={"lr": 5e-4, "weight_decay": 0.0},
        env=dataset,
        random_state=12345,
    )

    # regression-based
    policy = policy_learner.model_based_policy_gradient(
        logged_feedback=logged_feedback,
        save_path="logs/opl_regression.pt",
    )

    # IS-based
    policy = policy_learner.importance_sampling_based_policy_gradient(
        logged_feedback=logged_feedback,
        save_path="logs/opl_vanilla_is.pt",
    )

    # hybrid
    policy = policy_learner.hybrid_policy_gradient(
        logged_feedback=logged_feedback,
        save_path="logs/opl_hybrid.pt",
    )

The procedure consists of only 3 steps: (1) define a policy, (2) then setup a learner class (`PolicyLearner`), and (3) call one of the policy gradient methods. 
As seen in the above example code, all policy gradient methods can be called in similar formats. Researchers can also implement their own policy gradient methods in a similar way.

Direct Sentence Off-Policy Gradient (DSO)
^^^^^^^^^^
DSO is our proposal (please also refer to :doc:`dso` for details), which can also be run in a very similar way as the naive policy gradient. 

.. code:: python

    from off_prompts.opl import KernelPolicyLearner

    policy_learner = KernelPolicyLearner(
        model=policy,
        action_list=dataset.action_list,
        query_embeddings=dataset.query_embeddings,
        prompt_embeddings=dataset.prompt_embeddings,
        kernel_marginal_estimator=kernel_marginal_estimator,
        frozen_llm=dataset.frozen_llm,
        query_encoder=query_encoder,
        sentence_encoder=sentence_encoder,
        optimizer_kwargs={"lr": 5e-4, "weight_decay": 0.0},
        env="cuda",
        random_state=12345,
    )
    policy = policy_learner.importance_sampling_based_policy_gradient(
        logged_feedback=logged_feedback,
        save_path="logs/opl_kernel_is.pt",
    )


The key difference between the use of DSO and other methods is that DSO uses `KernelPolicyLearner` and `logging_marginal_density_model`. 
Only the IS-based policy gradient is implemented for DSO.

(Online) performance evaluation
^^^^^^^^^^
Finally, after learning a policy, we test its performance through online interaction. 
This can be done in a single line of code, as shown as follows.

.. code:: python

    policy_value = dataset.calc_expected_policy_value(
        policy=policy, 
        n_samples_to_approximate=10000,
    )

For the use of custom dataset, please also refer to :doc:`usage`.


Synthetic experiment
~~~~~~~~~~
Next, we also provide the example of running synthetic simulation with vectorial embeddings.


Setting up a synthetic simulation
^^^^^^^^^^

To set up the default environment, simply call the following codes.

.. code:: python

    from off_prompts_syn.dataset import SemiSyntheticDataset

    dataset = SemiSyntheticDataset(device="cpu", random_state=12345)


To customize the environment, please call the following instead.

.. code:: python

    from off_prompts_syn.dataset import (
        ContextQueryGenerator,
        CandidateActionsGenerator,
        TrigonometricAuxiliaryOutputGenerator,
        RewardSimulator,
    )

    context_query_generator = ContextQueryGenerator(
        dim_context=5,
        dim_query=5,
        device="cpu",
        random_state=12345,
    )
    candidate_action_generator = CandidateActionsGenerator(
        n_actions=1000,
        dim_action_embedding=5,
        device="cpu",
        random_state=12345,
    )
    auxiliary_output_generator = TrigonometricAuxiliaryOutputGenerator(
        dim_query=5,
        dim_action_embedding=5,
        dim_auxiliary_output=5,
        noise_level=0.1,
        device="cpu",
        random_state=12345,
    )
    reward_simulator = RewardSimulator(
        dim_context=5,
        dim_query=5,
        dim_auxiliary_output=5,
        device="cpu",
        random_state=12345,
    )

    dataset = SemiSyntheticDataset(
        n_actions=1000,
        dim_context=5,
        dim_query=5,
        dim_action_embedding=5,
        dim_auxiliary_output=5,
        context_query_generator=context_query_generator,
        candidate_action_generator=candidate_action_generator,
        auxiliary_output_generator=auxiliary_output_generator,
        reward_simulator=reward_simulator,
        reward_type="continuous",
        reward_std=1.0,
        device="cpu",
        random_state=12345,
    )

Logging policy
^^^^^^^^^^

We define the (value-based) logging policy and collect the logged data as follows.

.. code:: python

    from copy import deepcopy

    from off_prompts_syn.policy import UniformRandomPolicy, SoftmaxPolicy
    from off_prompts_syn.policy import ActionRewardLearner
    from off_prompts_syn.policy import NeuralActionRewardPredictor as ActionRewardPredictor

    # fit regression model for logging policy
    dataset_ = deepcopy(dataset)
    dataset_.random_state = random_state + 1

    uniform_policy = UniformRandomPolicy(
        action_list=dataset.action_list,
        device="cpu",
        random_state=12345,
    )
    logged_feedback_for_pretraining = dataset_.sample_dataset(
        policy=uniform_policy,
        n_samples=10000,
    )
    action_reward_predictor = ActionRewardPredictor(
        action_list=dataset.action_list,
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        device="cpu",
        random_state=12345,
    )
    action_reward_learner = ActionRewardLearner(
        model=action_reward_predictor,
        action_list=dataset.action_list,
        auxiliary_output_generator=dataset.auxiliary_output_generator,
        optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.0},
        env=dataset,
        random_state=12345,
    )
    logging_action_reward_predictor = action_reward_learner.offline_training(
        logged_feedback=logged_feedback_for_pretraining,
        random_state=12345,
    )

    # define logging policy
    logging_policy = SoftmaxPolicy(
        action_list=dataset.action_list,
        base_model=logging_action_reward_predictor,
        beta=1.0,
        device="cpu",
        random_state=12345,
    )

    # collect logged data
    logged_feedback = generate_logged_data(
        dataset=dataset,
        logging_policy=logging_policy,
        n_samples=10000,
    )


Regressions
^^^^^^^^^^

After obtaining the logged data, we regress the reward and logging marginal density as follows.

.. code:: python

    from off_prompts_syn.opl import MarginalDensityLearner
    from off_prompts_syn.policy import (
        NeuralMarginalDensityEstimator as KernelMarginalEstimator,
    )
    from off_prompts_syn.utils import gaussian_kernel

    action_reward_predictor = ActionRewardPredictor(
        action_list=dataset.action_list,
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        device="cpu",
        random_state=12345,
    )
    action_reward_learner = ActionRewardLearner(
        model=action_reward_predictor,
        action_list=dataset.action_list,
        auxiliary_output_generator=dataset.auxiliary_output_generator,
        optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.0},
        env=dataset,
        random_state=12345,
    )
    action_reward_predictor = action_reward_learner.offline_training(
        logged_feedback=logged_feedback,
        is_pessimistic=False,
    )

    kernel_marginal_estimator = KernelMarginalEstimator(
        action_list=dataset.action_list,
        auxiliary_output_generator=dataset.auxiliary_output_generator,
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        kernel_function=gaussian_kernel,
        kernel_kwargs={"tau": 1.0},
        emb_noise=1.0,
        device="cpu",
        random_state=12345,
    )
    marginal_density_learner = MarginalDensityLearner(
        model=kernel_marginal_estimator,
        action_list=dataset.action_list,
        auxiliary_output_generator=dataset.auxiliary_output_generator,
        optimizer_kwargs={"lr": 1e-4, "weight_decay": 0.0},
        random_state=12345,
    )
    kernel_marginal_estimator = marginal_density_learner.simulation_training(
        logged_feedback=logged_feedback,
    )


Baseline policy gradients
^^^^^^^^^^

For running the baseline policy gradient methods, please call the following.

.. code:: python

    policy = ActionPolicy(
        n_actions=dataset.n_actions,
        dim_context=dataset.dim_context,
        dim_query=dataset.dim_query,
        device="cpu",
        random_state=12345,
    )
    policy_learner = PolicyLearner(
        model=policy,
        action_list=dataset.action_list,
        action_reward_predictor=action_reward_predictor,
        optimizer_kwargs={"lr": 5e-4, "weight_decay": 0.0},
        env=dataset,
        random_state=12345,
    )

    # regression-based
    policy, learning_process = policy_learner.model_based_policy_gradient(
        logged_feedback=logged_feedback,
        return_training_logs=True,
        save_path="xxx.pt",
    )
    # IS-based
    policy, learning_process = policy_learner.importance_sampling_based_policy_gradient(
        logged_feedback=logged_feedback,
        return_training_logs=True,
        save_path="xxx.pt",
    )
    # DR-based
    policy, learning_process = policy_learner.hybrid_policy_gradient(
        logged_feedback=logged_feedback,
        return_training_logs=True,
        n_epochs=n_epochs,
        n_steps_per_epoch=n_steps_per_epoch,
        n_epochs_per_log=n_epochs_per_log,
        save_path="xxx.pt",
    )

.. raw:: html

    <div class="white-space-20px"></div>

.. grid::

    .. grid-item-card::
        :columns: 2
        :link: /Installation
        :link-type: doc
        :shadow: none
        :margin: 0
        :padding: 0

        <<< Prev
        **Installation!**

    .. grid-item::
        :columns: 8
        :margin: 0
        :padding: 0

    .. grid-item-card::
        :columns: 2
        :link: distinctive_features
        :link-type: doc
        :shadow: none
        :margin: 0
        :padding: 0

        Next >>>
        **Why OfflinePrompts**