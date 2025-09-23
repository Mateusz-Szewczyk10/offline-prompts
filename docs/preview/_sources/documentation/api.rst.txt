==========
OfflinePrompts Package Reference
==========

.. _api_full_llm:


Full-LLM benchmark
----------

.. _api_dataset:

Dataset module
^^^^^^
.. autosummary::
    :toctree: _autosummary/dataset
    :recursive:
    :nosignatures:

    off_prompts.dataset.base
    off_prompts.dataset.benchmark
    off_prompts.dataset.frozen_llm
    off_prompts.dataset.reward_simulator
    off_prompts.dataset.function
    off_prompts.dataset.encoder
    off_prompts.dataset.prompt_formatter
    off_prompts.dataset.assets.reward_finetuner
    off_prompts.dataset.assets.sentence_embedding_learner

.. _api_policy:

Policy module
^^^^^^
.. autosummary::
    :toctree: _autosummary/policy
    :recursive:
    :nosignatures:
    :template: module_head

    off_prompts.policy.base
    off_prompts.policy.model
    off_prompts.policy.policy

.. _api_opl:

OPE/L module
^^^^^^
.. autosummary::
    :toctree: _autosummary/opl
    :recursive:
    :nosignatures:
    :template: module_head

    off_prompts.opl.policy_learner
    off_prompts.opl.policy_evaluator
    off_prompts.opl.reward_learner
    off_prompts.opl.marginal_learner
    off_prompts.opl.behavior_cloning
    off_prompts.opl.dataset

.. _api_utils:

Others
^^^^^^
.. autosummary::
    :toctree: _autosummary/utils
    :recursive:
    :nosignatures:

    off_prompts.utils

.. _api_synthetic:

Synthetic simulation
----------

.. _api_dataset_syn:

Dataset module
^^^^^^
.. autosummary::
    :toctree: _autosummary/dataset
    :recursive:
    :nosignatures:

    off_prompts_syn.dataset.synthetic
    off_prompts_syn.dataset.function

.. _api_policy_syn:

Policy module
^^^^^^
.. autosummary::
    :toctree: _autosummary/policy
    :recursive:
    :nosignatures:
    :template: module_head

    off_prompts_syn.policy.base
    off_prompts_syn.policy.model
    off_prompts_syn.policy.policy

.. _api_opl_syn:

OPE/L module
^^^^^^
.. autosummary::
    :toctree: _autosummary/opl
    :recursive:
    :nosignatures:
    :template: module_head

    off_prompts_syn.opl.policy_learner
    off_prompts_syn.opl.policy_evaluator
    off_prompts_syn.opl.reward_learner
    off_prompts_syn.opl.marginal_learner
    off_prompts_syn.opl.behavior_cloning

.. _api_utils_syn:

Others
^^^^^^
.. autosummary::
    :toctree: _autosummary/utils
    :recursive:
    :nosignatures:

    off_prompts_syn.utils

.. raw:: html

    <div class="white-space-20px"></div>

.. grid::
    :margin: 0

    .. grid-item::
        :columns: 3
        :margin: 0
        :padding: 0

        .. grid::
            :margin: 0

            .. grid-item-card::
                :link: index
                :link-type: doc
                :shadow: none
                :margin: 0
                :padding: 0

                <<< Prev
                **Documentation (Back to Top)**

    .. grid-item::
        :columns: 6
        :margin: 0
        :padding: 0

    .. grid-item::
        :columns: 3
        :margin: 0
        :padding: 0