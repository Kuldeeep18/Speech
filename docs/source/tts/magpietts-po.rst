.. _magpie-tts-po:

=======================================
Magpie-TTS Preference Optimization
=======================================

Preference optimization is a powerful technique for improving the quality of Magpie-TTS outputs by learning from ranked examples. Rather than relying solely on supervised learning with ground-truth audio, preference optimization teaches the model to distinguish between good and bad generations, allowing it to internalize quality metrics like intelligibility and speaker similarity directly into its generation process.

Magpie-TTS supports two complementary approaches to preference optimization: offline alignment using Direct Preference Optimization (DPO) and online optimization using Group Relative Policy Optimization (GRPO). While DPO requires pre-generating preference data before training, GRPO generates candidates on the fly and is generally recommended for its simplicity and effectiveness.


Offline Preference Alignment (DPO)
##################################

Direct Preference Optimization works by fine-tuning the model on pairs of chosen and rejected outputs. The training objective encourages the model to increase the likelihood of generating outputs similar to the chosen examples while decreasing the likelihood of rejected ones. This approach is particularly useful when you have access to human quality judgments or want fine-grained control over the preference data.

The DPO pipeline consists of four distinct steps, each building on the output of the previous one.


Step 1: Create Text-Context Pairs
---------------------------------

The first step is to assemble a collection of text-context pairs that will be used for preference data generation. A well-designed dataset should include a mix of challenging texts (such as tongue twisters, technical terms, and complex sentence structures) alongside regular transcripts. These texts are paired with various speaker contexts—either audio samples for voice cloning or text descriptions for style conditioning.

The diversity of this dataset is crucial for robust preference optimization. Including challenging examples helps the model learn to handle edge cases, while regular transcripts ensure it maintains quality on typical inputs. You can also include examples with text contexts to improve style conditioning capabilities.

.. code-block:: bash

    python scripts/magpietts/dpo/create_text_contextpairs.py \
        --challenging_texts /path/to/challenging_texts.txt \
        --regular_texts_for_audiocontext /path/to/regular_texts_for_audiocontext.txt \
        --regular_texts_for_textcontext /path/to/regular_texts_for_textcontext.txt \
        --audio_contexts /path/to/audio_context_list.json \
        --text_contexts /path/to/text_context_list.txt \
        --output_manifest /path/to/text_context_pairs.json \
        --nsamples_perpair 6

The ``nsamples_perpair`` parameter specifies how many audio samples will be generated for each text-context pair in the next step. Setting this to 6 provides enough variety to create meaningful preference pairs while keeping computation manageable. The output manifest serves as input for the generation step.


Step 2: Generate Audio Samples
------------------------------

With the text-context pairs prepared, the next step is to generate multiple audio samples for each pair using a base Magpie-TTS checkpoint. The generation process also computes quality metrics—Character Error Rate (CER) and Speaker Similarity (SSIM)—for each output, which will be used to create preference pairs.

This step can be parallelized across multiple GPUs and nodes to speed up generation. Each generated audio file is accompanied by a JSON file containing the computed metrics.

.. code-block:: bash

    python examples/tts/magpietts.py \
        --config-name=magpietts_po_inference \
        mode=test \
        batch_size=64 \
        +init_from_ptl_ckpt=/path/to/magpie_checkpoint \
        exp_manager.exp_dir=/path/to/po_experiment \
        +test_ds_meta.textcontextpairs.manifest_path=/path/to/text_context_pairs.json \
        +test_ds_meta.textcontextpairs.audio_dir="/" \
        +test_ds_meta.textcontextpairs.feature_dir="/" \
        model.codecmodel_path=/path/to/codec_model.nemo \
        model.prior_scaling_factor=null \
        model.load_cached_codes_if_available=false

.. note::

    The manifest contains absolute audio paths, so ``audio_dir`` is set to ``"/"``. Adjust the model configuration parameters to match your base checkpoint architecture.


Step 3: Create Preference Pairs
-------------------------------

Once audio samples are generated, you need to create chosen-rejected pairs based on the computed metrics. The script analyzes the CER and SSIM scores for each group of samples and selects the best and worst performers to form preference pairs.

.. code-block:: bash

    python scripts/magpietts/dpo/create_preference_pairs.py \
        --input_manifest /path/to/text_context_pairs.json \
        --generated_audio_dir /path/to/po_experiment/MagpieTTS-PO-Infer/version_0/audio \
        --group_size 6 \
        --cer_threshold 0.01 \
        --val_size 256

The ``cer_threshold`` parameter filters out pairs where even the chosen example has poor intelligibility (CER > 0.01). This ensures the model learns from genuinely good examples rather than just "less bad" ones. The script outputs train and validation manifests in the ``manifests/`` subdirectory.


Step 4: DPO Fine-tuning
-----------------------

The final step is fine-tuning the base model on the preference pairs using the DPO loss. This teaches the model to prefer generating outputs similar to the chosen examples over the rejected ones.

.. code-block:: bash

    python examples/tts/magpietts.py \
        batch_size=4 \
        +init_from_ptl_ckpt=/path/to/magpie_checkpoint \
        +mode="dpo_train" \
        max_epochs=10 \
        exp_manager.exp_dir=/path/to/dpo_experiment \
        exp_manager.checkpoint_callback_params.always_save_nemo=false \
        model.train_ds.datasets._target_="nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDatasetDPO" \
        model.validation_ds.datasets._target_="nemo.collections.tts.data.text_to_speech_dataset.MagpieTTSDatasetDPO" \
        +train_ds_meta.dpopreftrain.manifest_path="/path/to/manifests/" \
        +train_ds_meta.dpopreftrain.audio_dir="/" \
        +train_ds_meta.dpopreftrain.feature_dir="/" \
        +val_ds_meta.dpoprefval.manifest_path="/path/to/manifests/dpo_val_manifest.json" \
        +val_ds_meta.dpoprefval.audio_dir="/" \
        +val_ds_meta.dpoprefval.feature_dir="/" \
        +model.dpo_beta=0.01 \
        +model.dpo_sft_loss_weight=0.0 \
        model.codecmodel_path=/path/to/codec_model.nemo \
        model.alignment_loss_scale=0.001 \
        model.prior_scaling_factor=null \
        trainer.val_check_interval=200 \
        trainer.log_every_n_steps=10 \
        model.optim.lr=2e-7 \
        ~model.optim.sched

Key parameters for DPO training include ``dpo_beta``, which controls the strength of the preference signal, and a low learning rate (2e-7) to ensure stable fine-tuning.


Online Preference Optimization (GRPO)
#####################################

Group Relative Policy Optimization offers a more streamlined approach that eliminates the need for pre-generating preference data. Instead, GRPO generates multiple candidate outputs for each training example on the fly, computes reward signals based on quality metrics, and optimizes the model to maximize these rewards through policy gradient methods.

GRPO is generally recommended over DPO for several reasons. It continuously adapts to the model's current capabilities rather than relying on static preference data. It requires less setup and storage since there's no need to pre-generate and store audio samples. Additionally, it can optimize for multiple reward signals simultaneously, including CER, SSIM, and PESQ.


Setting Up GRPO Training
------------------------

The GRPO sections below describe the MagpieTTS model trained with ``examples/tts/magpietts.py``; the
EasyMagpie-TTS model has its own online PO implementation and configuration, documented in
:ref:`easy-magpie-tts-online-po`.

The GRPO training process starts with preparing text-context pairs, similar to DPO but without the need for multiple samples per pair:

.. code-block:: bash

    python scripts/magpietts/dpo/create_text_contextpairs.py \
        --challenging_texts /path/to/challenging_texts.txt \
        --regular_texts_for_audiocontext /path/to/regular_texts_for_audiocontext.txt \
        --regular_texts_for_textcontext /path/to/regular_texts_for_textcontext.txt \
        --audio_contexts /path/to/audio_context_list.json \
        --text_contexts /path/to/text_context_list.txt \
        --output_manifest /path/to/text_context_pairs_grpo.json \
        --nsamples_perpair 1

Note that ``nsamples_perpair`` is set to 1 since GRPO generates candidates during training.


GRPO Training Configuration
---------------------------

GRPO training requires careful configuration of several hyperparameters. The following table summarizes the key parameters:

.. list-table:: GRPO Hyperparameters
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``num_generations_per_item``
     - 12
     - Number of candidate outputs generated per training example
   * - ``reference_free``
     - true
     - If true, skips KL divergence term and optimizes rewards directly
   * - ``grpo_beta``
     - 0.0
     - Coefficient for KL loss (only used when reference_free=false)
   * - ``cer_reward_weight``
     - 0.33
     - Weight of Character Error Rate in the reward function
   * - ``ssim_reward_weight``
     - 0.33
     - Weight of Speaker Similarity in the reward function
   * - ``pesq_reward_weight``
     - 0.33
     - Weight of PESQ score in the reward function
   * - ``use_pesq``
     - true
     - Whether to include PESQ in the reward computation
   * - ``reward_asr_model``
     - (none)
     - ASR model for CER computation; set to ``whisper`` for multilingual
   * - ``inference_temperature``
     - 0.8
     - Sampling temperature for candidate generation
   * - ``inference_topk``
     - 2016
     - Top-k sampling parameter (2016 effectively disables it)
   * - ``loss_type``
     - "grpo"
     - Loss function variant; can be "grpo" or "dr_grpo"
   * - ``scale_rewards``
     - true
     - Whether to normalize advantages by standard deviation


GRPO Training Command
---------------------

The following command demonstrates a complete GRPO training setup for multilingual models:

.. code-block:: bash

    python examples/tts/magpietts.py \
        --config-name=magpietts \
        batch_size=2 \
        +init_from_ptl_ckpt=/path/to/magpie_checkpoint \
        model.codecmodel_path=/path/to/codec_model.nemo \
        +mode="onlinepo_train" \
        max_epochs=20 \
        exp_manager.exp_dir=/path/to/grpo_experiment \
        +exp_manager.version=0 \
        exp_manager.checkpoint_callback_params.always_save_nemo=false \
        +train_ds_meta.dpopreftrain.manifest_path=/path/to/train_manifest.json \
        +train_ds_meta.dpopreftrain.audio_dir="/" \
        +train_ds_meta.dpopreftrain.feature_dir="/" \
        +val_ds_meta.dpoprefval.manifest_path=/path/to/val_manifest.json \
        +val_ds_meta.dpoprefval.audio_dir="/" \
        +val_ds_meta.dpoprefval.feature_dir="/" \
        +model.grpo_beta=0.0 \
        +model.num_generations_per_item=12 \
        +model.reference_free=true \
        +model.inference_cfg_prob=0.5 \
        +model.inference_cfg_scale=2.5 \
        +model.cer_reward_weight=0.45 \
        +model.ssim_reward_weight=0.45 \
        +model.pesq_reward_weight=0.1 \
        +model.use_pesq=true \
        +model.reward_asr_model="whisper" \
        model.cfg_unconditional_prob=0.0 \
        +model.inference_topk=2016 \
        +model.inference_temperature=0.7 \
        +model.use_kv_cache_during_online_po=true \
        +model.loss_type="grpo" \
        +model.max_decoder_steps=430 \
        model.decoder.p_dropout=0.0 \
        model.encoder.p_dropout=0.0 \
        model.alignment_loss_scale=0.0 \
        model.prior_scaling_factor=null \
        ~trainer.check_val_every_n_epoch \
        +trainer.val_check_interval=50 \
        trainer.log_every_n_steps=10 \
        model.optim.lr=1e-7 \
        ~model.optim.sched \
        exp_manager.checkpoint_callback_params.monitor="val_cer_gt" \
        exp_manager.checkpoint_callback_params.mode="min" \
        trainer.precision=32 \
        +trainer.gradient_clip_val=2.5


Important GRPO Training Considerations
--------------------------------------

Several configuration choices are critical for stable GRPO training:

**Disable Dropout**: Set ``p_dropout=0.0`` for all modules (encoder, decoder). This is essential when not using reference-free mode, as dropout causes the KL divergence loss to become unstable.

**Disable Attention Priors and CTC Loss**: Set ``alignment_loss_scale=0.0`` and ``prior_scaling_factor=null``. These training signals can interfere with the preference optimization objective.

**Use Small Batch Size**: Since GRPO generates ``num_generations_per_item`` samples for each batch item, the effective batch size becomes ``batch_size * num_generations_per_item``. A batch size of 2 with 12 generations per item results in 24 forward passes per step.

**Frequent Validation**: GRPO steps take longer than standard training steps due to the generation overhead. Configure more frequent validation with ``val_check_interval=50`` to monitor progress.

**Low Learning Rate**: Use a learning rate around 1e-7 to ensure stable optimization. The preference signal is noisy, and aggressive updates can destabilize training.

**Model-Specific Overrides**: Ensure your GRPO configuration matches the base model architecture, including attention heads, number of layers, local transformer settings, and tokenizer configuration.


Advanced: Local Transformer Optimization
----------------------------------------

For models with a Local Transformer, GRPO can optimize both with and without the LT by setting ``use_local_transformer_prob`` between 0 and 1. This trains the model to produce high-quality outputs regardless of whether the Local Transformer is used during inference, providing flexibility in the speed-quality trade-off at deployment time.

.. code-block:: bash

    +model.use_local_transformer_prob=0.5  # 50% of generations use LT


.. _easy-magpie-tts-online-po:

EasyMagpie-TTS Online Preference Optimization
---------------------------------------------

EasyMagpie-TTS has its own online preference optimization model, ``EasyMagpieTTSModelOnlinePO``
(``nemo/collections/tts/models/easy_magpietts_preference_optimization.py``), trained through
``examples/tts/easy_magpietts.py``. It follows the same rollout-then-reward loop as the GRPO recipe above
(``loss_type`` ``grpo`` or ``dr_grpo``, ``scale_rewards``, ``reference_free`` and ``grpo_beta``), but its
configuration differs: the reward ASR is a language-routed set of backends, classifier-free-guidance (CFG)
rollouts are scheduled by step rather than sampled, validation transcribes through the reward ASR, and the
multi-turn Lhotse dataset can inject challenging texts on a schedule. The subsections below document these
keys as the code reads them.


Launching
~~~~~~~~~

.. code-block:: bash

    python examples/tts/easy_magpietts.py \
        --config-name easy_magpietts \
        +mode=onlinepo_train \
        +init_from_ptl_ckpt=/path/to/easy_magpie_checkpoint.ckpt \
        <model, data and trainer overrides; see the complete command below>

In ``onlinepo_train`` mode the launcher copies ``init_from_ptl_ckpt`` into ``model.reference_model_ckpt_path``,
from which the frozen reference model is loaded unless ``model.reference_free=true``. The model constructor
validates the online-PO keys before any model is loaded, so a non-zero value of the removed
``inference_cfg_prob`` key or an unsupported ``rollout_cfg_mode`` / ``validation_asr_backend`` value fails
immediately with ``ValueError``.
When the run does not resume from a checkpoint, the launcher calls ``trainer.validate(model)`` before
``trainer.fit(model)``, so the first logged validation metrics describe the starting checkpoint.

The PO model uses manual optimization, so Lightning rejects a non-zero ``trainer.gradient_clip_val`` (the
recipe passes ``trainer.gradient_clip_val=0.0``); gradient clipping is configured with ``model.max_grad_norm``
(default ``0.0``, disabled) instead. Each training item is rolled out
``model.n_generations_per_item`` times (default 6); note that this key is named differently from the
``num_generations_per_item`` key of the MagpieTTS recipe above.

.. list-table:: EasyMagpie-TTS online-PO keys (``model.*``)
   :header-rows: 1
   :widths: 34 14 52

   * - Key
     - Default
     - Description
   * - ``n_generations_per_item``
     - 6
     - Rollouts per training item; a rollout batch has ``batch_size * n_generations_per_item`` items.
   * - ``reference_free``
     - false
     - Skip the frozen reference model and the KL term.
   * - ``grpo_beta``
     - 0.0
     - Weight of the KL term in the optimized loss (requires ``reference_free=false``).
   * - ``loss_type``
     - ``grpo``
     - ``grpo`` or ``dr_grpo`` token normalization of the policy loss.
   * - ``scale_rewards``
     - true
     - Divide group advantages by the group's reward standard deviation.
   * - ``cer_reward_weight`` / ``ssim_reward_weight`` / ``utmos_reward_weight``
     - 0.5 / 0.5 / 0.0
     - Weights of the shaped CER, speaker-similarity and UTMOSv2 rewards.
   * - ``use_utmos``
     - false
     - Score rollouts with UTMOSv2 (on ``utmos_device``, ``cpu`` by default).
   * - ``best_cer_threshold`` / ``worst_cer_threshold``
     - 1.0 / 1.0
     - A group is skipped when its best CER exceeds the first or its worst CER exceeds the second value.
   * - ``inference_temperature`` / ``inference_topk``
     - 0.7 / 80
     - Audio sampling parameters of the rollouts (``inference_topk<=0`` samples from the full codebook).
   * - ``max_decoder_steps``
     - 220
     - Maximum rollout length in decoder steps.
   * - ``gt_phoneme_input_prob``
     - 0.0
     - Probability that a training step rolls out with ground-truth phonemes (the auxiliary phoneme loss then
       applies); incompatible with challenging-text replacement.
   * - ``use_local_transformer_prob``
     - 0.0
     - Probability that a training step's rollouts use the local transformer.
   * - ``batch_size_for_chunked_tf`` / ``po_groups_per_subbatch``
     - 4 / 1
     - Rollouts per sub-batch of the teacher-forced forward/backward pass. ``po_groups_per_subbatch`` is used only
       when ``batch_size_for_chunked_tf`` is ``null``: the sub-batch then holds that many whole groups
       (``po_groups_per_subbatch * n_generations_per_item`` rollouts).
   * - ``aux_phoneme_loss_weight`` / ``phoneme_po_loss_weight`` / ``entropy_coeff``
     - 1.0 / 0.0 / 0.0
     - Weights of the auxiliary phoneme loss, the phoneme-stream policy loss and the entropy bonus.
   * - ``max_grad_norm``
     - 0.0
     - Gradient-norm clipping threshold (``0`` disables clipping; non-finite gradients always raise).
   * - ``speaker_verification_model_name``
     - ``titanet_large``
     - Speaker-verification model used for the speaker-similarity reward.


Reward ASR Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

Rewards (and, by default, validation CER/WER) are computed from transcripts produced by a ``RewardASRRouter``
(``nemo/collections/tts/parts/utils/reward_asr.py``) configured under ``model.reward_asr``. The router sends
each generated utterance to the backend registered for its language and returns the transcripts in input
order.

.. list-table:: ``model.reward_asr`` keys
   :header-rows: 1
   :widths: 24 16 60

   * - Key
     - Default
     - Description
   * - ``default_backend``
     - ``nemo``
     - Name of the ``backends`` entry used for every language without a route.
   * - ``language_routes``
     - ``{}``
     - Mapping from language code (``batch['languages']``) to a backend name.
   * - ``backends``
     - (required)
     - Mapping from backend name to its configuration. Every backend named by ``default_backend`` or
       ``language_routes`` must be present; a missing entry or an unknown ``type`` raises ``ValueError``.
       ``type`` selects the backend class and defaults to the entry name, but ``nemo_process`` entries must set
       ``type: nemo_process`` (or ``worker_backend: nemo``) explicitly: the process backend itself falls back to
       ``qwen`` when ``type`` is missing.
   * - ``log_samples``
     - 0
     - Number of transcripts per language and rollout batch logged on the global-zero rank as
       ``[reward_asr_transcript]`` lines (ground-truth text, raw and normalized transcript).

Each ``backends`` entry selects one of four backend types:

.. list-table:: Reward ASR backend types
   :header-rows: 1
   :widths: 18 82

   * - ``type``
     - Keys read (defaults in parentheses)
   * - ``nemo``
     - In-process NeMo ASR (``NemoRewardASRBackend``). ``model_name`` (``nvidia/parakeet-ctc-0.6b``; a
       pretrained name or a ``.nemo`` path), ``disable_cuda_graphs`` (``false``),
       ``reset_cuda_graphs_before_transcribe`` (``true``) and, for prompted models
       (``EncDecHybridRNNTCTCBPEModelWithPrompt`` / ``EncDecRNNTBPEModelWithPrompt``), ``language_map`` (see
       below; every locale must exist in the model's prompt dictionary) and ``attention_context`` (encoder
       context ``[left, right]``; must be one of the model's available contexts). Prompted models are
       transcribed per language with the mapped locale; other models transcribe the whole batch at once.
   * - ``nemo_process``
     - NeMo ASR in a separate worker process (``ProcessRewardASRBackend`` running
       ``scripts/tts/reward_asr_worker.py --backend nemo``). ``model_name``
       (``nvidia/nemotron-3.5-asr-streaming-0.6b``), ``python_executable`` (the training process's interpreter,
       ``sys.executable``), ``worker_script`` (the repository's ``scripts/tts/reward_asr_worker.py``),
       ``batch_size`` (4; request chunk size for prompted models), ``timeout_seconds`` (300),
       ``attention_context`` and ``language_map``.
   * - ``qwen``
     - Qwen3-ASR in a separate worker process (``ProcessRewardASRBackend`` running the worker with
       ``--backend qwen``). ``model_name`` (``Qwen/Qwen3-ASR-0.6B``), ``python_executable``
       (``/opt/qwen_asr/bin/python``), ``worker_script`` (as above), ``batch_size`` (4, the Qwen
       ``max_inference_batch_size``), ``max_new_tokens`` (256) and ``timeout_seconds`` (300).
   * - ``whisper``
     - In-process Hugging Face Whisper (``WhisperRewardASRBackend``). ``model_name``
       (``openai/whisper-large-v3``). The only backend whose transcripts are normalized by default (see
       :ref:`easy-magpie-tts-online-po-normalization`).

``language_map`` defaults to ``DEFAULT_NEMOTRON_LANGUAGE_MAP``: ``ar`` to ``ar-AR``, ``de`` to ``de-DE``, ``en``
to ``en-US``, ``es`` to ``es-ES``, ``fr`` to ``fr-FR``, ``hi`` to ``hi-IN``, ``it`` to ``it-IT``, ``ja`` to
``ja-JP``, ``ko`` to ``ko-KR``, ``pt`` to ``pt-BR``, ``vi`` to ``vi-VN`` and ``zh`` to ``zh-CN``. With a prompted
model, a language without an entry makes the in-process ``nemo`` backend raise ``ValueError``; in the
``nemo_process`` worker the same lookup fails inside the worker, whose error reply surfaces as ``RuntimeError``
after one worker restart. The two process backends also accept
``worker_backend`` (``nemo`` or ``qwen``), which defaults to ``nemo`` for ``type: nemo_process`` and to the
``type`` otherwise. ``ProcessRewardASRBackend`` only sees the entry's own keys and treats a missing ``type`` as
``qwen``, so a hand-written ``nemo_process`` entry without ``type: nemo_process`` or ``worker_backend: nemo``
starts the Qwen worker (the legacy-key translation below always sets ``type``).

A single in-process NeMo backend:

.. code-block:: yaml

    model:
      reward_asr:
        default_backend: nemo
        log_samples: 2
        backends:
          nemo:
            type: nemo
            model_name: nvidia/parakeet-ctc-0.6b

Qwen3-ASR for most languages, with Whisper for Hindi:

.. code-block:: yaml

    model:
      reward_asr:
        default_backend: qwen
        language_routes:
          hi: whisper
        backends:
          qwen:
            type: qwen
            model_name: Qwen/Qwen3-ASR-0.6B
            python_executable: /opt/qwen_asr/bin/python
            batch_size: 4
            max_new_tokens: 256
          whisper:
            type: whisper
            model_name: openai/whisper-large-v3

**Worker processes.** The ``nemo_process`` and ``qwen`` backends keep the ASR model out of the training
process's CUDA context. The worker starts lazily on the first transcription request and is pinned to the
training process's GPU: the backend resolves the parent's logical device index through the parent's own
``CUDA_VISIBLE_DEVICES`` mask and exports that single entry to the worker, which always loads its model on
``cuda:0``. Parent and worker exchange one JSON object per line over the worker's stdin/stdout: a ``ready``
message after the model is loaded, an ``ok`` or ``error`` reply to every ``transcribe`` request, and ``stopped``
after ``shutdown`` (the module docstring of ``scripts/tts/reward_asr_worker.py`` lists the message fields).
The worker's stderr, including all model logging, is appended to
``<tempdir>/<worker_backend>_asr_worker_<pid>_<device_index>.log``. A request that fails, times out
(``timeout_seconds``) or hits a broken pipe is retried once on a freshly started worker; a second failure
propagates. The model's ``teardown`` shuts every worker down.

The Qwen worker imports the ``qwen_asr`` package, which is not a NeMo dependency, so ``python_executable``
must point at an interpreter that has it installed. The default ``/opt/qwen_asr/bin/python`` is the externally
managed Qwen runtime of the Qwen-enabled training container; when that interpreter does not exist the backend
raises ``RuntimeError`` as soon as it is constructed. The ``nemo_process`` worker runs with the training
process's own interpreter by default.


Legacy Flat Keys
~~~~~~~~~~~~~~~~

When ``model.reward_asr`` is absent, the flat keys of earlier recipes are translated into an equivalent
``reward_asr`` configuration (``_translate_legacy_reward_asr_cfg``). New recipes should configure
``model.reward_asr`` directly.

.. list-table:: Legacy reward ASR keys
   :header-rows: 1
   :widths: 30 16 54

   * - Key
     - Default
     - Translation
   * - ``reward_asr_model``
     - ``nemo``
     - ``nemo``: in-process backend ``type: nemo``. ``nemotron``: worker backend ``type: nemo_process``.
       ``whisper``: ``type: whisper`` with ``openai/whisper-large-v3``. ``qwen_whisper``: a ``qwen`` default
       backend plus a ``whisper`` backend routed for ``qwen_asr_whisper_languages``. Any other value raises
       ``ValueError``.
   * - ``reward_asr_model_name``
     - (unset)
     - ``nemo`` / ``nemotron`` only. Forwarded as ``model_name`` only when set, so each backend otherwise
       applies its own default: ``nvidia/parakeet-ctc-0.6b`` in-process and
       ``nvidia/nemotron-3.5-asr-streaming-0.6b`` in the worker.
   * - ``reward_asr_batch_size``
     - 16
     - ``nemo`` / ``nemotron`` only, forwarded as ``batch_size``. The in-process ``nemo`` backend transcribes
       each rollout batch at once and does not read it; the ``nemotron`` worker uses it to split requests for
       prompted models (such as its default model) and transcribes non-prompted models at once.
   * - ``reward_asr_att_context_size``
     - (unset)
     - ``nemo`` / ``nemotron`` only, forwarded as ``attention_context``.
   * - ``reward_asr_log_samples``
     - 0
     - Forwarded as ``log_samples``.
   * - ``qwen_asr_whisper_languages``
     - ``[hi]``
     - ``qwen_whisper`` only: languages routed to the Whisper backend.
   * - ``qwen_asr_model_name`` / ``qwen_asr_batch_size`` / ``qwen_asr_max_new_tokens``
     - ``Qwen/Qwen3-ASR-0.6B`` / 4 / 256
     - ``qwen_whisper`` only: ``model_name``, ``batch_size`` and ``max_new_tokens`` of the Qwen worker.
   * - ``qwen_asr_python``
     - (unset)
     - ``qwen_whisper`` only: forwarded as ``python_executable`` when set.


.. _easy-magpie-tts-online-po-normalization:

Transcript Normalization
~~~~~~~~~~~~~~~~~~~~~~~~

Before CER/WER are computed, a reward transcript may be passed through the per-language TTS text normalizer so
that written-form ASR output (digits, punctuation, casing) matches the spoken-form reference text;
``process_text_for_cer`` is applied to every transcript afterwards. ``model.normalize_reward_transcript`` is a
tri-state switch:

* unset (or ``null``): decided per backend. Only Whisper declares ``normalizes_transcripts_by_default``, so
  Whisper transcripts are normalized while NeMo, Nemotron-worker and Qwen transcripts are used raw. The legacy
  ``model.normalize_whisper_transcript`` key (default ``true``) is still honoured here: ``false`` disables
  normalization for every backend.
* ``true``: normalize the transcripts of every backend.
* ``false``: normalize nothing.

.. note::

    #16301, which introduced the reward ASR router, normalized the transcripts of every backend by default.
    Recipes from #16301 that use a NeMo backend (including the default ``reward_asr_model=nemo``, the in-process
    parakeet model), Nemotron or Qwen and that relied on this must now set
    ``model.normalize_reward_transcript=true``. Whisper recipes, and recipes from before #16301 (which normalized
    Whisper output only), are unaffected.


Validation
~~~~~~~~~~

The PO model forces ``run_val_inference`` on, so every validation step generates audio with the base-class
validation inference (one generation per item with fixed sampling settings; CFG with scale 2.5 unless
``model.inference_use_cfg_in_val=false``) and reports ``val/cer``, ``val/wer``, ``val/ssim`` and, with
``use_utmos=true``, ``val/utmos``. ``model.validation_asr_backend`` selects the ASR that transcribes the
validation audio:

* unset or ``reward`` (the default): the reward ASR router described above, so validation CER/WER are computed
  with the same language-routed backends as the training rewards and the PO model does not load the
  base-class validation ASR models (unless ``model.run_val_inference=true`` is set in the config, which makes
  the base class load them).
* ``default``: the base-class validation ASR (Whisper when ``use_multilingual_asr=true``, otherwise the NeMo
  evaluation model). This requires ``model.run_val_inference=true`` in the config, because the base class only
  loads those models when that key is true; the combination is checked when the model is constructed.
* any other value raises ``ValueError``.

Per-language metrics ``val/cer_lang_<lang>`` and ``val/wer_lang_<lang>`` are always logged for the PO model,
whatever ``use_multilingual_asr`` is set to (the base model logs them only when that flag is true).


Rollout CFG
~~~~~~~~~~~

Whether a training rollout samples with classifier-free guidance is controlled by ``model.rollout_cfg_mode``:

* ``off`` (default): never use CFG.
* ``alternate``: use CFG on every odd ``global_step`` and plain sampling on even steps, so both decoding modes
  are optimized. ``train_cfg_fraction`` is therefore 1 and 0 on alternating steps, and
  ``train_mean_reward_cfg`` / ``train_mean_reward_no_cfg`` are logged on the steps that did / did not use CFG.

``model.inference_cfg_scale`` (default 2.5) is the guidance scale used when CFG is on. The old
``model.inference_cfg_prob`` key is no longer read: a non-zero value raises ``ValueError`` when the model is
constructed (``0`` or an absent key is accepted). Validation inference is not affected by ``rollout_cfg_mode``.


Codec Decoding Memory
~~~~~~~~~~~~~~~~~~~~~

Every rollout decodes ``batch_size * n_generations_per_item`` code sequences to waveforms, and the codec
decoder's temporary convolution activations grow with that batch. ``model.codec_decode_batch_size`` (default
``0``) makes ``streaming_finalize`` decode the generated codes in sub-batches of that many items; ``0``, or a
value not smaller than the batch, decodes everything at once. The output is identical either way; only the
peak memory changes.


Challenging-Text Schedule and Context Shuffling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``model.use_multiturn_dataset=true`` the Lhotse multi-turn dataset (``MagpieTTSLhotseMultiturnDataset``)
can replace the transcript of one English TTS output-role turn per training batch with a random line of a
challenging-text file, so that rollouts are also scored on hard texts. Up to 32 random candidates are tried
and the first one whose tokens fit the turn's available frames is used. The keys live under
``model.train_ds.dataset`` (``examples/tts/conf/magpietts/easy_magpietts_lhotse_multiturn.yaml`` lists them
commented out):

.. list-table:: Challenging-text keys (``model.train_ds.dataset``)
   :header-rows: 1
   :widths: 34 10 56

   * - Key
     - Default
     - Description
   * - ``challenging_texts_path``
     - ``null``
     - UTF-8 text file with one challenging text per line (blank lines are ignored). Required, and must contain
       at least one text, when ``challenging_text_end_prob > 0``.
   * - ``challenging_text_start_prob``
     - 0.0
     - Replacement probability at ``challenging_text_start_step``.
   * - ``challenging_text_end_prob``
     - 0.0
     - Replacement probability reached at ``challenging_text_end_step`` and kept afterwards; ``0`` disables
       the feature.
   * - ``challenging_text_start_step``
     - 0
     - Training step at which the probability starts growing linearly; it is ``0`` before this step.
   * - ``challenging_text_end_step``
     - 0
     - Training step at which ``challenging_text_end_prob`` is reached.

The dataset validates ``0 <= start_prob <= end_prob <= 1`` and, when the feature is enabled,
``0 <= start_step < end_step``. The PO model additionally rejects ``challenging_text_end_prob > 0`` together
with ``gt_phoneme_input_prob > 0``, because the replaced text has no matching IPA and the rollout must predict
its phonemes. Only ``EasyMagpieTTSModelOnlinePO.training_step`` advances the schedule (it pushes
``global_step`` into the dataset before every rollout); other training modes leave the step at 0. For a
constant probability ``p`` use ``challenging_text_start_prob=challenging_text_end_prob=p`` with
``challenging_text_start_step=0`` and ``challenging_text_end_step=1``. The removed
``challenging_text_replacement_prob`` key raises ``ValueError`` when the dataloader is built. A replaced turn
is excluded from partial-phoneme text augmentation, and validation batches are never modified.

``model.train_ds.dataset.context_audio_shuffle_batch_prob`` (default ``0.0``, in ``[0, 1]``) is the
probability that a training batch has its valid audio contexts shuffled between items, so the rollout must
clone a voice that does not belong to the target utterance; text contexts are left unchanged.


Training Losses and Logged Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each training step logs the components of the optimized objective separately:

* ``train_po_loss`` is the GRPO policy loss ``audio_po_loss + phoneme_po_loss_weight * phoneme_po_loss``. For
  each action stream (the audio codebooks and, when ``phoneme_po_loss_weight > 0`` and the rollout used
  predicted phonemes, the phoneme stream) the on-policy surrogate
  ``-exp(logp - logp.detach()) * advantage`` is averaged over the valid tokens of each generation
  (``loss_type=grpo``) or summed over all tokens and divided by the number of rollouts in the teacher-forced
  sub-batch times ``max_decoder_steps`` (``dr_grpo``), zeroed
  for groups rejected by ``best_cer_threshold`` / ``worst_cer_threshold``, and averaged over streams. Because
  ``exp(logp - logp.detach())`` equals 1, the logged value only reflects the advantages, while its gradient is
  ``-advantage * grad(log pi(action))``.
* ``train_kl_loss`` is the raw exact forward KL ``KL(policy || reference)``, evaluated over the full token
  distributions rather than estimated from the sampled token, with the same group masking and stream
  averaging, combined as ``audio_kl_loss + phoneme_po_loss_weight * phoneme_kl_loss``. It is non-zero only
  when ``reference_free=false`` and ``grpo_beta > 0``, and it is logged without the ``grpo_beta`` factor.
* ``train_loss`` is the optimized total
  ``po_loss + grpo_beta * kl_loss + aux_phoneme_loss_weight * phoneme_aux_loss - entropy_coeff * entropy``;
  the entropy term is present only when ``entropy_coeff > 0``, and ``phoneme_aux_loss`` is the supervised
  phoneme loss of rollouts that used ground-truth phonemes (``gt_phoneme_input_prob``).

The per-stream parts (``train_audio_po_loss``, ``train_phoneme_po_loss``, ``train_phoneme_aux_loss``,
``train_audio_kl_loss``, ``train_phoneme_kl_loss``, ``train_entropy``, ``train_audio_entropy``,
``train_phoneme_entropy``), the reward statistics (``train_mean_reward``, ``train_std_reward``,
``train_cfg_fraction``, ``train_mean_reward_cfg``, ``train_mean_reward_no_cfg``),
``train_used_gt_phoneme_input``, ``learning_rate``, timing metrics and gradient/weight diagnostics are logged
alongside.


Migrating from Earlier Recipes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Replace ``+model.inference_cfg_prob=...`` with ``+model.rollout_cfg_mode=alternate`` (or ``off``); a non-zero
  ``inference_cfg_prob`` now raises ``ValueError``. ``inference_cfg_scale`` keeps its meaning, but its default
  moved from 1.0 to 2.5; set it explicitly to keep the previous scale.
* Replace ``model.train_ds.dataset.challenging_text_replacement_prob=p`` with
  ``challenging_text_start_prob=p``, ``challenging_text_end_prob=p``, ``challenging_text_start_step=0`` and
  ``challenging_text_end_step=1``; the old key now raises ``ValueError``.
* Recipes from #16301 that relied on every transcript being normalized, including those using the default
  ``reward_asr_model=nemo`` (in-process parakeet) as well as Nemotron or Qwen recipes, must set
  ``model.normalize_reward_transcript=true``; without it only Whisper transcripts are normalized.
* ``reward_asr_model=nemotron`` without ``reward_asr_model_name`` now scores with the worker's own default,
  ``nvidia/nemotron-3.5-asr-streaming-0.6b``, instead of the in-process ``nvidia/parakeet-ctc-0.6b`` default
  that used to be forwarded; set ``reward_asr_model_name`` explicitly to keep a particular model.
* Validation CER/WER are now transcribed through the reward ASR router by default, and the per-language
  ``val/cer_lang_*`` / ``val/wer_lang_*`` metrics are always logged. Set ``model.validation_asr_backend=default``
  together with ``model.run_val_inference=true`` to keep using the base-class validation ASR.
* Since #16301, ``train_po_loss`` no longer contains the KL term, and ``train_kl_loss`` is the exact forward KL
  over the full token distributions (masked by group validity, a per-generation token mean whatever
  ``loss_type`` is, and logged without ``grpo_beta``) instead of the sampled-token k3 estimate, which was not
  group-masked and entered the policy loss weighted by ``grpo_beta``. ``grpo_beta * kl_loss`` is now added only
  in ``train_loss``, so the effective KL regularization for a given ``grpo_beta`` differs; compare runs with
  ``reference_free=false`` and ``grpo_beta > 0`` accordingly and re-tune ``grpo_beta`` if needed.
* ``model.codec_decode_batch_size`` is a new opt-in key (default ``0``, decode everything at once) and needs no
  migration.
* Prefer a ``model.reward_asr`` block over the flat ``reward_asr_model`` / ``qwen_asr_*`` keys; the flat keys
  are read only when ``model.reward_asr`` is absent.


Complete Command
~~~~~~~~~~~~~~~~

The following command is adapted from ``tests/functional_tests/L2_TTS_Fast_dev_runs_EasyMagpietts_OnlinePO.sh``
and uses the in-process Whisper reward ASR. Adjust the ``vector_quantizer`` overrides, tokenizer names and
``training_modes`` to your codec and checkpoint.

.. code-block:: bash

    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python examples/tts/easy_magpietts.py \
        --config-name easy_magpietts \
        name="EasyMagpieTTS-OnlinePO" \
        +mode=onlinepo_train \
        +init_from_ptl_ckpt=/path/to/easy_magpie_pretraining.ckpt \
        model.phoneme_tokenizer.tokenizer_path=/path/to/bpe_ipa_tokenizer.json \
        model.codecmodel_path=/path/to/codec_model.nemo \
        +model.vector_quantizer._target_=nemo.collections.tts.modules.audio_codec_modules.GroupFiniteScalarQuantizer \
        +model.vector_quantizer.num_groups=8 \
        +model.vector_quantizer.num_levels_per_group="[4, 4, 4, 4, 4]" \
        +train_ds_meta.train.manifest_path=/path/to/train_manifest.json \
        +train_ds_meta.train.audio_dir="/" \
        +train_ds_meta.train.tokenizer_names="[nemotron_nano_30b]" \
        +train_ds_meta.train.feature_dir=null \
        +val_ds_meta.val.manifest_path=/path/to/val_manifest.json \
        +val_ds_meta.val.audio_dir="/" \
        +val_ds_meta.val.tokenizer_names="[nemotron_nano_30b]" \
        +val_ds_meta.val.feature_dir=null \
        max_epochs=1 \
        batch_size=2 \
        ++model.add_language_to_context_text=true \
        '+model.ignore_phoneme_languages=[vi,zh]' \
        '+model.training_modes=[{text_input_mode:streaming,streaming_phonemes_delay:3,streaming_speech_delay:5}]' \
        +model.reference_free=true \
        +model.loss_type=grpo \
        +model.scale_rewards=true \
        +model.grpo_beta=0.0 \
        ++model.reward_asr_model=whisper \
        ++model.normalize_whisper_transcript=true \
        ++model.speaker_verification_model_name=titanet_large \
        +model.n_generations_per_item=2 \
        +model.batch_size_for_chunked_tf=2 \
        +model.max_decoder_steps=300 \
        +model.min_valid_codes_len=4 \
        +model.max_valid_codes_len=490 \
        ++model.aux_phoneme_loss_weight=0.1 \
        ++model.best_cer_threshold=1.0 \
        ++model.worst_cer_threshold=1.0 \
        +model.rollout_cfg_mode=alternate \
        +model.inference_cfg_scale=2.5 \
        +model.gt_phoneme_input_prob=1.0 \
        +model.inference_temperature=0.7 \
        +model.inference_topk=80 \
        +model.inference_phoneme_sampling_method=argmax \
        +model.use_local_transformer_prob=1.0 \
        +model.cer_reward_weight=0.5 \
        +model.ssim_reward_weight=0.5 \
        +model.use_utmos=false \
        +model.utmos_reward_weight=0.0 \
        model.optim.lr=5e-6 \
        ~model.optim.sched \
        trainer.log_every_n_steps=1 \
        trainer.precision=32 \
        trainer.gradient_clip_val=0.0 \
        trainer.devices="[0]" \
        trainer.strategy=auto \
        +trainer.val_check_interval=50 \
        ~trainer.check_val_every_n_epoch \
        model.train_ds.dataloader_params.num_workers=0 \
        model.validation_ds.dataloader_params.num_workers=0

To switch the reward ASR to the routed Qwen3-ASR + Whisper setup, drop ``++model.reward_asr_model`` and
``++model.normalize_whisper_transcript`` and add ``+model.reward_asr`` as shown in
`Reward ASR Configuration`_ (for example through a YAML override file), optionally with
``+model.normalize_reward_transcript=true``.


See Also
########

- :doc:`magpietts`: Main Magpie-TTS documentation
- `Preference Optimization Source Code <https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/tts/models/magpietts_preference_optimization.py>`__

