# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.parts.mixins.transcription import TranscribeConfig
from nemo.collections.tts.models import easy_magpietts as easy_magpietts_module
from nemo.collections.tts.models import easy_magpietts_preference_optimization as po_module
from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.models.easy_magpietts_preference_optimization import EasyMagpieTTSModelOnlinePO
from nemo.collections.tts.modules.magpietts_modules import LocalTransformerType
from nemo.collections.tts.parts.utils.helpers import process_text_for_cer
from nemo.collections.tts.parts.utils.reward_asr import ProcessRewardASRBackend


pytestmark = pytest.mark.unit


def _make_loss_only_model():
    model = EasyMagpieTTSModelOnlinePO.__new__(EasyMagpieTTSModelOnlinePO)
    torch.nn.Module.__init__(model)
    model._cfg = OmegaConf.create({"grpo_beta": 0.01})
    model.reference_free = False
    model.loss_type = "grpo"
    model.max_decoder_steps = 10
    return model


def _make_validation_asr_model(model_cls, cfg):
    """Build a bare ``model_cls`` instance with only ``_cfg`` set: no checkpoint, GPU, or eval ASR models."""
    model = model_cls.__new__(model_cls)
    torch.nn.Module.__init__(model)
    model._cfg = OmegaConf.create(cfg)
    return model


class _ReferenceModelConstructed(Exception):
    """Raised by the patched base ``__init__`` once the PO model starts building its frozen reference copy."""


def _make_validation_epoch_end_model(model_cls, cfg, use_multilingual_asr):
    """Bare ``model_cls`` carrying the state ``on_validation_epoch_end`` reads; ``self.log`` records into a dict.

    The two recorded validation steps cover languages ``en`` (CER 0.1 / WER 0.2) and ``de`` (CER 0.3, 0.5 /
    WER 0.4, 0.6).
    """
    model = _make_validation_asr_model(model_cls, cfg)
    model._device = torch.device("cpu")
    model.local_transformer_type = LocalTransformerType.NO_LT
    model.phoneme_tokenizer = None
    model.run_val_inference = True
    model.use_multilingual_asr = use_multilingual_asr
    model.validation_step_outputs = [
        {
            "val_loss": torch.tensor(1.0),
            "val_codebook_loss": torch.tensor(1.0),
            "val_cer": torch.tensor(0.2),
            "val_wer": torch.tensor(0.3),
            "val_languages": ["en", "de"],
            "val_cer_list": [0.1, 0.3],
            "val_wer_list": [0.2, 0.4],
        },
        {
            "val_loss": torch.tensor(1.0),
            "val_codebook_loss": torch.tensor(1.0),
            "val_cer": torch.tensor(0.5),
            "val_wer": torch.tensor(0.6),
            "val_languages": ["de"],
            "val_cer_list": [0.5],
            "val_wer_list": [0.6],
        },
    ]
    logged = {}

    def _log(name, value, **kwargs):
        logged[name] = value

    model.log = _log
    return model, logged


class _FakeRewardASRRouter:
    """Reward ASR router stand-in: ``hi`` goes to a Whisper-like backend, every other language to a NeMo-like one.

    ``transcribe`` returns ``"raw <language>"`` for each item; the fake backends only carry the per-backend
    ``normalizes_transcripts_by_default`` flag that ``RewardASRBackend`` subclasses declare.
    """

    def __init__(self):
        self.backends = {
            "whisper": SimpleNamespace(normalizes_transcripts_by_default=True),
            "nemo": SimpleNamespace(normalizes_transcripts_by_default=False),
        }

    def backend_for_language(self, language):
        return self.backends["whisper" if language == "hi" else "nemo"]

    def transcribe(self, audio_paths, languages):
        return [f"raw {language}" for language in languages]


def _install_fake_normalizer(model):
    """Make ``model._get_cached_normalizer`` return a normalizer that appends ``" normalized"``.

    Returns the list of languages for which a normalizer was requested, in call order.
    """
    normalized_languages = []

    def _get_cached_normalizer(language):
        normalized_languages.append(language)
        return SimpleNamespace(normalize=lambda text: f"{text} normalized")

    model._get_cached_normalizer = _get_cached_normalizer
    return normalized_languages


def _make_reward_transcript_model(cfg):
    """Bare PO model whose reward ASR is a ``_FakeRewardASRRouter`` and whose normalizer appends ``" normalized"``.

    Returns the model and the list of languages for which ``_get_cached_normalizer`` was requested.
    """
    model = _make_validation_asr_model(EasyMagpieTTSModelOnlinePO, cfg)
    model._reward_asr_router = _FakeRewardASRRouter()
    model.reward_asr_log_samples = 0
    return model, _install_fake_normalizer(model)


def _init_po_model_with_fake_reward_asr(monkeypatch, cfg):
    """Run the real ``EasyMagpieTTSModelOnlinePO.__init__`` on ``cfg`` with every model load stubbed out.

    The base ``EasyMagpieTTSModel.__init__`` only records ``cfg`` and the attributes the PO ``__init__`` reads, the
    reward ASR router is a ``_FakeRewardASRRouter`` and the speaker-verification model is a stub. Returns the model
    and the list of reward ASR configs handed to the router.
    """

    def _base_init(self, cfg, trainer=None):
        torch.nn.Module.__init__(self)
        self._cfg = cfg
        self._trainer = trainer
        self.num_all_tokens_per_codebook = 2048

    monkeypatch.setattr(EasyMagpieTTSModel, "__init__", _base_init)
    router_cfgs = []

    def _fake_router(reward_asr_cfg, device_getter):
        router_cfgs.append(reward_asr_cfg)
        return _FakeRewardASRRouter()

    monkeypatch.setattr(po_module, "RewardASRRouter", _fake_router)
    monkeypatch.setattr(
        po_module.nemo_asr.models.EncDecSpeakerLabelModel,
        "from_pretrained",
        lambda model_name: SimpleNamespace(freeze=lambda: None),
    )
    return EasyMagpieTTSModelOnlinePO(cfg), router_cfgs


def test_action_po_uses_exact_forward_kl():
    model = _make_loss_only_model()
    policy_probs = torch.tensor([0.75, 0.25])
    reference_probs = torch.tensor([0.5, 0.5])

    po_loss, kl_loss, _ = model._compute_action_po_components(
        logits=policy_probs.log().view(1, 1, 2),
        reference_logits=reference_probs.log().view(1, 1, 2),
        targets=torch.tensor([[[0]]]),
        target_lens=torch.tensor([1]),
        vocab_size=2,
        advantages=torch.tensor([0.0]),
        group_validities=torch.tensor([1.0]),
        sampling_temperature=1.0,
    )

    expected_kl = (policy_probs * (policy_probs.log() - reference_probs.log())).sum()
    assert po_loss.item() == pytest.approx(0.0)
    assert kl_loss.item() == pytest.approx(expected_kl.item())


def test_action_po_masks_kl_for_invalid_groups():
    model = _make_loss_only_model()

    po_loss, kl_loss, _ = model._compute_action_po_components(
        logits=torch.tensor([[[4.0, -4.0]]]),
        reference_logits=torch.tensor([[[-4.0, 4.0]]]),
        targets=torch.tensor([[[0]]]),
        target_lens=torch.tensor([1]),
        vocab_size=2,
        advantages=torch.tensor([3.0]),
        group_validities=torch.tensor([0.0]),
        sampling_temperature=1.0,
    )

    assert po_loss.item() == pytest.approx(0.0)
    assert kl_loss.item() == pytest.approx(0.0)


def test_po_validation_transcribes_through_reward_asr_by_default():
    # Without cfg.run_val_inference the PO model does not load the base-class validation ASR models, so
    # without an explicit validation_asr_backend it must transcribe through the reward ASR router.
    model = _make_validation_asr_model(EasyMagpieTTSModelOnlinePO, {})
    assert not hasattr(model, "whisper_model")
    assert not hasattr(model, "_eval_asr_model")
    calls = []

    def _stub_compute_pred_transcripts(predicted_audio_paths, batch_repeated):
        calls.append((predicted_audio_paths, batch_repeated))
        return ["hello", "world"]

    model._compute_pred_transcripts = _stub_compute_pred_transcripts
    batch = {"languages": ["en", "de"], "raw_texts": ["hello", "welt"]}

    transcripts = model._transcribe_for_validation(["a.wav", "b.wav"], batch)

    assert transcripts == ["hello", "world"]
    assert len(calls) == 1
    assert calls[0][0] == ["a.wav", "b.wav"]
    assert calls[0][1] is batch


def test_po_validation_defers_to_base_asr_when_backend_is_default(monkeypatch):
    model = _make_validation_asr_model(EasyMagpieTTSModelOnlinePO, {"validation_asr_backend": "default"})
    sentinel = ["base transcript"]
    base_calls = []

    def _base_transcribe_for_validation(self, predicted_audio_paths, batch):
        base_calls.append((self, predicted_audio_paths, batch))
        return sentinel

    monkeypatch.setattr(EasyMagpieTTSModel, "_transcribe_for_validation", _base_transcribe_for_validation)
    model._compute_pred_transcripts = lambda *args, **kwargs: pytest.fail("reward ASR must not be used")
    batch = {"languages": ["en"]}

    transcripts = model._transcribe_for_validation(["a.wav"], batch)

    assert transcripts is sentinel
    assert len(base_calls) == 1
    assert base_calls[0][0] is model
    assert base_calls[0][1] == ["a.wav"]
    assert base_calls[0][2] is batch


def test_base_validation_reward_backend_requires_reward_transcription_support():
    model = _make_validation_asr_model(EasyMagpieTTSModel, {"validation_asr_backend": "reward"})
    assert not hasattr(model, "_compute_pred_transcripts")

    with pytest.raises(RuntimeError, match="validation_asr_backend='reward'"):
        model._transcribe_for_validation(["a.wav"], {"languages": ["en"]})


def test_base_validation_default_backend_transcribes_with_nemo_asr():
    model = _make_validation_asr_model(EasyMagpieTTSModel, {})
    model.use_multilingual_asr = False
    calls = []

    def _transcribe(paths, batch_size, override_config):
        calls.append((paths, batch_size, override_config))
        return [SimpleNamespace(text="Hello, World!"), SimpleNamespace(text="Second  file")]

    model._eval_asr_model = SimpleNamespace(transcribe=_transcribe)

    transcripts = model._transcribe_for_validation(["a.wav", "b.wav"], {"languages": ["en", "en"]})

    assert transcripts == [process_text_for_cer("Hello, World!"), process_text_for_cer("Second  file")]
    assert len(calls) == 1
    assert calls[0][0] == ["a.wav", "b.wav"]
    assert calls[0][1] == 2
    assert isinstance(calls[0][2], TranscribeConfig)
    assert calls[0][2].batch_size == 2
    assert calls[0][2].num_workers == 0
    assert calls[0][2].use_lhotse is False


def test_po_logs_per_language_val_metrics_regardless_of_multilingual_asr_flag():
    # The PO model no longer forces use_multilingual_asr=True, but its validation ASR is language-routed, so
    # per-language CER/WER must still be aggregated with the cfg default of the flag (False).
    model = _make_validation_asr_model(EasyMagpieTTSModelOnlinePO, {})
    model.use_multilingual_asr = False

    assert model._should_log_per_language_val_metrics() is True


@pytest.mark.parametrize("use_multilingual_asr", [False, True])
def test_base_per_language_val_metrics_follow_multilingual_asr_flag(use_multilingual_asr):
    model = _make_validation_asr_model(EasyMagpieTTSModel, {})
    model.use_multilingual_asr = use_multilingual_asr

    assert model._should_log_per_language_val_metrics() is use_multilingual_asr


def test_po_validation_epoch_end_logs_per_language_cer_wer_without_multilingual_asr_flag():
    model, logged = _make_validation_epoch_end_model(EasyMagpieTTSModelOnlinePO, {}, use_multilingual_asr=False)

    model.on_validation_epoch_end()

    assert logged["val/cer"].item() == pytest.approx(0.35)
    assert logged["val/wer"].item() == pytest.approx(0.45)
    assert logged["val/cer_lang_en"].item() == pytest.approx(0.1)
    assert logged["val/wer_lang_en"].item() == pytest.approx(0.2)
    assert logged["val/cer_lang_de"].item() == pytest.approx(0.4)
    assert logged["val/wer_lang_de"].item() == pytest.approx(0.5)
    assert model.validation_step_outputs == []


@pytest.mark.parametrize("use_multilingual_asr", [False, True])
def test_base_validation_epoch_end_per_language_cer_wer_follow_multilingual_asr_flag(use_multilingual_asr):
    model, logged = _make_validation_epoch_end_model(EasyMagpieTTSModel, {}, use_multilingual_asr)

    model.on_validation_epoch_end()

    assert logged["val/cer"].item() == pytest.approx(0.35)
    per_language_keys = sorted(key for key in logged if "_lang_" in key)
    if use_multilingual_asr:
        assert per_language_keys == ["val/cer_lang_de", "val/cer_lang_en", "val/wer_lang_de", "val/wer_lang_en"]
    else:
        assert per_language_keys == []


def test_online_po_cfg_rejects_removed_inference_cfg_prob():
    # #16301 replaced inference_cfg_prob with rollout_cfg_mode; a recipe that still sets it must fail fast
    # instead of silently training without CFG rollouts.
    with pytest.raises(ValueError, match="rollout_cfg_mode"):
        po_module._validate_online_po_cfg(OmegaConf.create({"inference_cfg_prob": 0.5}))


@pytest.mark.parametrize("cfg", [{}, {"inference_cfg_prob": 0.0}])
def test_online_po_cfg_accepts_zero_or_absent_inference_cfg_prob(cfg):
    po_module._validate_online_po_cfg(OmegaConf.create(cfg))


@pytest.mark.parametrize("backend", ["defualt", "whisper"])
def test_online_po_cfg_rejects_unknown_validation_asr_backend(backend):
    cfg = OmegaConf.create({"validation_asr_backend": backend, "run_val_inference": True})

    with pytest.raises(ValueError, match="validation_asr_backend"):
        po_module._validate_online_po_cfg(cfg)


def test_online_po_cfg_rejects_default_validation_asr_backend_without_run_val_inference():
    with pytest.raises(ValueError, match="run_val_inference"):
        po_module._validate_online_po_cfg(OmegaConf.create({"validation_asr_backend": "default"}))


@pytest.mark.parametrize(
    "cfg",
    [{"validation_asr_backend": "default", "run_val_inference": True}, {"validation_asr_backend": "reward"}, {}],
)
def test_online_po_cfg_accepts_supported_validation_asr_backends(cfg):
    po_module._validate_online_po_cfg(OmegaConf.create(cfg))


def test_online_po_cfg_rejects_unknown_rollout_cfg_mode():
    with pytest.raises(ValueError, match="rollout_cfg_mode"):
        po_module._validate_online_po_cfg(OmegaConf.create({"rollout_cfg_mode": "always"}))


def test_po_init_validates_cfg_before_loading_any_model(monkeypatch):
    def _base_init(self, cfg, trainer=None):
        pytest.fail("the config must be validated before the base model is built")

    monkeypatch.setattr(EasyMagpieTTSModel, "__init__", _base_init)

    with pytest.raises(ValueError, match="inference_cfg_prob"):
        EasyMagpieTTSModelOnlinePO(OmegaConf.create({"inference_cfg_prob": 0.5}))


@pytest.mark.parametrize(
    ("rollout_cfg_mode", "global_step", "expected"),
    [("off", 0, False), ("off", 1, False), ("alternate", 0, False), ("alternate", 1, True), ("alternate", 2, False)],
)
def test_rollout_uses_cfg_only_on_odd_steps_in_alternate_mode(rollout_cfg_mode, global_step, expected):
    model = _make_validation_asr_model(EasyMagpieTTSModelOnlinePO, {})
    # LightningModule.global_step returns self.trainer.global_step once a trainer is attached via self._trainer.
    model._trainer = SimpleNamespace(global_step=global_step)

    assert model._rollout_uses_cfg(rollout_cfg_mode) is expected


def test_po_reference_model_cfg_disables_validation_inference_and_utmos(monkeypatch):
    cfg = OmegaConf.create(
        {"run_val_inference": True, "use_utmos": True, "train_ds": {"dataset": {}}, "validation_ds": {"dataset": {}}}
    )
    received_cfgs = []

    def _base_init(self, cfg, trainer=None):
        received_cfgs.append(cfg)
        if len(received_cfgs) == 1:
            # The PO model's own super().__init__(): give it just enough state to reach the reference-model build.
            torch.nn.Module.__init__(self)
            self._cfg = cfg
            return
        raise _ReferenceModelConstructed

    monkeypatch.setattr(EasyMagpieTTSModel, "__init__", _base_init)

    with pytest.raises(_ReferenceModelConstructed):
        EasyMagpieTTSModelOnlinePO(cfg)

    assert len(received_cfgs) == 2
    assert received_cfgs[0] is cfg
    ref_model_cfg = received_cfgs[1]
    assert ref_model_cfg is not cfg
    assert ref_model_cfg.run_val_inference is False
    assert ref_model_cfg.use_utmos is False
    assert ref_model_cfg.train_ds is None
    assert ref_model_cfg.validation_ds is None
    # The policy model's own cfg keeps its validation settings.
    assert cfg.run_val_inference is True
    assert cfg.use_utmos is True


def test_lhotse_dataloader_rejects_removed_challenging_text_replacement_prob():
    # The key check runs before the multiturn dataset is built, so a bare model with only `_cfg` is enough.
    model = _make_validation_asr_model(EasyMagpieTTSModel, {"use_multiturn_dataset": True})
    dataset_cfg = OmegaConf.create({"dataset": {"challenging_text_replacement_prob": 0.3}})

    with pytest.raises(ValueError, match="challenging_text_start_prob / challenging_text_end_prob"):
        model.get_lhotse_dataloader(dataset_cfg, mode="train")


def test_challenging_text_dataset_keys_check_accepts_schedule_keys():
    dataset_cfg = OmegaConf.create(
        {
            "challenging_texts_path": "texts.txt",
            "challenging_text_start_prob": 0.1,
            "challenging_text_end_prob": 0.5,
            "challenging_text_start_step": 100,
            "challenging_text_end_step": 500,
        }
    )

    easy_magpietts_module._check_challenging_text_dataset_keys(dataset_cfg)


_LEGACY_NEMO_KEYS = {"reward_asr_batch_size": 8, "reward_asr_att_context_size": [70, 1], "reward_asr_log_samples": 2}
_WHISPER_BACKEND_CFG = {"type": "whisper", "model_name": "openai/whisper-large-v3"}


@pytest.mark.parametrize(
    ("legacy_cfg", "expected"),
    [
        pytest.param(
            {},
            {
                "default_backend": "nemo",
                "language_routes": {},
                "log_samples": 0,
                "backends": {"nemo": {"type": "nemo", "batch_size": 16, "attention_context": None}},
            },
            id="defaults",
        ),
        pytest.param(
            {"reward_asr_model": "nemo", **_LEGACY_NEMO_KEYS},
            {
                "default_backend": "nemo",
                "language_routes": {},
                "log_samples": 2,
                "backends": {"nemo": {"type": "nemo", "batch_size": 8, "attention_context": [70, 1]}},
            },
            id="nemo",
        ),
        pytest.param(
            {"reward_asr_model": "nemotron", **_LEGACY_NEMO_KEYS},
            {
                "default_backend": "nemotron",
                "language_routes": {},
                "log_samples": 2,
                "backends": {"nemotron": {"type": "nemo_process", "batch_size": 8, "attention_context": [70, 1]}},
            },
            id="nemotron",
        ),
        pytest.param(
            {"reward_asr_model": "whisper", "reward_asr_log_samples": 1},
            {
                "default_backend": "whisper",
                "language_routes": {},
                "log_samples": 1,
                "backends": {"whisper": _WHISPER_BACKEND_CFG},
            },
            id="whisper",
        ),
        pytest.param(
            {"reward_asr_model": "qwen_whisper"},
            {
                "default_backend": "qwen",
                "language_routes": {"hi": "whisper"},
                "log_samples": 0,
                "backends": {
                    "qwen": {
                        "type": "qwen",
                        "model_name": "Qwen/Qwen3-ASR-0.6B",
                        "batch_size": 4,
                        "max_new_tokens": 256,
                    },
                    "whisper": _WHISPER_BACKEND_CFG,
                },
            },
            id="qwen_whisper-defaults",
        ),
        pytest.param(
            {
                "reward_asr_model": "qwen_whisper",
                "qwen_asr_whisper_languages": ["hi", "ta"],
                "qwen_asr_model_name": "Qwen/Qwen3-ASR-1.7B",
                "qwen_asr_batch_size": 2,
                "qwen_asr_max_new_tokens": 64,
                "qwen_asr_python": "/opt/qwen_asr/bin/python",
                "reward_asr_log_samples": 3,
            },
            {
                "default_backend": "qwen",
                "language_routes": {"hi": "whisper", "ta": "whisper"},
                "log_samples": 3,
                "backends": {
                    "qwen": {
                        "type": "qwen",
                        "model_name": "Qwen/Qwen3-ASR-1.7B",
                        "batch_size": 2,
                        "max_new_tokens": 64,
                        "python_executable": "/opt/qwen_asr/bin/python",
                    },
                    "whisper": _WHISPER_BACKEND_CFG,
                },
            },
            id="qwen_whisper-overrides",
        ),
    ],
)
def test_legacy_reward_asr_cfg_translation(legacy_cfg, expected):
    # Without reward_asr_model_name the nemo / nemotron backend configs must not carry a model_name at all, so each
    # backend applies its own default (parakeet-ctc-0.6b in-process, nemotron-3.5-asr-streaming-0.6b in the worker).
    translated = po_module._translate_legacy_reward_asr_cfg(OmegaConf.create(legacy_cfg))

    # Values read from the DictConfig may still be OmegaConf containers; compare as plain Python.
    assert OmegaConf.to_container(OmegaConf.create(translated)) == expected


@pytest.mark.parametrize(("reward_asr_model", "backend_type"), [("nemo", "nemo"), ("nemotron", "nemo_process")])
def test_legacy_reward_asr_model_name_is_forwarded_when_set(reward_asr_model, backend_type):
    cfg = OmegaConf.create({"reward_asr_model": reward_asr_model, "reward_asr_model_name": "nvidia/custom-asr.nemo"})

    translated = po_module._translate_legacy_reward_asr_cfg(cfg)

    assert translated["backends"][reward_asr_model] == {
        "type": backend_type,
        "model_name": "nvidia/custom-asr.nemo",
        "batch_size": 16,
        "attention_context": None,
    }


def test_legacy_reward_asr_cfg_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown legacy reward_asr_model='canary'; configure model.reward_asr"):
        po_module._translate_legacy_reward_asr_cfg(OmegaConf.create({"reward_asr_model": "canary"}))


def test_legacy_nemotron_reward_asr_uses_the_worker_default_model():
    # Before the fix the translation always forwarded the parakeet default, so reward_asr_model=nemotron scored
    # multilingual audio with the English nvidia/parakeet-ctc-0.6b model in the worker's non-prompted path.
    translated = po_module._translate_legacy_reward_asr_cfg(OmegaConf.create({"reward_asr_model": "nemotron"}))

    # ProcessRewardASRBackend only records its settings here; the worker process starts on the first transcribe().
    # NemoRewardASRBackend is deliberately not built: it downloads and loads an ASR model in its constructor.
    backend = ProcessRewardASRBackend(translated["backends"]["nemotron"], device_getter=lambda: torch.device("cpu"))

    assert backend.worker_backend == "nemo"
    assert backend.model_name == "nvidia/nemotron-3.5-asr-streaming-0.6b"
    assert backend.process is None


@pytest.mark.parametrize("cfg", [{"reward_asr_model": "nemotron"}, {"reward_asr": {"default_backend": "x"}}])
def test_po_init_builds_reward_asr_router_from_reward_asr_or_translated_legacy_keys(monkeypatch, cfg):
    cfg = OmegaConf.create({"reference_free": True, **cfg})

    model, router_cfgs = _init_po_model_with_fake_reward_asr(monkeypatch, cfg)

    assert len(router_cfgs) == 1
    assert isinstance(model._reward_asr_router, _FakeRewardASRRouter)
    if "reward_asr" in cfg:
        assert router_cfgs[0] is cfg.reward_asr
    else:
        # The translated legacy config reaches the router without a model_name, so the nemotron worker applies its
        # own default instead of the English parakeet CTC model the translation used to forward unconditionally.
        assert router_cfgs[0]["default_backend"] == "nemotron"
        nemotron_cfg = router_cfgs[0]["backends"]["nemotron"]
        assert nemotron_cfg["type"] == "nemo_process"
        backend = ProcessRewardASRBackend(nemotron_cfg, device_getter=lambda: torch.device("cpu"))
        assert backend.model_name == "nvidia/nemotron-3.5-asr-streaming-0.6b"
        assert "model_name" not in nemotron_cfg


@pytest.mark.parametrize(
    ("cfg", "expected_normalized_languages"),
    [
        pytest.param({}, ["hi"], id="unset-normalizes-whisper-only"),
        pytest.param({"normalize_reward_transcript": True}, ["hi", "en"], id="explicit-true-normalizes-every-backend"),
        pytest.param({"normalize_reward_transcript": False}, [], id="explicit-false-normalizes-nothing"),
        pytest.param({"normalize_whisper_transcript": False}, [], id="legacy-false-disables-whisper"),
        pytest.param(
            {"normalize_whisper_transcript": False, "normalize_reward_transcript": True},
            ["hi", "en"],
            id="explicit-true-overrides-legacy-false",
        ),
    ],
)
def test_reward_transcripts_are_normalized_per_backend_unless_configured(cfg, expected_normalized_languages):
    # "hi" is routed to the Whisper-like backend (normalizes by default), "en" to the NeMo-like one (does not).
    # Before the fix every backend was normalized whenever the keys were unset, changing the default reward and
    # validation CER of reward_asr_model=nemo recipes.
    model, normalized_languages = _make_reward_transcript_model(cfg)
    languages = ["hi", "en"]

    transcripts = model._compute_pred_transcripts(["a.wav", "b.wav"], {"languages": languages, "raw_texts": ["", ""]})

    assert normalized_languages == expected_normalized_languages
    expected_transcripts = []
    for language in languages:
        suffix = " normalized" if language in expected_normalized_languages else ""
        expected_transcripts.append(process_text_for_cer(f"raw {language}{suffix}"))
    assert transcripts == expected_transcripts


def test_po_init_normalizes_only_whisper_reward_transcripts_when_keys_are_unset(monkeypatch):
    # Same scaffold as the router wiring test, but here _compute_pred_transcripts consults the state the real
    # __init__ leaves behind. Before the fix __init__ resolved one flag (default True) that normalized the transcripts
    # of every backend; with both keys unset only the Whisper-like "hi" transcript may be normalized.
    model, _ = _init_po_model_with_fake_reward_asr(monkeypatch, OmegaConf.create({"reference_free": True}))
    normalized_languages = _install_fake_normalizer(model)
    languages = ["hi", "en"]

    transcripts = model._compute_pred_transcripts(["a.wav", "b.wav"], {"languages": languages, "raw_texts": ["", ""]})

    assert normalized_languages == ["hi"]
    assert transcripts == [process_text_for_cer("raw hi normalized"), process_text_for_cer("raw en")]
