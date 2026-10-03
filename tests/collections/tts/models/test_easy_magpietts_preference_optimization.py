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
from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.models.easy_magpietts_preference_optimization import EasyMagpieTTSModelOnlinePO
from nemo.collections.tts.modules.magpietts_modules import LocalTransformerType
from nemo.collections.tts.parts.utils.helpers import process_text_for_cer


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
