# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""The ParallelExpertEncoder mount must exist on BOTH streaming model paths.

`StreamingSTTModelAutomodel` subclasses `StreamingSTTModel` but builds its own perception module in
`configure_model` instead of reusing the base class's `__init__`, so the mount is NOT inherited --
it has to be repeated. Without it, `model.pe_encoder_path` / `model.parallel_expert_encoder` are
silently ignored under Automodel and training runs with a plain ASR encoder.

Most checks are source-level, so they stay cheap and do not need checkpoints or a GPU. The
behavioural tests mount a toy bundle `.nemo` into a real `StreamingSTTModel` over a tiny LLM.
"""

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = REPO_ROOT / "nemo/collections/speechlm2/models/streaming_stt_model.py"
AUTOMODEL = REPO_ROOT / "nemo/collections/speechlm2/models/streaming_stt_model_automodel.py"

MOUNTS = ("setup_parallel_expert_encoder", "setup_parallel_expert_encoder_from_checkpoints")


def _function_node(path, name):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def _called_names(node):
    """{callee_name: first_lineno} for both plain and method calls inside `node`."""
    out = {}
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name and name not in out:
            out[name] = sub.lineno
    return out


@pytest.mark.unit
@pytest.mark.parametrize(
    "path, fn",
    [(BASE, "__init__"), (AUTOMODEL, "configure_model")],
    ids=["StreamingSTTModel", "StreamingSTTModelAutomodel"],
)
def test_both_paths_mount_the_parallel_expert_encoder(path, fn):
    calls = _called_names(_function_node(path, fn))
    assert "setup_perception" in calls, f"{path.name}:{fn} no longer builds perception here"
    for mount in MOUNTS:
        assert mount in calls, f"{path.name}:{fn} does not mount the PE encoder via {mount}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "path, fn",
    [(BASE, "__init__"), (AUTOMODEL, "configure_model")],
    ids=["StreamingSTTModel", "StreamingSTTModelAutomodel"],
)
def test_pe_mount_runs_after_perception_and_before_freeze(path, fn):
    """Order matters: the mount replaces `perception.encoder`, and `_apply_freeze_config` must run
    afterwards so a composite encoder's `apply_internal_freeze` is applied to the MOUNTED encoder
    rather than the one the mount discards."""
    calls = _called_names(_function_node(path, fn))
    assert calls["setup_perception"] < calls[MOUNTS[0]], "PE mount runs before perception is built"
    assert calls[MOUNTS[0]] < calls["_apply_freeze_config"], "PE mount runs after the freeze config"


@pytest.mark.unit
@pytest.mark.parametrize(
    "path, fn",
    [(BASE, "__init__"), (AUTOMODEL, "configure_model")],
    ids=["StreamingSTTModel", "StreamingSTTModelAutomodel"],
)
def test_both_paths_mount_a_bundle_as_the_streaming_encoder(path, fn):
    """Both models decode chunk by chunk, so both must ask the bundle mount for the streaming class.
    The Automodel path cannot be built on CPU without Automodel; the behavioural test below covers
    `StreamingSTTModel`."""
    calls = [
        node
        for node in ast.walk(_function_node(path, fn))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == MOUNTS[0]
    ]
    assert calls, f"{path.name}:{fn} does not call {MOUNTS[0]}"
    for call in calls:
        streaming = [kw.value for kw in call.keywords if kw.arg == "streaming"]
        assert (
            streaming and isinstance(streaming[0], ast.Constant) and streaming[0].value is True
        ), f"{path.name}:{call.lineno} mounts a bundle without streaming=True"


# Keys the bundle mount reads from the raw `model.cfg` rather than from `StreamingSTTModelConfig`.
_PE_BUNDLE_MOUNT_KEYS = {
    "pe_encoder_overrides": {"sync_max_audio_length": False},
    "encoder_chunk_size_seconds": 1.0,
    "spk_kernel_scale": 1.0,
    "packed_encoder_sequences": False,
    "encoder_chunk_batch_size": None,
}


@pytest.mark.unit
def test_no_unsupported_warning_for_pe_keys(tmp_path, monkeypatch):
    """`to_dataclass` must not call the bundle mount's keys "not supported and will be ignored":
    the mount reads and honours them. Keys nothing reads are still reported."""
    import torch

    import nemo.collections.speechlm2.parts.utils.misc as misc
    from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel
    from tests.collections.asr.test_parallel_expert_encoder import write_toy_bundle
    from tests.collections.speechlm2.test_streaming_stt_dynamic_diarizer import _tiny_llm, make_pe_cfg

    previous = torch.get_default_device()
    torch.set_default_device("cpu")
    warnings = []
    monkeypatch.setattr(misc.logging, "warning", lambda msg, *args, **kwargs: warnings.append(msg % args))
    _tiny_llm(monkeypatch)
    try:
        cfg = make_pe_cfg(pe_encoder_path=write_toy_bundle(tmp_path / "pe.nemo"), **_PE_BUNDLE_MOUNT_KEYS)
        model = StreamingSTTModel(cfg)
    finally:
        torch.set_default_device(previous)

    unsupported = [line for line in warnings if "not supported and will be ignored" in line]
    assert unsupported, "the control key `optimizer` is no longer reported"
    assert "optimizer" in unsupported[0]
    for key in _PE_BUNDLE_MOUNT_KEYS:
        assert all(key not in line for line in unsupported), f"{key!r} is reported as ignored: {unsupported}"
    # ...and the mount did honour them.
    assert model.perception.encoder.asr_encoder.sync_max_audio_length is False
    assert model.perception.encoder.chunk_size_seconds == 1.0


def _write_streaming_toy_bundle(path) -> str:
    """A toy PE bundle whose ASR branch is cache-aware, so chunked decoding can run over it."""
    from tests.collections.asr.test_parallel_expert_encoder import write_toy_bundle
    from tests.collections.asr.test_parallel_expert_encoder_streaming import (
        build_toy_streaming_pe_encoder,
        streaming_asr_encoder_cfg,
    )

    return write_toy_bundle(
        path,
        encoder=build_toy_streaming_pe_encoder(),
        asr_encoder_cfg=streaming_asr_encoder_cfg(),
        asr_normalize_type=None,
    )


@pytest.mark.unit
@pytest.mark.parametrize("state_machine", [False, True], ids=["chunked", "state_machine"])
def test_bundle_mounts_the_streaming_encoder_and_chunked_generate_runs(tmp_path, monkeypatch, state_machine):
    """A StreamingSTT model decodes chunk by chunk, so a bundle (`model.pe_encoder_path`) must mount
    a `StreamingParallelExpertEncoder`, as the two-checkpoint route already does. Mounting the plain
    `ParallelExpertEncoder` fails at the first chunk (`AttributeError: ... 'setup_streaming_params'`)."""
    import torch

    from nemo.collections.asr.modules.parallel_expert_encoder import StreamingParallelExpertEncoder
    from nemo.collections.asr.parts.mixins.streaming import StreamingEncoder
    from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel
    from tests.collections.speechlm2.test_streaming_stt_dynamic_diarizer import (
        CHUNK_SIZE,
        _tiny_llm,
        _unequal_length_audio,
        make_pe_cfg,
    )

    previous = torch.get_default_device()
    torch.set_default_device("cpu")
    _tiny_llm(monkeypatch)
    try:
        model = StreamingSTTModel(make_pe_cfg(pe_encoder_path=_write_streaming_toy_bundle(tmp_path / "pe.nemo")))
        model = model.eval()
        encoder = model.perception.encoder
        assert isinstance(encoder, StreamingParallelExpertEncoder), type(encoder).__name__
        assert isinstance(encoder, StreamingEncoder)

        audios, lengths = _unequal_length_audio()
        with torch.no_grad():
            result = model.generate(
                audios=audios,
                audio_lens=torch.tensor(lengths),
                system_prompt="Transcribe the audio into text.",
                max_new_tokens=8,
                use_state_machine_inference=state_machine,
                chunk_size_override=CHUNK_SIZE,
            )
    finally:
        torch.set_default_device(previous)
    assert len(result.texts) == len(lengths)
    assert all(isinstance(text, str) for text in result.texts)


# ----------------------------------------------------------------------------- #
# Speaker-count guards at construction (model <-> dataset <-> mounted PE)
# ----------------------------------------------------------------------------- #
def _write_toy_bundle_with_speakers(path, n_spk: int, **cfg_overrides) -> str:
    """A toy PE bundle whose diarizer fuses ``n_spk`` speaker columns."""
    from tests.collections.asr.test_parallel_expert_encoder import (
        build_toy_pe_encoder,
        toy_diarization_model_cfg,
        write_toy_bundle,
    )

    diar_cfg = toy_diarization_model_cfg()
    diar_cfg.sortformer_modules.num_spks = n_spk
    diar_cfg.max_num_of_spks = n_spk
    encoder = build_toy_pe_encoder(diarization_model_cfg=diar_cfg)
    return write_toy_bundle(path, encoder=encoder, diarization_model_cfg=diar_cfg, **cfg_overrides)


def _speaker_data_cfg(num_speakers: int, **multispeaker_overrides):
    from omegaconf import OmegaConf

    multispeaker_cfg = {"enable": True, "num_speakers": num_speakers, **multispeaker_overrides}
    return OmegaConf.create({"words_per_group": 1, "multispeaker_cfg": multispeaker_cfg})


def _build_speaker_model(monkeypatch, bundle, max_speakers, data_cfg=None, val_data_cfg=None):
    import torch

    from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel
    from tests.collections.speechlm2.test_streaming_stt_dynamic_diarizer import _tiny_llm, make_pe_cfg

    previous = torch.get_default_device()
    torch.set_default_device("cpu")
    _tiny_llm(monkeypatch)
    try:
        cfg = make_pe_cfg(
            pe_encoder_path=bundle,
            speaker_tokens={"enable": True, "template": "<spk:{i}>", "max_speakers": max_speakers},
        )
        return StreamingSTTModel(cfg, data_cfg=data_cfg, val_data_cfg=val_data_cfg)
    finally:
        torch.set_default_device(previous)


@pytest.mark.unit
@pytest.mark.parametrize(
    "pe_spk,data_spk,in_val_only",
    [(4, 8, False), (8, 4, False), (4, 8, True)],
    ids=["8_columns_into_a_4_speaker_pe", "4_columns_into_an_8_speaker_pe", "validation_config_only"],
)
def test_streaming_stt_rejects_fusion_width_mismatch(tmp_path, monkeypatch, pe_spk, data_spk, in_val_only):
    """PF-5: `multispeaker_cfg.num_speakers` must equal the mounted PE's `n_spk`. Before the guard
    the model constructed and the first forward with speaker targets failed with an opaque shape
    `RuntimeError` (`Given normalized_shape=[n_spk]` in the fusion LayerNorm, or a size mismatch in
    the missing-RTTM splice)."""
    bundle = _write_toy_bundle_with_speakers(tmp_path / "pe.nemo", n_spk=pe_spk)
    data_cfg = _speaker_data_cfg(pe_spk if in_val_only else data_spk)
    val_data_cfg = _speaker_data_cfg(data_spk) if in_val_only else None
    with pytest.raises(ValueError, match=rf"n_spk={pe_spk} .*num_speakers={data_spk}"):
        _build_speaker_model(monkeypatch, bundle, max_speakers=4, data_cfg=data_cfg, val_data_cfg=val_data_cfg)


@pytest.mark.unit
def test_eight_columns_four_tags_constructs(tmp_path, monkeypatch):
    """Compatibility pin: the reference configuration fuses 8 diarizer columns (an 8-speaker
    Sortformer, `num_speakers: 8`) and registers 4 tags (`max_speakers: 4`). It must construct, its
    dataset must build against the model's tokenizer, and 8-column targets must fuse."""
    import torch

    from nemo.collections.speechlm2.data.streaming_stt_dataset import StreamingSTTDataset

    data_cfg = _speaker_data_cfg(8)
    model = _build_speaker_model(
        monkeypatch, _write_toy_bundle_with_speakers(tmp_path / "pe.nemo", n_spk=8), 4, data_cfg=data_cfg
    )
    assert model.perception.encoder.n_spk == 8
    assert len(model.speaker_token_ids) == 4

    dataset_cfg = _speaker_data_cfg(8, sample_rate=16000, window_stride=0.01, subsampling_factor=8)
    dataset_cfg.update({"sample_rate": 16000, "frame_length_in_secs": 0.08, "chunk_size": 2, "blank_token": "<blank>"})
    dataset = StreamingSTTDataset(cfg=dataset_cfg, tokenizer=model.tokenizer)
    assert dataset._num_speaker_tags == 4

    previous = torch.get_default_device()
    torch.set_default_device("cpu")
    try:
        audio = torch.randn(1, 16000) * 0.1
        frames = 16000 // 160 // 8
        with torch.no_grad():
            embs, _ = model.perception(
                input_signal=audio, input_signal_length=torch.tensor([16000]), spk_targets=torch.rand(1, frames, 8)
            )
    finally:
        torch.set_default_device(previous)
    assert torch.isfinite(embs).all()


@pytest.mark.unit
def test_streaming_stt_rejects_more_tags_than_columns(tmp_path, monkeypatch):
    """`speaker_tokens.max_speakers` must not exceed `num_speakers`: a tag beyond the last target
    column names a speaker the targets cannot describe."""
    bundle = _write_toy_bundle_with_speakers(tmp_path / "pe.nemo", n_spk=4)
    with pytest.raises(ValueError, match=r"max_speakers=8 exceeds .*num_speakers=4"):
        _build_speaker_model(monkeypatch, bundle, max_speakers=8, data_cfg=_speaker_data_cfg(4))


@pytest.mark.unit
def test_streaming_stt_rejects_sentinel_mismatch(tmp_path, monkeypatch):
    """The PE fills rows at or below its `missing_rttm_target` from the diarizer. A dataset writing
    another sentinel would have those rows fused as real speaker activity."""
    bundle = _write_toy_bundle_with_speakers(tmp_path / "pe.nemo", n_spk=4)
    with pytest.raises(ValueError, match=r"missing_rttm_target=-1.0 .*missing_rttm_target=-2.0"):
        _build_speaker_model(
            monkeypatch, bundle, max_speakers=4, data_cfg=_speaker_data_cfg(4, missing_rttm_target=-2.0)
        )
    # Equal values on both sides construct.
    bundle = _write_toy_bundle_with_speakers(tmp_path / "pe2.nemo", n_spk=4, missing_rttm_target=-2.0)
    model = _build_speaker_model(
        monkeypatch, bundle, max_speakers=4, data_cfg=_speaker_data_cfg(4, missing_rttm_target=-2.0)
    )
    assert model.perception.encoder.missing_rttm_target == -2.0


@pytest.mark.unit
@pytest.mark.parametrize("data_cfg", ["absent", "disabled"])
def test_speaker_guard_is_inert_without_multispeaker_data(tmp_path, monkeypatch, data_cfg):
    """Without an enabled `multispeaker_cfg` no speaker targets reach the PE, so a width that
    differs from the (unused) data setting is not an error, and inference/HF reload
    (`data_cfg=None`) is unaffected."""
    bundle = _write_toy_bundle_with_speakers(tmp_path / "pe.nemo", n_spk=4)
    cfg = None if data_cfg == "absent" else _speaker_data_cfg(8, enable=False)
    model = _build_speaker_model(monkeypatch, bundle, max_speakers=8, data_cfg=cfg)
    assert model.perception.encoder.n_spk == 4


@pytest.mark.unit
def test_automodel_rejects_fusion_width_mismatch_after_mount(tmp_path, monkeypatch):
    """The Automodel variant mounts the PE in `configure_model`, so its PE checks run there, against
    the dataset configs given to `__init__`. The encoder-free check runs in `__init__`."""
    import torch

    from nemo.collections.speechlm2.models import StreamingSTTModelAutomodel
    from tests.collections.speechlm2.test_streaming_stt_automodel import tiny_llm_factory
    from tests.collections.speechlm2.test_streaming_stt_dynamic_diarizer import make_pe_cfg

    bundle = _write_toy_bundle_with_speakers(tmp_path / "pe.nemo", n_spk=4)
    speaker_tokens = {"enable": True, "template": "<spk:{i}>", "max_speakers": 4}
    previous = torch.get_default_device()
    torch.set_default_device("cpu")
    tiny_llm_factory(monkeypatch)
    try:
        cfg = make_pe_cfg(pe_encoder_path=bundle, use_nemo_automodel=True, speaker_tokens=speaker_tokens)
        with pytest.raises(ValueError, match=r"max_speakers=4 exceeds .*num_speakers=2"):
            StreamingSTTModelAutomodel(cfg, data_cfg=_speaker_data_cfg(2))
        model = StreamingSTTModelAutomodel(cfg, data_cfg=_speaker_data_cfg(8))
        with pytest.raises(ValueError, match=r"n_spk=4 .*num_speakers=8"):
            model.configure_model()
    finally:
        torch.set_default_device(previous)
