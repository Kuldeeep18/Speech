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
