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
"""Multi-speaker decoding on the state-machine path.

``_generate_dynamic_streaming`` is the correct decoder -- it advances each stream's own state
machine and feeds one embedding per stream per step. The chunked path instead pads every stream's
text response out to the length of the slowest stream in the batch, injecting blank tokens into a
stream's own KV cache. All plain-ASR inference uses the state machine; multi-speaker ASR could not,
because a mounted Sortformer dies the moment two streams desynchronise.

The refill block encodes only the streams that need audio, slicing the ASR cache down to that
subset and scattering it back. That works for the ASR encoder. The diarizer keeps ONE batched state
on the module, and in sync mode its ``spkcache`` grows with step count, so it cannot be sliced --
hence the batch-mismatch guard.

Step 0 of the plan: pin the failure end to end, at the decoder rather than at the encoder (where
``tests/collections/asr/test_parallel_expert_encoder_streaming.py`` already pins it).
"""

import collections
import math

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from tests.collections.asr.test_parallel_expert_encoder import _MEL_FEATURES, _SUBSAMPLING_FACTOR
from tests.collections.asr.test_parallel_expert_encoder_streaming import build_toy_streaming_pe_encoder
from tests.collections.speechlm2.test_streaming_stt_automodel import resolve_pretrained_llm

#: Must match the toy PE encoder's ASR branch: `mount_parallel_expert_encoder` rejects a d_model
#: that disagrees with the perception encoder, the modality adapter or the projection.
PE_D_MODEL = 32
CHUNK_SIZE = 2
BLANK_TOKEN = "<blank>"


@pytest.fixture(autouse=True)
def cpu_default_device():
    """Pin CPU, and restore. Sibling modules set a CUDA default at import and never put it back."""
    previous = torch.get_default_device()
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(previous)


def make_pe_cfg(**overrides) -> dict:
    """Model config whose perception dimensions match the toy ParallelExpertEncoder."""
    cfg = {
        "pretrained_llm": resolve_pretrained_llm(),
        "pretrained_asr": "unused-because-load_asr_weights-is-false",
        "load_llm_weights": False,
        "load_asr_weights": False,
        "blank_token": BLANK_TOKEN,
        "chunk_size": CHUNK_SIZE,
        "att_context_size": [12, 3],
        "audio_pad_to": 0,
        "sample_rate": 16000,
        "frame_length_in_secs": _SUBSAMPLING_FACTOR * 0.01,
        "dtype": "float32",
        "freeze_speech_encoder": False,
        "freeze_modality_adapter": False,
        "freeze_modality_proj": False,
        "freeze_llm_model": True,
        "freeze_llm_head": False,
        "freeze_embed_tokens": False,
        "freeze_params": [],
        "prevent_freeze_params": [],
        "perception": {
            "target": "nemo.collections.speechlm2.modules.perception.AudioPerceptionModule",
            "output_dim": 128,
            "encoder": {
                "_target_": "nemo.collections.asr.modules.ConformerEncoder",
                "att_context_size": [12, 3],
                "att_context_style": "chunked_limited",
                "causal_downsampling": True,
                "conv_context_size": "causal",
                "conv_kernel_size": 9,
                "d_model": PE_D_MODEL,
                "feat_in": _MEL_FEATURES,
                "feat_out": -1,
                "ff_expansion_factor": 4,
                "n_heads": 4,
                "n_layers": 1,
                "self_attention_model": "rel_pos",
                "subsampling": "dw_striding",
                "subsampling_conv_channels": 16,
                "subsampling_factor": _SUBSAMPLING_FACTOR,
            },
            "modality_adapter": {
                "_target_": "nemo.collections.speechlm2.modules.perception.IdentityConnector",
                "d_model": PE_D_MODEL,
            },
            "preprocessor": {
                "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
                "features": _MEL_FEATURES,
                "normalize": "per_feature",
                "sample_rate": 16000,
                "window_size": 0.025,
                "window_stride": 0.01,
            },
        },
        "optimizer": {"_target_": "torch.optim.AdamW", "lr": 1e-4},
    }
    cfg.update(overrides)
    return cfg


def _tiny_llm(monkeypatch):
    """Patch the LLM loader with a small real Qwen3, so the turn template and KV cache are real."""
    import nemo.collections.speechlm2.models.streaming_stt_model as mod

    def _fake_loader(model_path_or_name, pretrained_weights=True, dtype=torch.float32, **kwargs):
        config = AutoConfig.from_pretrained(model_path_or_name)
        config.num_hidden_layers = 2
        config.hidden_size = 128
        config.intermediate_size = 256
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.head_dim = 32
        config.tie_word_embeddings = False
        return AutoModelForCausalLM.from_config(config, dtype=dtype)

    monkeypatch.setattr(mod, "load_pretrained_hf", _fake_loader)


@pytest.fixture
def model_with_diarizer(monkeypatch):
    """A StreamingSTTModel whose perception encoder is a real StreamingParallelExpertEncoder."""
    return _build_model_with_diarizer(monkeypatch)


def _build_model_with_diarizer(monkeypatch):
    from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel
    from nemo.collections.speechlm2.parts.pretrained import mount_parallel_expert_encoder

    _tiny_llm(monkeypatch)
    m = StreamingSTTModel(make_pe_cfg())
    m.configure_model()
    mount_parallel_expert_encoder(m, build_toy_streaming_pe_encoder(), source="toy-fixture")
    return m.eval()


def _unequal_length_audio(sample_rate=16000):
    """Three streams of different durations -- the shortest finishes first and they desynchronise."""
    durations = [0.6, 0.9, 1.4]
    lengths = [int(d * sample_rate) for d in durations]
    audios = torch.zeros(len(lengths), max(lengths))
    for i, n in enumerate(lengths):
        generator = torch.Generator(device="cpu").manual_seed(i)
        audios[i, :n] = torch.randn(n, generator=generator) * 0.1
    return audios, lengths


@pytest.mark.unit
def test_every_perception_call_sees_the_full_batch(model_with_diarizer):
    """The invariant that replaced the crash, and the reason the diarizer is now safe.

    The old refill block encoded only the streams that wanted audio, slicing the ASR cache to that
    subset. The ASR encoder tolerated it; the mounted Sortformer did not -- in sync mode its
    ``spkcache`` grows with step count, so a subset step raised ``RuntimeError: The diarizer's
    streaming state was allocated for batch N but this step has batch M``.

    Eager precompute removes subset steps entirely rather than teaching the diarizer to survive
    them. Asserting "no crash" would not pin that: a future change could reintroduce lazy refills
    and still pass whenever the streams happened to stay in lockstep. This asserts the schedule.
    """
    audios, lengths = _unequal_length_audio()
    batch_sizes = []
    original = type(model_with_diarizer.perception).forward

    def spy(self, *args, **kwargs):
        signal = kwargs.get("processed_signal")
        if signal is not None:
            batch_sizes.append(signal.shape[0])
        return original(self, *args, **kwargs)

    type(model_with_diarizer.perception).forward = spy
    try:
        with torch.no_grad():
            model_with_diarizer.generate(
                audios=audios,
                audio_lens=torch.tensor(lengths),
                system_prompt="Transcribe the audio into text.",
                max_new_tokens=8,
                use_state_machine_inference=True,
                chunk_size_override=CHUNK_SIZE,
            )
    finally:
        type(model_with_diarizer.perception).forward = original

    assert batch_sizes, "perception was never called"
    assert set(batch_sizes) == {len(lengths)}, (
        f"perception saw a partial batch: expected every call at {len(lengths)} rows, got "
        f"{sorted(set(batch_sizes))}"
    )


@pytest.mark.unit
def test_dynamic_streaming_decodes_every_stream_with_a_mounted_diarizer(model_with_diarizer):
    """What the fix delivers: the correct decoder, on multi-speaker audio, without crashing.

    Deliberately weak on content -- an untrained toy model emits noise, so asserting anything about
    the words would pin randomness rather than behaviour. What matters is that every stream is
    decoded, independently, and that a short stream finishing early does not take the batch down.
    """
    audios, lengths = _unequal_length_audio()
    with torch.no_grad():
        result = model_with_diarizer.generate(
            audios=audios,
            audio_lens=torch.tensor(lengths),
            system_prompt="Transcribe the audio into text.",
            max_new_tokens=8,
            use_state_machine_inference=True,
            chunk_size_override=CHUNK_SIZE,
        )
    assert len(result.texts) == len(lengths)
    assert all(isinstance(text, str) for text in result.texts)


# ==============================================================================================
# Step 1 -- row independence. The property the whole design rests on.
# ==============================================================================================


def _capture_row0_embeddings(model, audios, lengths):
    """Run one chunked decode, returning row 0's perception output for every perception call.

    The chunked path is used because it is the one that runs today with a mounted diarizer, and
    because its perception schedule -- full batch, every call -- is exactly the schedule eager
    precompute will adopt.
    """
    captured = []
    original = type(model.perception).forward

    def spy(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        captured.append(out[0][0].detach().clone())
        return out

    type(model.perception).forward = spy
    try:
        with torch.no_grad():
            model.generate(
                audios=audios,
                audio_lens=torch.tensor(lengths),
                system_prompt="Transcribe the audio into text.",
                max_new_tokens=4,
                use_state_machine_inference=False,
                chunk_size_override=CHUNK_SIZE,
            )
    finally:
        type(model.perception).forward = original
    return captured


@pytest.mark.unit
@pytest.mark.parametrize("neighbour", ["different_content", "exhausted"])
def test_a_rows_embeddings_do_not_depend_on_its_neighbours(model_with_diarizer, neighbour):
    """Swapping what the OTHER rows carry must not perturb row 0 by a single bit.

    This licenses two things at once. It is why eager precompute can encode a stream earlier than
    the decoder asks for it -- the answer does not depend on who else is in the batch. And the
    ``exhausted`` case is why the precompute loop may zero-pad a row whose audio has run out to keep
    the batch at exactly B, which the diarizer requires.

    Measured at **0.0** on the real phPEE checkpoint; this keeps it true.
    """
    audios, lengths = _unequal_length_audio()
    baseline = _capture_row0_embeddings(model_with_diarizer, audios, lengths)

    altered = audios.clone()
    altered_lengths = list(lengths)
    if neighbour == "different_content":
        altered[1:] = torch.randn(altered[1:].shape, generator=torch.Generator().manual_seed(99)) * 0.1
    else:  # the neighbours have no audio left at all -- what a padded row looks like
        altered[1:] = 0.0
        altered_lengths[1:] = [0, 0]
    variant = _capture_row0_embeddings(model_with_diarizer, altered, altered_lengths)

    common = min(len(baseline), len(variant))
    assert common > 0, "no perception calls captured -- the decode did not run"
    for step in range(common):
        assert torch.equal(baseline[step], variant[step]), (
            f"row 0 changed at perception call {step} when only its NEIGHBOURS changed "
            f"({neighbour}); max|diff| = {(baseline[step] - variant[step]).abs().max().item():.3e}"
        )


# ==============================================================================================
# Step 2 -- frame geometry. The invariant the alignment helper silently papers over.
# ==============================================================================================


@pytest.mark.unit
def test_diarizer_and_asr_frame_counts_already_agree_exactly(model_with_diarizer):
    """``_align_diar_frames`` must be a no-op in production, and never see a zero-width chunk.

    It pads-by-repeat or truncates when the diarizer's frame count disagrees with the ASR's, which
    is a silent correction: a real disagreement would surface as a slightly stale speaker signal
    rather than as an error. Measured over 1137 streaming calls across four configurations, it
    corrected nothing. Pinning that here is what stops a future change to the chunk schedule
    reintroducing the staleness the warning at ``parallel_expert_encoder.py:1131-1138`` describes.
    """
    from nemo.collections.asr.modules.parallel_expert_encoder import ParallelExpertEncoder

    observed = []
    # Accessing a staticmethod through the class yields the plain function already.
    original = ParallelExpertEncoder._align_diar_frames

    def spy(spk_targets, target_len):
        observed.append((spk_targets.shape[1], target_len))
        return original(spk_targets, target_len)

    ParallelExpertEncoder._align_diar_frames = staticmethod(spy)
    try:
        audios, lengths = _unequal_length_audio()
        _capture_row0_embeddings(model_with_diarizer, audios, lengths)
    finally:
        ParallelExpertEncoder._align_diar_frames = staticmethod(original)

    # A lower bound, not just non-emptiness: if a refactor stopped routing audio through the
    # diarizer the assertions below would all hold vacuously and this test would go quiet.
    assert len(observed) >= 10, f"only {len(observed)} alignment calls -- the diarizer barely ran"
    mismatched = [(cur, target) for cur, target in observed if cur != target]
    assert not mismatched, f"{len(mismatched)} of {len(observed)} calls needed correction: {mismatched[:5]}"
    zero_width = [(cur, target) for cur, target in observed if target <= 0 or cur <= 0]
    assert not zero_width, f"zero-width chunk reached the diarizer: {zero_width[:5]}"


# ==============================================================================================
# Step 4 -- schedule equivalence. Eager encoding must be inert.
# ==============================================================================================


def _decode_with_identity_frames(model, audios, lengths, frames_per_chunk):
    """Decode with perception stubbed to emit identifiable frames, returning what each row read.

    Frame ``f`` of chunk ``k`` for row ``b`` is encoded as ``[b, k, f, ...]``, so a consumed frame
    names its own origin. Anything the queues got wrong -- a row reading a neighbour's audio, a
    chunk out of order, a duplicated or dropped frame -- shows up as a mismatch rather than as a
    plausible-looking transcript.

    Returns:
        tuple: ``(consumed, chunks_encoded)`` where ``consumed[b]`` is the ordered list of
        ``(row, chunk, frame)`` triples row ``b`` actually read.
    """
    chunk_counter = [0]
    consumed = collections.defaultdict(list)

    original_perception = type(model.perception).forward

    def stub(self, *args, **kwargs):
        signal = kwargs["processed_signal"]
        rows, hidden = signal.shape[0], model.perception.proj.out_features
        embs = torch.zeros(rows, frames_per_chunk, hidden)
        for b in range(rows):
            for f in range(frames_per_chunk):
                embs[b, f, 0] = b
                embs[b, f, 1] = chunk_counter[0]
                embs[b, f, 2] = f
        chunk_counter[0] += 1
        return embs, None, None

    original_precompute = type(model)._precompute_audio_embeddings

    def spy_precompute(self, *args, **kwargs):
        buffers = original_precompute(self, *args, **kwargs)

        class _Queue(list):
            def __init__(self, row, items):
                super().__init__(items)
                self.row = row

            def pop(self, index=-1):
                frame = super().pop(index)
                consumed[self.row].append((int(frame[0]), int(frame[1]), int(frame[2])))
                return frame

        return [_Queue(b, q) for b, q in enumerate(buffers)]

    type(model.perception).forward = stub
    type(model)._precompute_audio_embeddings = spy_precompute
    try:
        with torch.no_grad():
            model.generate(
                audios=audios,
                audio_lens=torch.tensor(lengths),
                system_prompt="Transcribe the audio into text.",
                max_new_tokens=4,
                use_state_machine_inference=True,
                chunk_size_override=frames_per_chunk,
            )
    finally:
        type(model.perception).forward = original_perception
        type(model)._precompute_audio_embeddings = original_precompute
    return consumed, chunk_counter[0]


@pytest.mark.unit
def test_each_row_reads_exactly_its_own_chunks_in_order(model_with_diarizer):
    """The proof that encoding eagerly changed *when* audio is encoded, never *what* is read.

    The lazy path cannot be run alongside this one for a differential test -- it is deleted, and it
    could not execute with a mounted diarizer anyway. So the contract is asserted against its
    analytic form instead, which is what the lazy path delivered by construction: row ``b`` reads
    its own chunks, in ascending order, every frame exactly once with no gaps.

    Together with row independence (Step 1), this is equivalence: same frames, same order.
    """
    audios, lengths = _unequal_length_audio()
    samples_per_chunk = CHUNK_SIZE * int(model_with_diarizer._samples_per_encoder_frame())
    consumed, chunks_encoded = _decode_with_identity_frames(
        model_with_diarizer, audios, lengths, frames_per_chunk=CHUNK_SIZE
    )

    assert set(consumed) == set(range(len(lengths))), "not every row consumed audio"
    for row, length in enumerate(lengths):
        own_chunks = math.ceil(length / samples_per_chunk)
        expected = [(row, k, f) for k in range(own_chunks) for f in range(CHUNK_SIZE)]
        # A row need not exhaust its queue -- it can reach DONE early -- so compare the prefix it
        # did read. What must hold is that the prefix is exactly right.
        actual = consumed[row]
        assert actual == expected[: len(actual)], (
            f"row {row} read the wrong frames.\n  expected prefix: {expected[:len(actual)][:6]}\n"
            f"  actually read : {actual[:6]}"
        )
        assert len(actual) <= len(expected), f"row {row} read {len(actual)} frames, owns only {len(expected)}"
        # Must cross at least one chunk boundary, or the ordering assertion above proves nothing
        # about chunk sequencing -- which is exactly where a scatter bug would hide. Observed:
        # every row reads 100% of its frames (8/8, 12/12, 18/18 across 9 encoded chunks).
        assert len(actual) > CHUNK_SIZE, (
            f"row {row} read only {len(actual)} frames -- fewer than one chunk, so this test "
            "degenerated into a tautology"
        )


@pytest.mark.unit
def test_a_row_never_reads_a_neighbours_frame(model_with_diarizer):
    """The failure eager encoding could plausibly introduce, isolated.

    Precompute fills B queues from one batched tensor. An off-by-one in that scatter would hand a
    row its neighbour's audio -- which would still decode, still look like speech, and move WER by
    an amount easily mistaken for the expected decoder change.
    """
    audios, lengths = _unequal_length_audio()
    consumed, _ = _decode_with_identity_frames(model_with_diarizer, audios, lengths, frames_per_chunk=CHUNK_SIZE)
    for row, frames in consumed.items():
        foreign = {origin for origin, _, _ in frames if origin != row}
        assert not foreign, f"row {row} read frames belonging to rows {sorted(foreign)}"


@pytest.mark.unit
def test_precompute_encodes_the_longest_stream_and_no_further(model_with_diarizer):
    """Exhausted rows are padded to hold the batch at B, so the schedule is set by the longest row.

    Pinning the call count is what stops a regression that quietly encodes past the end of the
    batch -- correct output, wasted GPU, and invisible without counting.
    """
    audios, lengths = _unequal_length_audio()
    samples_per_chunk = CHUNK_SIZE * int(model_with_diarizer._samples_per_encoder_frame())
    _, chunks_encoded = _decode_with_identity_frames(model_with_diarizer, audios, lengths, frames_per_chunk=CHUNK_SIZE)
    assert chunks_encoded == math.ceil(max(lengths) / samples_per_chunk)


# ==============================================================================================
# Step 6 -- oracle speaker targets, now that the schedule permits slicing them.
# ==============================================================================================


@pytest.mark.unit
@pytest.mark.parametrize("state_machine", [False, True], ids=["chunked", "state_machine"])
def test_oracle_targets_actually_steer_both_decoders(model_with_diarizer, state_machine):
    """Oracle targets must *change the decode* on both paths, not merely be accepted by them.

    A test that only asserts "no crash" would pass just as happily if the targets were dropped on
    the floor -- which is exactly what `oracle_spk_targets` did in the eval script for its whole
    life. So this compares three decodes: no targets, speaker 0 active, speaker 1 active. All three
    must differ, which can only happen if the targets reach the fusion and influence it.
    """
    audios, lengths = _unequal_length_audio()
    frames = math.ceil(max(lengths) / int(model_with_diarizer._samples_per_encoder_frame()))
    n_spk = model_with_diarizer.perception.encoder.diarization_model.sortformer_modules.n_spk

    def decode(active):
        targets = None
        if active is not None:
            targets = torch.zeros(len(lengths), frames, n_spk)
            targets[:, :, active] = 1.0
        with torch.no_grad():
            return model_with_diarizer.generate(
                audios=audios,
                audio_lens=torch.tensor(lengths),
                system_prompt="Transcribe the audio into text.",
                max_new_tokens=6,
                use_state_machine_inference=state_machine,
                chunk_size_override=CHUNK_SIZE,
                **({"spk_targets": targets} if targets is not None else {}),
            ).texts

    without, speaker0, speaker1 = decode(None), decode(0), decode(1)
    assert speaker0 != without, "supplying oracle targets changed nothing -- they were ignored"
    assert speaker0 != speaker1, "which speaker the targets mark active changed nothing"


@pytest.mark.unit
def test_oracle_spk_targets_run_on_the_state_machine_path(model_with_diarizer):
    """Previously refused outright; eager precompute is what makes the slice well defined.

    The old refusal was correct for the old schedule: the dynamic path advanced a subset of
    streams per refill, so no window of a full-utterance target tensor lined up with what was being
    encoded. Every stream now advances together on a fixed ``k * chunk_samples`` grid, so the same
    frame-index window the chunked path always used applies unchanged.

    This matters beyond convenience: oracle targets separate *fusion-alignment* error from
    *diarizer-quality* error, and without them a weak multi-speaker result is ambiguous.
    """
    audios, lengths = _unequal_length_audio()
    frames = math.ceil(max(lengths) / int(model_with_diarizer._samples_per_encoder_frame()))
    n_spk = model_with_diarizer.perception.encoder.diarization_model.sortformer_modules.n_spk
    spk_targets = torch.zeros(len(lengths), frames, n_spk)
    spk_targets[:, :, 0] = 1.0  # one speaker active throughout; content is irrelevant here

    with torch.no_grad():
        result = model_with_diarizer.generate(
            audios=audios,
            audio_lens=torch.tensor(lengths),
            system_prompt="Transcribe the audio into text.",
            max_new_tokens=8,
            use_state_machine_inference=True,
            chunk_size_override=CHUNK_SIZE,
            spk_targets=spk_targets,
        )
    assert len(result.texts) == len(lengths)


@pytest.mark.unit
def test_oracle_targets_reach_the_encoder_on_the_right_frame_window(model_with_diarizer):
    """Each chunk must receive its OWN slice, advancing by exactly one chunk width per call.

    A wrong window still decodes -- it just attributes words to the wrong speaker, which looks like
    a diarizer-quality problem rather than an indexing bug. Encoding the slice's identity is what
    distinguishes the two.
    """
    audios, lengths = _unequal_length_audio()
    frames = math.ceil(max(lengths) / int(model_with_diarizer._samples_per_encoder_frame()))
    n_spk = model_with_diarizer.perception.encoder.diarization_model.sortformer_modules.n_spk
    # Frame t carries its own index, so a received slice names the window it came from.
    spk_targets = torch.arange(frames, dtype=torch.float32).reshape(1, frames, 1).repeat(len(lengths), 1, n_spk)

    seen = []
    original = type(model_with_diarizer.perception).forward

    def spy(self, *args, **kwargs):
        targets = kwargs.get("spk_targets")
        seen.append(None if targets is None or targets.numel() == 0 else int(targets[0, 0, 0].item()))
        return original(self, *args, **kwargs)

    type(model_with_diarizer.perception).forward = spy
    try:
        with torch.no_grad():
            model_with_diarizer.generate(
                audios=audios,
                audio_lens=torch.tensor(lengths),
                system_prompt="Transcribe the audio into text.",
                max_new_tokens=8,
                use_state_machine_inference=True,
                chunk_size_override=CHUNK_SIZE,
                spk_targets=spk_targets,
            )
    finally:
        type(model_with_diarizer.perception).forward = original

    starts = [first for first in seen if first is not None]
    assert starts, "spk_targets never reached the encoder"
    expected = [k * CHUNK_SIZE for k in range(len(starts))]
    assert starts == expected, f"target windows were {starts}, expected {expected}"


def _capture_embeddings(model, audios, lengths, state_machine, spk_targets):
    """Every row's perception output, per perception call, for one decode."""
    captured = []
    original = type(model.perception).forward

    def spy(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        captured.append(out[0].detach().clone())
        return out

    type(model.perception).forward = spy
    try:
        with torch.no_grad():
            model.generate(
                audios=audios,
                audio_lens=torch.tensor(lengths),
                system_prompt="Transcribe the audio into text.",
                max_new_tokens=4,
                use_state_machine_inference=state_machine,
                chunk_size_override=CHUNK_SIZE,
                **({"spk_targets": spk_targets} if spk_targets is not None else {}),
            )
    finally:
        type(model.perception).forward = original
    return captured


@pytest.mark.unit
@pytest.mark.parametrize("state_machine", [False, True], ids=["chunked", "state_machine"])
def test_sentinel_rows_use_the_diarizer_on_both_decoders(monkeypatch, state_machine):
    """A row whose oracle targets are the missing-RTTM sentinel (a cut without an RTTM) must be
    encoded with the embedded diarizer's predictions, not fused as silence, while the other rows
    keep their oracle targets. Checked on the embeddings, which the LLM's output does not feed back.

    Seeded: the toy fuses thresholded activity, and some initialisations put every prediction above
    the threshold. Such a speaker-constant row maps to ``diar_norm``'s bias, as silence does.
    """
    torch.manual_seed(0)
    model_with_diarizer = _build_model_with_diarizer(monkeypatch)
    audios, lengths = _unequal_length_audio()
    frames = math.ceil(max(lengths) / int(model_with_diarizer._samples_per_encoder_frame()))
    n_spk = model_with_diarizer.perception.encoder.diarization_model.sortformer_modules.n_spk
    generator = torch.Generator().manual_seed(0)
    oracle = torch.rand(len(lengths), frames, n_spk, generator=generator)
    mixed = oracle.clone()
    mixed[1] = model_with_diarizer.perception.encoder.missing_rttm_target
    silence = oracle.clone()
    silence[1] = 0.0

    def capture(targets):
        return _capture_embeddings(model_with_diarizer, audios, lengths, state_machine, targets)

    diarizer, with_oracle, with_mixed, with_silence = capture(None), capture(oracle), capture(mixed), capture(silence)
    assert len(diarizer) == len(with_oracle) == len(with_mixed) == len(with_silence) > 0
    assert any(not torch.equal(d[1], s[1]) for d, s in zip(diarizer, with_silence)), "degenerate toy"
    for call, (d, o, m) in enumerate(zip(diarizer, with_oracle, with_mixed)):
        assert torch.equal(m[1], d[1]), f"the sentinel row is not the diarizer's at perception call {call}"
        assert torch.equal(m[0], o[0]) and torch.equal(m[2], o[2]), f"a real row changed at call {call}"
