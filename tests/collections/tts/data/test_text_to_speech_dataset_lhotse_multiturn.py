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
from lhotse import CutSet, SupervisionSegment
from lhotse.testing.dummies import dummy_cut, dummy_recording

from nemo.collections.tts.data import text_to_speech_dataset_lhotse_multiturn as multiturn_dataset_module
from nemo.collections.tts.data.text_to_speech_dataset_lhotse_multiturn import MagpieTTSLhotseMultiturnDataset

pytestmark = pytest.mark.unit

SAMPLE_RATE = 22050
CUT_DURATION = 10.0
DATASET_KWARGS = {
    "sample_rate": SAMPLE_RATE,
    "codec_model_samples_per_frame": 1024,
    "codec_model_input_sample_rate": SAMPLE_RATE,
    "frame_stacking_factor": 1,
}
ENABLED_SCHEDULE = {
    "challenging_text_start_prob": 1.0,
    "challenging_text_end_prob": 1.0,
    "challenging_text_start_step": 0,
    "challenging_text_end_step": 1,
}


class _FakeTextTokenizer:
    def encode(self, text, tokenizer_name):
        return [10] * len(text.split())


def _build_dataset(tmp_path, challenging_texts=None, **challenging_text_kwargs):
    kwargs = {**DATASET_KWARGS, **challenging_text_kwargs}
    if challenging_texts is not None:
        challenging_texts_path = tmp_path / "challenging_texts.txt"
        challenging_texts_path.write_text("\n".join(challenging_texts) + "\n", encoding="utf-8")
        kwargs["challenging_texts_path"] = str(challenging_texts_path)
    dataset = MagpieTTSLhotseMultiturnDataset(**kwargs)
    # Tokenizers are initialized lazily inside the dataloader workers; `_prepare_cuts` only needs `encode`.
    dataset.text_tokenizer = _FakeTextTokenizer()
    return dataset


def _tts_cut(supervision_custom):
    cut = dummy_cut(
        0,
        duration=CUT_DURATION,
        recording=dummy_recording(0, duration=CUT_DURATION, sampling_rate=SAMPLE_RATE),
    )
    cut.supervisions = [
        SupervisionSegment(
            id="turn-0",
            recording_id=cut.recording_id,
            start=0.0,
            duration=CUT_DURATION,
            text="original <|12|> text",
            language="en",
            speaker="Assistant",
            custom=supervision_custom,
        )
    ]
    cut.custom = {"task": "tts", "lang": "en"}
    return cut


def test_challenging_text_probability_linear_schedule():
    dataset = MagpieTTSLhotseMultiturnDataset.__new__(MagpieTTSLhotseMultiturnDataset)
    dataset.challenging_text_start_prob = 0.1
    dataset.challenging_text_end_prob = 0.5
    dataset.challenging_text_start_step = 100
    dataset.challenging_text_end_step = 500
    dataset._training_step = SimpleNamespace(value=0)

    assert dataset.get_challenging_text_replacement_prob(99) == 0.0
    assert dataset.get_challenging_text_replacement_prob(100) == pytest.approx(0.1)
    assert dataset.get_challenging_text_replacement_prob(300) == pytest.approx(0.3)
    assert dataset.get_challenging_text_replacement_prob(500) == pytest.approx(0.5)
    assert dataset.get_challenging_text_replacement_prob(700) == pytest.approx(0.5)

    dataset.set_training_step(250)
    assert dataset.get_challenging_text_replacement_prob() == pytest.approx(0.25)


@pytest.mark.parametrize(
    "supervision_custom",
    [{"normalized_text": "original normalized text"}, {"context_text": "speaker prompt"}],
    ids=["normalized_text", "text"],
)
def test_prepare_cuts_challenging_text_replacement_leaves_source_cut_untouched(tmp_path, supervision_custom):
    dataset = _build_dataset(tmp_path, ["CHALLENGE"], **ENABLED_SCHEDULE)
    dataset.set_training_step(1)
    source_cut = _tts_cut(dict(supervision_custom))
    source_supervision = source_cut.supervisions[0]

    prepared_cuts, _ = dataset._prepare_cuts(CutSet.from_cuts([source_cut]))

    prepared_supervision = next(iter(prepared_cuts)).supervisions[0]
    assert prepared_supervision.text == "CHALLENGE"
    assert prepared_supervision.challenging_text_replaced is True
    if "normalized_text" in supervision_custom:
        assert prepared_supervision.normalized_text == "CHALLENGE"
    # The returned cuts are copies; the source cut that the sampler keeps yielding must stay untouched.
    assert source_supervision.text == "original <|12|> text"
    assert source_supervision.custom == supervision_custom
    assert not source_supervision.has_custom("challenging_text_replaced")


@pytest.mark.parametrize(
    "challenging_texts, schedule, step",
    [
        (None, {"challenging_text_end_prob": 0.0}, 0),
        (
            ["CHALLENGE"],
            {
                "challenging_text_start_prob": 0.5,
                "challenging_text_end_prob": 1.0,
                "challenging_text_start_step": 100,
                "challenging_text_end_step": 200,
            },
            99,
        ),
    ],
    ids=["disabled", "not_started"],
)
def test_prepare_cuts_skips_random_draw_when_replacement_probability_is_zero(
    tmp_path, monkeypatch, challenging_texts, schedule, step
):
    dataset = _build_dataset(tmp_path, challenging_texts, **schedule)
    dataset.set_training_step(step)
    assert dataset.get_challenging_text_replacement_prob() == 0.0

    def _fail_on_draw():
        raise AssertionError("random.random() was drawn although the replacement probability is 0")

    monkeypatch.setattr(multiturn_dataset_module.random, "random", _fail_on_draw)
    source_cut = _tts_cut({"normalized_text": "original normalized text"})

    prepared_cuts, batch_tokenizer_names = dataset._prepare_cuts(CutSet.from_cuts([source_cut]))

    prepared_supervision = next(iter(prepared_cuts)).supervisions[0]
    assert batch_tokenizer_names == ["english_phoneme"]
    assert prepared_supervision.text == "original text"
    assert prepared_supervision.normalized_text == "original normalized text"
    assert not prepared_supervision.has_custom("challenging_text_replaced")


@pytest.mark.parametrize(
    "overrides, match",
    [
        (
            {"challenging_text_start_prob": 0.5, "challenging_text_end_prob": 0.2, "challenging_text_end_step": 10},
            "0 <= start_prob <= end_prob <= 1",
        ),
        ({"challenging_text_end_prob": 1.5, "challenging_text_end_step": 10}, "0 <= start_prob <= end_prob <= 1"),
        (
            {"challenging_text_end_prob": 0.5, "challenging_text_start_step": 10, "challenging_text_end_step": 10},
            "0 <= start_step < end_step",
        ),
        ({"challenging_text_end_prob": 0.5}, "0 <= start_step < end_step"),
        ({"challenging_text_end_prob": 0.5, "challenging_text_end_step": 10}, "challenging_texts_path is required"),
    ],
    ids=[
        "start_prob_above_end_prob",
        "end_prob_above_one",
        "start_step_equals_end_step",
        "default_zero_length_schedule",
        "missing_challenging_texts_path",
    ],
)
def test_init_rejects_invalid_challenging_text_config(overrides, match):
    with pytest.raises(ValueError, match=match):
        MagpieTTSLhotseMultiturnDataset(**DATASET_KWARGS, **overrides)


def test_init_rejects_challenging_texts_file_without_texts(tmp_path):
    with pytest.raises(ValueError, match="No challenging texts found"):
        _build_dataset(tmp_path, ["", "   "], challenging_text_end_prob=0.5, challenging_text_end_step=10)


def test_init_loads_challenging_texts_and_skips_blank_lines(tmp_path):
    dataset = _build_dataset(
        tmp_path, ["first", "", "  second  "], challenging_text_end_prob=0.5, challenging_text_end_step=10
    )

    assert dataset.challenging_texts == ["first", "second"]
