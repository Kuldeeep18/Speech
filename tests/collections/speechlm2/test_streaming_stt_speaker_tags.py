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

"""SOT ``<spk:N>`` tag emission in the interleaved per-chunk targets."""

import re

import pytest

from nemo.collections.asr.parts.utils.sot_speaker_alignment import sot_to_speaker_texts
from nemo.collections.speechlm2.data.streaming_stt_dataset import get_llm_messages_for_sample
from nemo.collections.speechlm2.parts.alignments import WordAlignment

TAG = re.compile(r"<spk:(\d+)>")


def _align(spec, step=0.2):
    """``spec`` is [(word, speaker), ...] laid out on a regular grid."""
    return [WordAlignment(w, i * step, i * step + step * 0.9, speaker=s) for i, (w, s) in enumerate(spec)]


def _turns(alignments, transcript, **kwargs):
    """Non-blank assistant contents; ``kwargs`` go to :func:`_messages`."""
    messages = _messages(alignments, transcript, **kwargs)
    return [m["content"] for m in messages if m["role"] == "assistant" and m["content"] != "<blank>"]


def _messages(
    alignments,
    transcript,
    *,
    chunk_size=2,
    write_token="",
    prepend=False,
    template="<spk:{i}>",
    num_delay_frames=0,
    audio_duration_secs=None,
    **kwargs,
):
    return get_llm_messages_for_sample(
        system_role="system",
        system_prompt="Transcribe.",
        audio_tag="<audio>",
        blank_token="<blank>",
        chunk_size=chunk_size,
        num_delay_frames=num_delay_frames,
        audio_duration_secs=(len(alignments) * 0.2 + 0.5 if audio_duration_secs is None else audio_duration_secs),
        frame_length_in_secs=0.08,
        alignments=alignments,
        transcript=transcript,
        words_per_group=1,
        prepend_write_token=prepend,
        write_token=write_token,
        speaker_token_template=template,
        **kwargs,
    )


def _speaker_texts(text, placement="prefix", switch="<spk_switch>"):
    """Per-speaker word lists, as cpWER sees them."""
    text = text.replace("<|write|>", " ").replace(switch, " ")
    return {k: v.split() for k, v in sot_to_speaker_texts(text, placement=placement).items()}


class TestSpeakerTagEmission:
    @pytest.mark.unit
    def test_tag_emitted_only_on_speaker_change(self):
        al = _align([("a", 0), ("b", 0), ("c", 1), ("d", 1)])
        out = " ".join(_turns(al, "<spk:0> a b <spk:1> c d"))
        assert TAG.findall(out) == ["0", "1"], "one tag per change, not per word"

    @pytest.mark.unit
    def test_emitted_tag_sequence_matches_the_transcript(self):
        transcript = "<spk:0> a <spk:1> b <spk:0> c d <spk:2> e"
        al = _align([("a", 0), ("b", 1), ("c", 0), ("d", 0), ("e", 2)])
        assert TAG.findall(" ".join(_turns(al, transcript))) == TAG.findall(transcript)

    @pytest.mark.unit
    def test_write_token_stays_outermost(self):
        # Q11: `prepend_write_token` exists so the LM's first output token is a binary blank/write
        # decision. A tag outside it would make that distribution multi-modal.
        al = _align([("a", 0), ("b", 1)])
        out = _turns(al, "<spk:0> a <spk:1> b", write_token="<|write|>", prepend=True)
        tagged = [c for c in out if "<spk:" in c]
        assert tagged, "expected at least one tagged turn"
        assert all(c.startswith("<|write|><spk:") for c in tagged)

    @pytest.mark.unit
    def test_no_double_space_after_an_injected_tag(self):
        al = _align([("alpha", 0), ("beta", 1)])
        for content in _turns(al, "<spk:0> alpha <spk:1> beta"):
            assert "  " not in content

    @pytest.mark.unit
    def test_template_none_disables_tagging(self):
        al = _align([("a", 0), ("b", 1)])
        out = " ".join(_turns(al, "<spk:0> a <spk:1> b", template=None))
        assert "<spk:" not in out

    @pytest.mark.unit
    def test_words_without_speaker_are_untagged(self):
        # Single-speaker manifests carry no `speaker_ids`; the path must stay inert.
        al = [WordAlignment("a", 0.0, 0.2), WordAlignment("b", 0.3, 0.5)]
        assert "<spk:" not in " ".join(_turns(al, "a b"))

    @pytest.mark.unit
    def test_state_follows_the_last_word_of_a_group(self):
        # A group's transcript slice can already carry a mid-group tag, so the "last emitted
        # speaker" is the group's LAST word, not its first. Tracking the first would re-emit a
        # redundant tag (or drop a needed one) on the following group.
        transcript = "<spk:0> a <spk:1> b c"
        al = _align([("a", 0), ("b", 1), ("c", 1)])
        out = " ".join(_turns(al, transcript, chunk_size=40))  # force one big group
        assert TAG.findall(out) == ["0", "1"]


# Multi-speaker transcripts in the manifests' prefix format, with the speaker of every word.
_SOT_CASES = {
    "two_runs": ("<spk:0> a b <spk:1> c d", [("a", 0), ("b", 0), ("c", 1), ("d", 1)]),
    "back_and_forth": (
        "<spk:0> a <spk:1> b <spk:0> c d <spk:2> e",
        [("a", 0), ("b", 1), ("c", 0), ("d", 0), ("e", 2)],
    ),
    "one_word_runs": (
        "<spk:1> yes <spk:0> no <spk:1> yes <spk:2> maybe",
        [("yes", 1), ("no", 0), ("yes", 1), ("maybe", 2)],
    ),
    "punctuation": (
        "<spk:0> Hello, there. <spk:1> Hi! <spk:0> OK.",
        [("hello", 0), ("there", 0), ("hi", 1), ("ok", 0)],
    ),
    "single_speaker": ("<spk:0> just one voice here", [("just", 0), ("one", 0), ("voice", 0), ("here", 0)]),
}


class TestSuffixPlacementAndSwitchToken:
    """PF-6: suffix placement, the flush turn and the switch token build targets from ``speaker_ids``."""

    @pytest.mark.unit
    @pytest.mark.parametrize("chunk_size", [2, 5, 40])
    @pytest.mark.parametrize("switch", [None, "<spk_switch>"])
    @pytest.mark.parametrize("case", sorted(_SOT_CASES))
    def test_suffix_placement_attributes_words(self, case, switch, chunk_size):
        # A chunk that spans a speaker change used to carry the NEXT run's prefix tag from the
        # transcript slice where the suffix target needs the closing tag of the run it ends, so
        # the words before it went to the wrong speaker.
        transcript, spec = _SOT_CASES[case]
        al = _align(spec)
        out = " ".join(
            _turns(al, transcript, chunk_size=chunk_size, speaker_tag_placement="suffix", speaker_switch_token=switch)
        )
        assert _speaker_texts(out, placement="suffix") == _speaker_texts(transcript)
        # Every run is closed exactly once, by its own speaker, and opened by one switch token.
        runs = [s for i, (_, s) in enumerate(spec) if i + 1 == len(spec) or spec[i + 1][1] != s]
        assert TAG.findall(out) == [str(s) for s in runs]
        assert out.count("<spk_switch>") == (len(runs) if switch else 0)

    @pytest.mark.unit
    def test_suffix_placement_one_chunk_exact(self):
        al = _align(_SOT_CASES["two_runs"][1])
        transcript = _SOT_CASES["two_runs"][0]
        assert _turns(al, transcript, chunk_size=40, speaker_tag_placement="suffix") == [" a b <spk:0> c d <spk:1>"]
        assert _turns(
            al, transcript, chunk_size=40, speaker_tag_placement="suffix", speaker_switch_token="<spk_switch>"
        ) == ["<spk_switch> a b <spk:0> <spk_switch> c d <spk:1>"]

    @pytest.mark.unit
    def test_suffix_placement_keeps_non_speaker_markup(self):
        # Only speaker tags are rebuilt; other markup between the runs stays in the target.
        al = _align([("a", 0), ("b", 1)])
        out = _turns(al, "<spk:0> a <laugh> <spk:1> b", chunk_size=40, speaker_tag_placement="suffix")
        assert out == [" a <laugh> <spk:0> b <spk:1>"]

    @pytest.mark.unit
    @pytest.mark.parametrize("placement", ["prefix", "suffix"])
    def test_flush_turn_keeps_leading_tag(self, placement):
        # `a b` are emitted in the last chunk; the delay pushes `c` (speaker 1) and `d` (speaker 2)
        # past the last boundary, into the flush turn. That turn opens speaker 1's run, so under
        # prefix it must start with <spk:1>; without it `c` was attributed to speaker 0.
        transcript = "<spk:0> a b <spk:1> c <spk:2> d"
        al = [
            WordAlignment("a", 0.00, 0.06, speaker=0),
            WordAlignment("b", 0.08, 0.14, speaker=0),
            WordAlignment("c", 0.16, 0.22, speaker=1),
            WordAlignment("d", 0.24, 0.30, speaker=2),
        ]
        messages = _messages(
            al,
            transcript,
            chunk_size=2,
            num_delay_frames=2,
            audio_duration_secs=0.32,
            write_token="<|write|>",
            prepend=True,
            speaker_tag_placement=placement,
            use_flush_token=True,
            flush_token="<|flush|>",
        )
        flush_at = [m["content"] for m in messages].index("<|flush|>")
        before = [m["content"] for m in messages[:flush_at] if m["role"] == "assistant" and m["content"] != "<blank>"]
        flush_turn = messages[flush_at + 1]["content"]
        assert before == (["<|write|><spk:0> a b"] if placement == "prefix" else ["<|write|> a b <spk:0>"])
        if placement == "prefix":
            assert flush_turn == "<|write|><spk:1> c <spk:2> d"
        else:
            assert flush_turn == "<|write|> c <spk:1> d <spk:2>"
        out = " ".join(before + [flush_turn])
        assert _speaker_texts(out, placement=placement) == _speaker_texts(transcript)

    @pytest.mark.unit
    @pytest.mark.parametrize("chunk_size", [2, 5, 40])
    @pytest.mark.parametrize("case", sorted(_SOT_CASES))
    def test_switch_token_under_prefix(self, case, chunk_size):
        # The switch token used to be read by the suffix branch only, so under prefix it was
        # silently dropped. It now sits directly in front of every identity tag.
        transcript, spec = _SOT_CASES[case]
        al = _align(spec)
        plain = _turns(al, transcript, chunk_size=chunk_size)
        switched = _turns(al, transcript, chunk_size=chunk_size, speaker_switch_token="<spk_switch>")
        out = " ".join(switched)
        assert out.count("<spk_switch>") == len(TAG.findall(out)) > 0
        assert re.findall(r"<spk_switch>(<spk:\d+>)", out) == re.findall(r"<spk:\d+>", out)
        # Removing the switch token gives back the plain prefix target, word for word.
        assert [c.replace("<spk_switch>", "") for c in switched] == plain

    @pytest.mark.unit
    def test_switch_token_under_prefix_exact(self):
        al = _align(_SOT_CASES["two_runs"][1])
        transcript = _SOT_CASES["two_runs"][0]
        kwargs = dict(speaker_switch_token="<spk_switch>", write_token="<|write|>", prepend=True)
        assert _turns(al, transcript, chunk_size=40, **kwargs) == [
            "<|write|><spk_switch><spk:0> a b <spk_switch><spk:1> c d"
        ]
        assert _turns(al, transcript, chunk_size=2, **kwargs) == [
            "<|write|><spk_switch><spk:0> a",
            "<|write|> b",
            "<|write|><spk_switch><spk:1> c",
            "<|write|> d",
        ]


# Prefix placement without flush or switch: the exact targets the builder produced before PF-6.
# Any change here changes the supervision of every prefix recipe.
_PREFIX_PINS = [
    ("two_runs", dict(chunk_size=40), ["<spk:0> a b <spk:1> c d"]),
    ("two_runs", dict(chunk_size=2), ["<spk:0> a", " b", "<spk:1> c", " d"]),
    ("two_runs", dict(chunk_size=5), ["<spk:0> a b", "<spk:1> c d"]),
    ("back_and_forth", dict(chunk_size=40), ["<spk:0> a <spk:1> b <spk:0> c d <spk:2> e"]),
    ("back_and_forth", dict(chunk_size=5, num_delay_frames=3), ["<spk:0> a <spk:1> b", "<spk:0> c d", "<spk:2> e"]),
    ("one_word_runs", dict(chunk_size=2), ["<spk:1> yes", "<spk:0> no", "<spk:1> yes", "<spk:2> maybe"]),
    ("punctuation", dict(chunk_size=40), ["<spk:0> Hello, there. <spk:1> Hi! <spk:0> OK."]),
    (
        "punctuation",
        dict(chunk_size=2, num_delay_frames=6),
        ["<spk:0> Hello,", " there.", "<spk:1> Hi!", "<spk:0> OK."],
    ),
    ("single_speaker", dict(chunk_size=2), ["<spk:0> just", " one", " voice", " here"]),
    # residual fold: the delay pushes the last words past the final boundary
    (
        "back_and_forth",
        dict(chunk_size=2, num_delay_frames=3, audio_duration_secs=0.6),
        ["<spk:0> a", "<spk:1> b <spk:0> c d <spk:2> e"],
    ),
    # dynamic chunking
    ("back_and_forth", dict(chunk_size=0), ["<spk:0> a", "<spk:1> b", "<spk:0> c", " d", "<spk:2> e"]),
    # write token outermost
    (
        "two_runs",
        dict(chunk_size=5, write_token="<|write|>", prepend=True),
        ["<|write|><spk:0> a b", "<|write|><spk:1> c d"],
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize("case,kwargs,expected", _PREFIX_PINS)
def test_prefix_placement_is_unchanged(case, kwargs, expected):
    transcript, spec = _SOT_CASES[case]
    al = _align(spec)
    assert _turns(al, transcript, **kwargs) == expected
    assert _turns(al, transcript, speaker_switch_token=None, use_flush_token=False, **kwargs) == expected
