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
"""Unit tests for context-type gating of metrics and audio discovery in the TTS comparison report tool."""

import json
import math
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Generator, Optional

import matplotlib
import pytest

from scripts.tts_comparison_report.generate_report import _validate_audio_report_benchmarks
from scripts.tts_comparison_report.reporting.components.boxplots import BoxPlotsConfig, prepare_boxplots
from scripts.tts_comparison_report.reporting.components.eval_report import prepare_eval_artifacts
from scripts.tts_comparison_report.reporting.components.metrics_table import (
    prepare_benchmark_metrics_table_rows,
    prepare_summary_metrics_table_rows,
)
from scripts.tts_comparison_report.reporting.components.stat_tests import run_stat_tests
from scripts.tts_comparison_report.reporting.constants import BENCHMARK_META, ContextType
from scripts.tts_comparison_report.reporting.metrics import DistributionMetricsRegistry, MetricsRegistry
from scripts.tts_comparison_report.reporting.models import BenchmarkData, BucketData, BucketStructure
from scripts.tts_comparison_report.reporting.storage import BaseStorage

AUDIO_BENCHMARK = "de_qa"
TEXT_BENCHMARK = "de_qa_ct_text"
CONTEXT_SSIM_NAME = "SSIM (pred vs context)"
GT_SSIM_NAME = "SSIM (pred vs GT)"
NUM_SAMPLES = 8
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True, scope="module")
def _headless_matplotlib():
    matplotlib.use("Agg")


def _values(start: float, step: float) -> list[float]:
    return [round(start + i * step, 4) for i in range(NUM_SAMPLES)]


def _make_benchmark(
    name: str,
    offset: float,
    context_ssim: Optional[list[float]],
    context_ssim_avg: Optional[float],
) -> BenchmarkData:
    """Build an in-memory benchmark with every metric the registries require.

    Args:
        name: Benchmark name, must be a key of ``BENCHMARK_META``.
        offset: Shift applied to the sample values so the two buckets differ.
        context_ssim: Filewise ``pred_context_ssim`` values, or ``None`` to omit the key.
        context_ssim_avg: Aggregated ``ssim_pred_context_avg`` value, or ``None`` to omit the key.
    """
    cer = _values(0.01 + offset, 0.005)
    utmos = _values(3.0 + offset, 0.05)
    filewise_metrics = []

    for i in range(NUM_SAMPLES):
        item = {"pred_audio_filepath": f"predicted_audio_{i}.wav", "cer": cer[i], "utmosv2": utmos[i]}
        if context_ssim is not None:
            item["pred_context_ssim"] = context_ssim[i]
        filewise_metrics.append(item)

    metrics = {
        "wer_cumulative": 0.05 + offset,
        "cer_cumulative": 0.02 + offset,
        "wer_filewise_avg": 0.05 + offset,
        "cer_filewise_avg": 0.02 + offset,
        "utmosv2_avg": 3.2 + offset,
        # QA benchmarks have no ground-truth audio, so the evaluator emits NaN here.
        "ssim_pred_gt_avg": float("nan"),
        "eou_cutoff_rate": 0.0,
        "eou_silence_rate": 0.0,
        "eou_noise_rate": 0.0,
        "eou_error_rate": 0.0,
        "total_gen_audio_seconds": 100.0,
    }
    if context_ssim_avg is not None:
        metrics["ssim_pred_context_avg"] = context_ssim_avg

    return BenchmarkData(name=name, metrics=metrics, filewise_metrics=filewise_metrics)


def _make_bucket(name: str, benchmarks: list[BenchmarkData]) -> BucketData:
    return BucketData(
        name=name,
        path=Path(f"/buckets/{name}"),
        configuration_str="cfg",
        benchmarks={benchmark.name: benchmark for benchmark in benchmarks},
    )


def _audio_benchmark(offset: float) -> BenchmarkData:
    return _make_benchmark(
        AUDIO_BENCHMARK,
        offset,
        context_ssim=_values(0.70 + offset, 0.01),
        context_ssim_avg=0.735 + offset,
    )


def _text_benchmark(offset: float) -> BenchmarkData:
    """Text-context benchmark exactly as the evaluator writes it: context SSIM is NaN everywhere."""
    return _make_benchmark(
        TEXT_BENCHMARK,
        offset,
        context_ssim=[float("nan")] * NUM_SAMPLES,
        context_ssim_avg=float("nan"),
    )


def _make_buckets(*builders) -> tuple[BucketData, BucketData]:
    baseline = _make_bucket("baseline", [build(0.0) for build in builders])
    candidate = _make_bucket("candidate", [build(0.1) for build in builders])
    return baseline, candidate


def _row_names(rows: list[list[str]]) -> list[str]:
    return [row[0] for row in rows]


def _stat_metric_names(results) -> list[str]:
    return [result.metric_name for result in results]


class TestBenchmarkMeta:
    @pytest.mark.unit
    def test_context_ssim_metrics_are_restricted_to_audio_context(self):
        aggregated = {metric.key: metric for metric in MetricsRegistry}
        distribution = {metric.key: metric for metric in DistributionMetricsRegistry}

        assert aggregated["ssim_pred_context_avg"].context_type == ContextType.audio
        assert distribution["pred_context_ssim"].context_type == ContextType.audio
        # Ground-truth SSIM depends on ground-truth audio, not on the context type.
        assert aggregated["ssim_pred_gt_avg"].context_type is None
        assert aggregated["ssim_pred_gt_avg"].optional

    @pytest.mark.unit
    def test_text_context_benchmarks_are_declared_by_name_suffix(self):
        for name, meta in BENCHMARK_META.items():
            expected = ContextType.text if name.endswith("_ct_text") else ContextType.audio
            assert meta.context_type == expected, name
            assert len(meta.lang) == 2, name


class TestBucketContextType:
    @pytest.mark.unit
    def test_benchmark_names_filtered_by_context_type(self):
        baseline, _ = _make_buckets(_audio_benchmark, _text_benchmark)

        assert baseline.get_benchmark_names() == [AUDIO_BENCHMARK, TEXT_BENCHMARK]
        assert baseline.get_benchmark_names(ContextType.audio) == [AUDIO_BENCHMARK]
        assert baseline.get_benchmark_names(ContextType.text) == [TEXT_BENCHMARK]
        assert baseline.get_benchmark_context_type(AUDIO_BENCHMARK) == ContextType.audio
        assert baseline.get_benchmark_context_type(TEXT_BENCHMARK) == ContextType.text

    @pytest.mark.unit
    def test_has_context_type(self):
        baseline, _ = _make_buckets(_audio_benchmark, _text_benchmark)
        text_only, _ = _make_buckets(_text_benchmark)

        assert baseline.has_context_type(None)
        assert baseline.has_context_type(None, TEXT_BENCHMARK)
        assert baseline.has_context_type(ContextType.audio)
        assert baseline.has_context_type(ContextType.audio, AUDIO_BENCHMARK)
        assert not baseline.has_context_type(ContextType.audio, TEXT_BENCHMARK)
        assert not text_only.has_context_type(ContextType.audio)

    @pytest.mark.unit
    def test_unknown_benchmark_raises(self):
        baseline, _ = _make_buckets(_audio_benchmark)

        with pytest.raises(ValueError, match="Unknown benchmark"):
            baseline.get_benchmark_context_type("libritts")

    @pytest.mark.unit
    def test_metric_samples_pooled_over_requested_context_type_only(self):
        baseline, _ = _make_buckets(_audio_benchmark, _text_benchmark)

        pooled_all = baseline.get_metric_samples("cer")
        pooled_audio = baseline.get_metric_samples("cer", context_type=ContextType.audio)

        assert len(pooled_all) == 2 * NUM_SAMPLES
        assert pooled_audio == baseline.get_metric_samples("cer", AUDIO_BENCHMARK)

    @pytest.mark.unit
    def test_metric_samples_reject_mismatched_context_type(self):
        baseline, _ = _make_buckets(_audio_benchmark, _text_benchmark)
        text_only, _ = _make_buckets(_text_benchmark)

        with pytest.raises(ValueError, match="was not generated with 'audio' context"):
            baseline.get_metric_samples("cer", TEXT_BENCHMARK, ContextType.audio)

        with pytest.raises(ValueError, match="No benchmarks with 'audio' context"):
            text_only.get_metric_samples("cer", context_type=ContextType.audio)

    @pytest.mark.unit
    def test_empty_bucket_aggregation_raises_value_error(self):
        empty = _make_bucket("empty", [])

        with pytest.raises(ValueError, match="No benchmarks are available"):
            empty.get_metric_samples("cer")


class TestTextContextBenchmark:
    @pytest.mark.unit
    def test_context_ssim_omitted_even_when_values_are_numeric(self):
        """Gating is declared per benchmark, not inferred from NaN values."""

        def numeric_text_benchmark(offset: float) -> BenchmarkData:
            return _make_benchmark(
                TEXT_BENCHMARK,
                offset,
                context_ssim=_values(0.5 + offset, 0.01),
                context_ssim_avg=0.5 + offset,
            )

        baseline, candidate = _make_buckets(numeric_text_benchmark)

        rows = prepare_benchmark_metrics_table_rows(TEXT_BENCHMARK, baseline, candidate)
        results = run_stat_tests(baseline, candidate, TEXT_BENCHMARK)

        assert CONTEXT_SSIM_NAME not in _row_names(rows)
        assert CONTEXT_SSIM_NAME not in _stat_metric_names(results)

    @pytest.mark.unit
    def test_context_ssim_omitted_for_nan_values(self):
        baseline, candidate = _make_buckets(_text_benchmark)

        rows = prepare_benchmark_metrics_table_rows(TEXT_BENCHMARK, baseline, candidate)
        results = run_stat_tests(baseline, candidate, TEXT_BENCHMARK)

        assert CONTEXT_SSIM_NAME not in _row_names(rows)
        assert GT_SSIM_NAME not in _row_names(rows)
        assert _stat_metric_names(results) == ["CER", "UTMOS v2"]

    @pytest.mark.unit
    def test_text_only_bucket_summary_has_no_context_ssim(self):
        baseline, candidate = _make_buckets(_text_benchmark)

        rows = prepare_summary_metrics_table_rows(baseline, candidate)
        results = run_stat_tests(baseline, candidate)
        image = prepare_boxplots(baseline, candidate, results, BoxPlotsConfig())

        assert CONTEXT_SSIM_NAME not in _row_names(rows)
        assert _stat_metric_names(results) == ["CER", "UTMOS v2"]
        assert image.getvalue().startswith(PNG_SIGNATURE)


class TestAudioContextBenchmark:
    @pytest.mark.unit
    def test_context_ssim_reported(self):
        baseline, candidate = _make_buckets(_audio_benchmark)

        rows = prepare_benchmark_metrics_table_rows(AUDIO_BENCHMARK, baseline, candidate)
        results = run_stat_tests(baseline, candidate, AUDIO_BENCHMARK)
        image = prepare_boxplots(baseline, candidate, results, BoxPlotsConfig(), AUDIO_BENCHMARK)

        assert CONTEXT_SSIM_NAME in _row_names(rows)
        assert _stat_metric_names(results) == ["CER", "UTMOS v2", CONTEXT_SSIM_NAME]
        assert image.getvalue().startswith(PNG_SIGNATURE)

    @pytest.mark.unit
    def test_nan_filewise_context_ssim_is_an_error(self):
        """A broken context-SSIM evaluation on an audio-context benchmark must not be silently dropped."""

        def broken_audio_benchmark(offset: float) -> BenchmarkData:
            return _make_benchmark(
                AUDIO_BENCHMARK,
                offset,
                context_ssim=[float("nan")] * NUM_SAMPLES,
                context_ssim_avg=0.7 + offset,
            )

        baseline, candidate = _make_buckets(broken_audio_benchmark)

        with pytest.raises(ValueError, match="pred_context_ssim.*contains NaN"):
            run_stat_tests(baseline, candidate, AUDIO_BENCHMARK)

        with pytest.raises(ValueError, match="pred_context_ssim.*contains NaN"):
            run_stat_tests(baseline, candidate)

    @pytest.mark.unit
    @pytest.mark.parametrize("context_ssim_avg", [None, float("nan")], ids=["missing", "nan"])
    def test_missing_aggregated_context_ssim_is_an_error(self, context_ssim_avg):
        def broken_audio_benchmark(offset: float) -> BenchmarkData:
            return _make_benchmark(
                AUDIO_BENCHMARK,
                offset,
                context_ssim=_values(0.7 + offset, 0.01),
                context_ssim_avg=context_ssim_avg,
            )

        baseline, candidate = _make_buckets(broken_audio_benchmark)

        with pytest.raises(ValueError, match="ssim_pred_context_avg"):
            prepare_benchmark_metrics_table_rows(AUDIO_BENCHMARK, baseline, candidate)

        with pytest.raises(ValueError, match="ssim_pred_context_avg"):
            prepare_summary_metrics_table_rows(baseline, candidate)

    @pytest.mark.unit
    def test_optional_ground_truth_ssim_still_skipped_when_unavailable(self):
        baseline, candidate = _make_buckets(_audio_benchmark)

        rows = prepare_benchmark_metrics_table_rows(AUDIO_BENCHMARK, baseline, candidate)
        summary_rows = prepare_summary_metrics_table_rows(baseline, candidate)

        assert GT_SSIM_NAME not in _row_names(rows)
        assert GT_SSIM_NAME not in _row_names(summary_rows)


class TestMixedContextBuckets:
    @pytest.mark.unit
    def test_summary_context_ssim_averaged_over_audio_benchmarks_only(self):
        baseline, candidate = _make_buckets(_audio_benchmark, _text_benchmark)

        rows = prepare_summary_metrics_table_rows(baseline, candidate)
        row = next(row for row in rows if row[0] == CONTEXT_SSIM_NAME)

        # Only the audio-context benchmark contributes; its NaN text-context sibling is excluded.
        assert "0.735" in row[1]
        assert "0.835" in row[2]

    @pytest.mark.unit
    def test_pooled_stat_tests_and_plots_use_audio_benchmarks_only(self):
        baseline, candidate = _make_buckets(_audio_benchmark, _text_benchmark)

        # Pooling NaN text-context samples would raise, so a successful run proves they are excluded.
        results = run_stat_tests(baseline, candidate)
        image = prepare_boxplots(baseline, candidate, results, BoxPlotsConfig())

        assert _stat_metric_names(results) == ["CER", "UTMOS v2", CONTEXT_SSIM_NAME]
        assert image.getvalue().startswith(PNG_SIGNATURE)

    @pytest.mark.unit
    def test_eval_artifacts_gate_context_ssim_per_section(self):
        baseline, candidate = _make_buckets(_audio_benchmark, _text_benchmark)

        artifacts = prepare_eval_artifacts(baseline, candidate, BoxPlotsConfig())

        assert CONTEXT_SSIM_NAME in _row_names(artifacts.summary.metrics_table_row)
        assert CONTEXT_SSIM_NAME in _row_names(artifacts.summary.stat_test_table_row)

        audio_result = artifacts.benchmarks[AUDIO_BENCHMARK]
        assert CONTEXT_SSIM_NAME in _row_names(audio_result.metrics_table_row)
        assert CONTEXT_SSIM_NAME in _row_names(audio_result.stat_test_table_row)

        text_result = artifacts.benchmarks[TEXT_BENCHMARK]
        assert CONTEXT_SSIM_NAME not in _row_names(text_result.metrics_table_row)
        assert CONTEXT_SSIM_NAME not in _row_names(text_result.stat_test_table_row)

        for result in [artifacts.summary, audio_result, text_result]:
            assert result.box_plots.getvalue().startswith(PNG_SIGNATURE)

    @pytest.mark.unit
    def test_metrics_without_context_restriction_pool_every_benchmark(self):
        baseline, _ = _make_buckets(_audio_benchmark, _text_benchmark)

        pooled_cer = baseline.get_metric_samples("cer")

        assert len(pooled_cer) == 2 * NUM_SAMPLES
        assert not any(math.isnan(value) for value in pooled_cer)


class _InMemoryStorage(BaseStorage):
    """Minimal storage backend over an in-memory file tree."""

    def __init__(self, files: dict[str, bytes]):
        self._files = {Path(path): content for path, content in files.items()}
        self._dirs = {parent for path in self._files for parent in path.parents}

    def exists(self, path: Path) -> bool:
        return path in self._files or path in self._dirs

    def iter_dir(self, path: Path, only_dirs: bool = False) -> Generator[Path, None, None]:
        children = {p for p in self._files if p.parent == path} | {d for d in self._dirs if d.parent == path}
        for child in sorted(children):
            if only_dirs and child not in self._dirs:
                continue
            yield child

    def open_file(self, path: Path) -> BinaryIO:
        return BytesIO(self._files[path])

    def read_json(self, path: Path) -> Any:
        return json.loads(self._files[path])

    def read_bytes(self, path: Path) -> bytes:
        return self._files[path]


def _bucket_files(root: str, benchmark_dirs: dict[str, bool]) -> dict[str, bytes]:
    """Lay out a results bucket; benchmarks mapped to True also get context and generated audio."""
    files = {}

    for dir_name, with_audio in benchmark_dirs.items():
        benchmark_name = dir_name.split("_", 2)[2]
        base = f"{root}/results/{dir_name}"
        files[f"{base}/{benchmark_name}_metrics_0.json"] = b"{}"
        files[f"{base}/{benchmark_name}_filewise_metrics_0.json"] = b"[]"
        if with_audio:
            files[f"{base}/audio/repeat_0/context_audio_0.wav"] = b"RIFF"
            files[f"{base}/audio/repeat_0/predicted_audio_0.wav"] = b"RIFF"

    return files


class TestAudioReportGating:
    @pytest.mark.unit
    def test_bucket_loading_skips_audio_discovery_for_text_context_benchmarks(self):
        storage = _InMemoryStorage(
            _bucket_files("/buckets/a", {f"cfg_de_{AUDIO_BENCHMARK}": True, f"cfg_de_{TEXT_BENCHMARK}": False})
        )

        bucket = BucketData.from_storage(
            bucket_name="A",
            bucket_path=Path("/buckets/a"),
            bucket_structure=BucketStructure(),
            benchmark_names=(TEXT_BENCHMARK, AUDIO_BENCHMARK),
            check_audio=True,
            storage=storage,
        )

        assert set(bucket.benchmarks) == {AUDIO_BENCHMARK, TEXT_BENCHMARK}
        assert bucket.configuration_str == "cfg"
        assert set(bucket.benchmarks[AUDIO_BENCHMARK].context_audio_paths) == {"context_audio_0"}
        assert set(bucket.benchmarks[AUDIO_BENCHMARK].generated_audio_paths) == {"predicted_audio_0"}
        assert bucket.benchmarks[TEXT_BENCHMARK].context_audio_paths == {}
        assert bucket.benchmarks[TEXT_BENCHMARK].generated_audio_paths == {}

    @pytest.mark.unit
    def test_bucket_loading_still_requires_audio_for_audio_context_benchmarks(self):
        storage = _InMemoryStorage(_bucket_files("/buckets/a", {f"cfg_de_{AUDIO_BENCHMARK}": False}))

        with pytest.raises(FileNotFoundError, match="Missing audio directory"):
            BucketData.from_storage(
                bucket_name="A",
                bucket_path=Path("/buckets/a"),
                bucket_structure=BucketStructure(),
                benchmark_names=(AUDIO_BENCHMARK,),
                check_audio=True,
                storage=storage,
            )

    @pytest.mark.unit
    def test_audio_report_rejects_text_context_benchmarks(self):
        benchmarks = [AUDIO_BENCHMARK, TEXT_BENCHMARK]

        _validate_audio_report_benchmarks(benchmarks, [AUDIO_BENCHMARK])

        with pytest.raises(ValueError, match="text context"):
            _validate_audio_report_benchmarks(benchmarks, [AUDIO_BENCHMARK, TEXT_BENCHMARK])

        with pytest.raises(ValueError, match="not included in evaluation benchmarks"):
            _validate_audio_report_benchmarks([TEXT_BENCHMARK], [AUDIO_BENCHMARK])
