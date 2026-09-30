# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compressed Shar coverage through NeMo's index builder and dataloader."""

from pathlib import Path

import numpy as np
import pytest
import torch
from lhotse import CutSet
from lhotse.indexing import create_shar_index, index_exists
from lhotse.shar.writers import SharWriter
from lhotse.testing.dummies import DummyManifest
from omegaconf import OmegaConf
from scripts.dataloading import build_indexes

from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config


class _Identity(torch.utils.data.Dataset):
    def __getitem__(self, cuts: CutSet) -> CutSet:
        return cuts


@pytest.fixture
def gzip_shar(tmp_path):
    pytest.importorskip("indexed_gzip")
    root = tmp_path / "shar"
    root.mkdir()
    cuts = DummyManifest(CutSet, begin_id=0, end_id=8, with_data=True)
    for cut in cuts:
        cut.features = None
        cut.custom = None
        cut.supervisions[0].custom = None
    with SharWriter(
        root, fields={"recording": "wav"}, shard_size=3, compress_jsonl=True, create_index=False
    ) as writer:
        for cut in cuts:
            writer.write(cut)
    return root, [cut.id for cut in cuts]


def _fields(root: Path) -> dict[str, list[str]]:
    return {
        "cuts": [str(path) for path in sorted(root.glob("cuts.*.jsonl.gz"))],
        "recording": [str(path) for path in sorted(root.glob("recording.*.tar"))],
    }


@pytest.mark.parametrize("declaration", ["directory", "fields"])
def test_gzip_shar_index_build_and_dataloader(gzip_shar, tmp_path, declaration, monkeypatch):
    root, expected_ids = gzip_shar
    mirror = tmp_path / "mirror"
    shar_path = str(root) if declaration == "directory" else _fields(root)
    entry = {"type": "lhotse_shar", "shar_path": shar_path, "indexed": True}
    jobs = []
    build_indexes.discover(entry, jobs, str(mirror))
    assert len([job for job in jobs if job.kind == build_indexes.JSONL]) == 3
    assert len([job for job in jobs if job.kind == build_indexes.WDS_TAR]) == 3
    for job in jobs:
        build_indexes._build_one(job)
        assert build_indexes._is_indexed(job)
    for path in _fields(root)["cuts"]:
        assert not Path(f"{path}.idx").exists()
        assert index_exists(path, index_path=build_indexes.IndexJob(path, build_indexes.JSONL, str(mirror)).idx_path())
    assert len(list(mirror.rglob("*.gzidx"))) == 3

    def reject_autobuild(*args, **kwargs):
        raise AssertionError("Runtime must use the prebuilt mirror sidecars")

    monkeypatch.setattr("lhotse.shar.readers.indexed.create_jsonl_index", reject_autobuild)
    monkeypatch.setattr("lhotse.shar.readers.indexed.create_tar_index", reject_autobuild)

    config = OmegaConf.create(
        {
            "shar_path": shar_path,
            "indexed": True,
            "indexes_root": str(mirror),
            "force_finite": True,
            "shuffle": False,
            "shard_seed": 0,
            "sample_rate": 16000,
            "batch_size": 2,
            "num_workers": 0,
            "drop_last": False,
        }
    )
    cuts, is_tarred = read_cutset_from_config(config)
    assert is_tarred
    assert cuts.is_indexed
    assert sorted(cut.id for cut in cuts) == sorted(expected_ids)
    loader = get_lhotse_dataloader_from_config(config=config, global_rank=0, world_size=1, dataset=_Identity())
    batch = next(iter(loader))
    assert len(batch) == 2
    assert isinstance(batch[0].load_audio(), np.ndarray)

    gzip_job = next(job for job in jobs if job.kind == build_indexes.JSONL)
    seek_index = Path(str(gzip_job.idx_path())[:-4] + ".gzidx")
    seek_index.unlink()
    assert not build_indexes._is_indexed(gzip_job)
    build_indexes._build_one(gzip_job, force=True)
    assert seek_index.is_file()
    assert build_indexes._is_indexed(gzip_job)


def test_gzip_shar_metadata_only_excludes_sidecars(gzip_shar):
    root, expected_ids = gzip_shar
    create_shar_index(root)
    assert list(root.glob("cuts.*.jsonl.gz.idx"))
    assert list(root.glob("cuts.*.jsonl.gz.gzidx"))
    config = OmegaConf.create(
        {"shar_path": str(root), "indexed": True, "metadata_only": True, "force_finite": True, "shard_seed": 0}
    )
    cuts, is_tarred = read_cutset_from_config(config)
    assert is_tarred
    assert cuts.is_indexed
    assert sorted(cut.id for cut in cuts) == sorted(expected_ids)


def test_gzip_shar_exact_restore_through_nemo_config(gzip_shar):
    root, expected_ids = gzip_shar
    create_shar_index(root)
    config = OmegaConf.create({"shar_path": str(root), "indexed": True, "force_finite": True, "shard_seed": 23})
    full, _ = read_cutset_from_config(config)
    expected_order = [cut.id for cut in full]
    assert sorted(expected_order) == sorted(expected_ids)

    interrupted, _ = read_cutset_from_config(config)
    stream = iter(interrupted)
    consumed = [next(stream).id for _ in range(5)]
    state = interrupted.state_dict()
    restored, _ = read_cutset_from_config(config)
    restored.load_state_dict(state)
    assert consumed + [cut.id for cut in restored] == expected_order
