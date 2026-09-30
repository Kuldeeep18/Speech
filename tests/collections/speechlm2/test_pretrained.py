# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file

from nemo.collections.speechlm2.parts import pretrained


def test_setup_speech_encoder_hydrates_missing_config_without_weights():
    model = SimpleNamespace(
        cfg=DictConfig(
            {
                "pretrained_asr": "fake-asr",
                "perception": {
                    "target": "nemo.collections.speechlm2.modules.perception.AudioPerceptionModule",
                    "output_dim": 1,
                    "modality_adapter": {"output_dim": 1},
                },
            }
        ),
        llm=SimpleNamespace(config=SimpleNamespace(hidden_size=8)),
    )
    asr_cfg = DictConfig(
        {
            "preprocessor": {"_target_": "fake.Preprocessor"},
            "encoder": {"d_model": 4, "n_layers": 2},
        }
    )

    with (
        patch.object(pretrained, "load_pretrained_nemo_config", return_value=asr_cfg) as load_config,
        patch.object(pretrained, "AudioPerceptionModule") as perception,
    ):
        pretrained.setup_speech_encoder(model, pretrained_weights=False)

    load_config.assert_called_once_with(pretrained.ASRModel, "fake-asr")
    perception.assert_called_once()
    assert model.cfg.perception.preprocessor._target_ == "fake.Preprocessor"
    assert model.cfg.perception.encoder.n_layers == 2
    assert model.cfg.perception.output_dim == 8
    assert model.cfg.perception.modality_adapter.output_dim == 8


@pytest.mark.parametrize(
    ("chunk_size_seconds", "packed_encoder_sequences"),
    [(None, False), (30, False), (30, True)],
)
def test_setup_parallel_expert_encoder_maps_shared_chunk_size(chunk_size_seconds, packed_encoder_sequences):
    pe_encoder_overrides = {
        "speaker_feature_config_version": 1,
        "speaker_feature_mode": "continuous",
        "speaker_activity_threshold": None,
        "diar_normalize_type": "per_feature",
    }
    pe_encoder = SimpleNamespace(
        d_model=4,
        n_spk=8,
        _feat_in=80,
        freeze_asr=False,
        freeze_diar=True,
        spk_kernel_scale=1.0,
        chunk_size_seconds=45.0,
        _bundle_config=DictConfig({"chunk_size_seconds": 45.0}),
        online_inference_enabled=False,
    )
    model = SimpleNamespace(
        cfg=DictConfig(
            {
                "pe_encoder_path": "/tmp/placeholderParallelExpertEncoder.nemo",
                "pe_encoder_overrides": pe_encoder_overrides,
                "encoder_chunk_size_seconds": chunk_size_seconds,
                "packed_encoder_sequences": packed_encoder_sequences,
                "perception": {
                    "preprocessor": {"features": 80, "normalize": "per_feature"},
                    "modality_adapter": {"d_model": 4},
                },
            }
        ),
        perception=SimpleNamespace(
            encoder=SimpleNamespace(d_model=4),
            modality_adapter=object(),
            proj=torch.nn.Linear(4, 8),
            preprocessor=SimpleNamespace(featurizer=SimpleNamespace(normalize="per_feature")),
        ),
    )

    with patch.object(pretrained.ParallelExpertEncoderPT, "load_from_nemo", return_value=pe_encoder) as load:
        pretrained.setup_parallel_expert_encoder(model)

    load.assert_called_once_with(
        "/tmp/placeholderParallelExpertEncoder.nemo",
        map_location="cpu",
        strict=True,
        config_overrides=pe_encoder_overrides,
    )
    assert pe_encoder.chunk_size_seconds == chunk_size_seconds
    assert pe_encoder._bundle_config.chunk_size_seconds == chunk_size_seconds
    assert model.perception.preprocessor.featurizer.normalize is None
    assert model.cfg.perception.preprocessor.normalize is None


@pytest.mark.parametrize(
    ("cfg_update", "match"),
    [
        (
            {
                "encoder_chunk_size_seconds": 30.0,
                "packed_encoder_sequences": True,
                "encoder_chunk_batch_size": 2,
            },
            "encoder_chunk_batch_size is not supported",
        ),
        (
            {"encoder_chunk_size_seconds": -1.0, "packed_encoder_sequences": True},
            "encoder_chunk_size_seconds must be positive or null",
        ),
        (
            {"pe_asr_chunk_size_seconds": 30.0},
            "use model.encoder_chunk_size_seconds",
        ),
    ],
)
def test_setup_parallel_expert_encoder_validates_shared_chunking_config(cfg_update, match):
    pe_encoder = SimpleNamespace(chunk_size_seconds=None)
    cfg = {
        "pe_encoder_path": "/tmp/placeholderParallelExpertEncoder.nemo",
        "perception": {},
    }
    cfg.update(cfg_update)
    model = SimpleNamespace(
        cfg=DictConfig(cfg),
        perception=SimpleNamespace(encoder=object()),
    )

    with (
        patch.object(pretrained.ParallelExpertEncoderPT, "load_from_nemo", return_value=pe_encoder),
        pytest.raises(ValueError, match=match),
    ):
        pretrained.setup_parallel_expert_encoder(model)


def _mock_automodel_loader(config):
    automodel = SimpleNamespace(from_config=MagicMock(return_value=object()), from_pretrained=MagicMock())
    return (
        automodel,
        patch.object(pretrained.AutoConfig, "from_pretrained", return_value=config),
        patch.dict(
            "sys.modules",
            {"nemo_automodel": SimpleNamespace(NeMoAutoModelForCausalLM=automodel)},
        ),
        patch("nemo.collections.speechlm2.parts.automodel_compat.remove_automodel_backend_for_hf_fallback"),
    )


def test_load_pretrained_automodel_llm_builds_missing_mtp_before_loading_weights():
    config = SimpleNamespace(num_nextn_predict_layers=0, name_or_path="base-checkpoint")
    automodel, config_patch, module_patch, compat_patch = _mock_automodel_loader(config)

    with (
        config_patch,
        module_patch,
        compat_patch,
        patch.object(
            pretrained,
            "_resolve_automodel_checkpoint_path",
            return_value="base-checkpoint",
        ),
        patch.object(pretrained, "_load_automodel_base_checkpoint_without_mtp", create=True) as base_load,
    ):
        result = pretrained.load_pretrained_automodel_llm(
            "base-checkpoint",
            pretrained_weights=True,
            dtype=torch.bfloat16,
            mtp_config_overrides={
                "num_nextn_predict_layers": 1,
                "mtp_hybrid_override_pattern": "*",
                "mtp_layers_block_type": None,
            },
        )

    assert result is automodel.from_config.return_value
    assert config.num_nextn_predict_layers == 1
    assert config.mtp_hybrid_override_pattern == "*"
    assert config.mtp_layers_block_type is None
    automodel.from_config.assert_called_once_with(
        config,
        torch_dtype=torch.bfloat16,
        load_base_model=False,
        trust_remote_code=False,
    )
    base_load.assert_called_once_with(result, "base-checkpoint", {})
    automodel.from_pretrained.assert_not_called()


def test_load_pretrained_automodel_llm_preserves_native_mtp_config_by_default():
    config = SimpleNamespace(
        num_nextn_predict_layers=1,
        mtp_hybrid_override_pattern="*E",
        name_or_path="native-mtp-checkpoint",
    )
    automodel, config_patch, module_patch, compat_patch = _mock_automodel_loader(config)

    with (
        config_patch,
        module_patch,
        compat_patch,
        patch.object(
            pretrained,
            "_resolve_automodel_checkpoint_path",
            return_value="native-mtp-checkpoint",
        ),
        patch.object(pretrained, "_load_automodel_base_checkpoint_without_mtp", create=True) as base_load,
    ):
        pretrained.load_pretrained_automodel_llm(
            "native-mtp-checkpoint",
            mtp_config_overrides={
                "num_nextn_predict_layers": 2,
                "mtp_hybrid_override_pattern": "**",
            },
        )

    assert config.num_nextn_predict_layers == 1
    assert config.mtp_hybrid_override_pattern == "*E"
    automodel.from_pretrained.assert_called_once_with(
        "native-mtp-checkpoint",
        torch_dtype=torch.float32,
        trust_remote_code=False,
    )
    automodel.from_config.assert_not_called()
    base_load.assert_not_called()


def test_load_pretrained_automodel_llm_can_replace_native_mtp_config():
    config = SimpleNamespace(
        num_nextn_predict_layers=1,
        mtp_hybrid_override_pattern="*E",
        name_or_path="native-mtp-checkpoint",
    )
    automodel, config_patch, module_patch, compat_patch = _mock_automodel_loader(config)

    with (
        config_patch,
        module_patch,
        compat_patch,
        patch.object(
            pretrained,
            "_resolve_automodel_checkpoint_path",
            return_value="native-mtp-checkpoint",
        ),
        patch.object(pretrained, "_load_automodel_base_checkpoint_without_mtp", create=True) as base_load,
    ):
        result = pretrained.load_pretrained_automodel_llm(
            "native-mtp-checkpoint",
            mtp_config_overrides={
                "num_nextn_predict_layers": 2,
                "mtp_hybrid_override_pattern": "**",
                "mtp_layers_block_type": None,
            },
            replace_mtp_config=True,
        )

    assert config.num_nextn_predict_layers == 2
    assert config.mtp_hybrid_override_pattern == "**"
    assert config.mtp_layers_block_type is None
    automodel.from_config.assert_called_once_with(
        config,
        torch_dtype=torch.float32,
        load_base_model=False,
        trust_remote_code=False,
    )
    base_load.assert_called_once_with(result, "native-mtp-checkpoint", {})


_REPEATED_MTP_OVERRIDES = {
    "num_nextn_predict_layers": 1,
    "mtp_hybrid_override_pattern": "*",
}


@pytest.mark.parametrize(
    "mtp_config_overrides",
    [
        pytest.param(None, id="native-config-only"),
        pytest.param(_REPEATED_MTP_OVERRIDES, id="fallback-config-present"),
    ],
)
def test_load_pretrained_automodel_llm_accepts_one_depth_native_head_as_repeated(
    mtp_config_overrides,
):
    config = SimpleNamespace(
        num_nextn_predict_layers=1,
        mtp_hybrid_override_pattern="*E",
        name_or_path="one-depth-mtp-checkpoint",
    )
    automodel, config_patch, module_patch, compat_patch = _mock_automodel_loader(config)
    loader_kwargs = {
        "num_nextn_predict_layers": 4,
        "mtp_use_repeated_layer": True,
    }
    if mtp_config_overrides is not None:
        loader_kwargs["mtp_config_overrides"] = mtp_config_overrides

    with (
        config_patch as config_loader,
        module_patch,
        compat_patch,
        patch.object(
            pretrained,
            "_resolve_automodel_checkpoint_path",
            return_value="one-depth-mtp-checkpoint",
        ) as resolve_checkpoint,
    ):
        pretrained.load_pretrained_automodel_llm("one-depth-mtp-checkpoint", **loader_kwargs)

    resolve_checkpoint.assert_called_once_with("one-depth-mtp-checkpoint", {})
    config_loader.assert_called_once_with(
        "one-depth-mtp-checkpoint",
        trust_remote_code=False,
        local_files_only=True,
    )
    automodel.from_pretrained.assert_called_once_with(
        "one-depth-mtp-checkpoint",
        torch_dtype=torch.float32,
        trust_remote_code=False,
        num_nextn_predict_layers=4,
        mtp_use_repeated_layer=True,
    )
    automodel.from_config.assert_not_called()


@pytest.mark.parametrize(
    "mtp_config_overrides",
    [
        pytest.param(None, id="native-config-only"),
        pytest.param(_REPEATED_MTP_OVERRIDES, id="fallback-config-present"),
    ],
)
def test_load_pretrained_automodel_llm_rejects_multi_depth_native_head_as_repeated(
    mtp_config_overrides,
):
    config = SimpleNamespace(
        num_nextn_predict_layers=4,
        mtp_hybrid_override_pattern="*E",
        name_or_path="independent-mtp-checkpoint",
    )
    automodel, config_patch, module_patch, compat_patch = _mock_automodel_loader(config)
    loader_kwargs = {
        "num_nextn_predict_layers": 4,
        "mtp_use_repeated_layer": True,
    }
    if mtp_config_overrides is not None:
        loader_kwargs["mtp_config_overrides"] = mtp_config_overrides

    with (
        config_patch,
        module_patch,
        compat_patch,
        patch.object(
            pretrained,
            "_resolve_automodel_checkpoint_path",
            return_value="independent-mtp-checkpoint",
        ),
        pytest.raises(ValueError, match="one physical MTP depth"),
    ):
        pretrained.load_pretrained_automodel_llm("independent-mtp-checkpoint", **loader_kwargs)

    automodel.from_pretrained.assert_not_called()
    automodel.from_config.assert_not_called()


def test_load_pretrained_automodel_llm_builds_repeated_head_for_checkpoint_without_mtp():
    config = SimpleNamespace(num_nextn_predict_layers=0, name_or_path="base-checkpoint")
    automodel, config_patch, module_patch, compat_patch = _mock_automodel_loader(config)

    with (
        config_patch,
        module_patch,
        compat_patch,
        patch.object(
            pretrained,
            "_resolve_automodel_checkpoint_path",
            return_value="base-checkpoint",
        ),
        patch.object(pretrained, "_load_automodel_base_checkpoint_without_mtp", create=True) as base_load,
    ):
        result = pretrained.load_pretrained_automodel_llm(
            "base-checkpoint",
            mtp_config_overrides=_REPEATED_MTP_OVERRIDES,
            num_nextn_predict_layers=4,
            mtp_use_repeated_layer=True,
        )

    assert config.num_nextn_predict_layers == 1
    automodel.from_config.assert_called_once_with(
        config,
        torch_dtype=torch.float32,
        load_base_model=False,
        trust_remote_code=False,
        num_nextn_predict_layers=4,
        mtp_use_repeated_layer=True,
    )
    base_load.assert_called_once_with(
        result,
        "base-checkpoint",
        {"num_nextn_predict_layers": 4, "mtp_use_repeated_layer": True},
    )
    automodel.from_pretrained.assert_not_called()


@pytest.mark.parametrize(
    ("checkpoint_depth", "mtp_config_overrides"),
    [
        pytest.param(0, _REPEATED_MTP_OVERRIDES, id="fresh-head"),
        pytest.param(1, None, id="native-head"),
    ],
)
def test_load_pretrained_automodel_llm_builds_repeated_model_without_checkpoint_weights(
    checkpoint_depth, mtp_config_overrides
):
    config = SimpleNamespace(num_nextn_predict_layers=checkpoint_depth, name_or_path="config-only-checkpoint")
    automodel, config_patch, module_patch, compat_patch = _mock_automodel_loader(config)
    loader_kwargs = {
        "num_nextn_predict_layers": 4,
        "mtp_use_repeated_layer": True,
    }
    if mtp_config_overrides is not None:
        loader_kwargs["mtp_config_overrides"] = mtp_config_overrides

    with (
        config_patch,
        module_patch,
        compat_patch,
        patch.object(
            pretrained,
            "_resolve_automodel_checkpoint_path",
            return_value="config-only-checkpoint",
        ) as resolve_checkpoint,
        patch.object(pretrained, "_load_automodel_base_checkpoint_without_mtp", create=True) as base_load,
    ):
        result = pretrained.load_pretrained_automodel_llm(
            "config-only-checkpoint",
            pretrained_weights=False,
            **loader_kwargs,
        )

    assert result is automodel.from_config.return_value
    assert config.num_nextn_predict_layers == 1
    resolve_checkpoint.assert_called_once_with("config-only-checkpoint", {}, include_weights=False)
    automodel.from_config.assert_called_once_with(
        config,
        torch_dtype=torch.float32,
        load_base_model=False,
        trust_remote_code=False,
        num_nextn_predict_layers=4,
        mtp_use_repeated_layer=True,
    )
    automodel.from_pretrained.assert_not_called()
    base_load.assert_not_called()


def test_load_pretrained_automodel_llm_rejects_repeated_mode_without_head_definition():
    config = SimpleNamespace(num_nextn_predict_layers=0, name_or_path="base-checkpoint")
    automodel, config_patch, module_patch, compat_patch = _mock_automodel_loader(config)

    with (
        config_patch,
        module_patch,
        compat_patch,
        patch.object(
            pretrained,
            "_resolve_automodel_checkpoint_path",
            return_value="base-checkpoint",
        ),
        pytest.raises(ValueError, match="requires either a checkpoint with a native MTP head"),
    ):
        pretrained.load_pretrained_automodel_llm(
            "base-checkpoint",
            num_nextn_predict_layers=4,
            mtp_use_repeated_layer=True,
        )

    automodel.from_pretrained.assert_not_called()
    automodel.from_config.assert_not_called()


def test_load_pretrained_automodel_llm_rejects_replace_without_config_overrides():
    with pytest.raises(ValueError, match="requires mtp_config_overrides"):
        pretrained.load_pretrained_automodel_llm(
            "native-mtp-checkpoint",
            replace_mtp_config=True,
        )


def test_load_pretrained_automodel_llm_forwards_hf_resolution_kwargs():
    config = SimpleNamespace(num_nextn_predict_layers=0, name_or_path="private-checkpoint")
    automodel, config_patch, module_patch, compat_patch = _mock_automodel_loader(config)

    with (
        config_patch as config_loader,
        module_patch,
        compat_patch,
        patch.object(
            pretrained,
            "_resolve_automodel_checkpoint_path",
            return_value="/cache/exact-snapshot/subdir",
            create=True,
        ) as resolve_checkpoint,
        patch.object(pretrained, "_load_automodel_base_checkpoint_without_mtp", create=True) as base_load,
    ):
        result = pretrained.load_pretrained_automodel_llm(
            "private-checkpoint",
            trust_remote_code=True,
            mtp_config_overrides={
                "num_nextn_predict_layers": 1,
                "mtp_hybrid_override_pattern": "*",
            },
            token="secret-token",
            revision="exact-revision",
            cache_dir="/cache",
            local_files_only=True,
            subfolder="subdir",
        )

    config_loader.assert_called_once_with(
        "/cache/exact-snapshot/subdir",
        trust_remote_code=True,
        local_files_only=True,
    )
    resolve_checkpoint.assert_called_once_with(
        "private-checkpoint",
        {
            "token": "secret-token",
            "revision": "exact-revision",
            "cache_dir": "/cache",
            "local_files_only": True,
            "subfolder": "subdir",
        },
    )
    automodel.from_config.assert_called_once_with(
        config,
        torch_dtype=torch.float32,
        load_base_model=False,
        trust_remote_code=True,
        cache_dir="/cache",
    )
    base_load.assert_called_once_with(result, "/cache/exact-snapshot/subdir", {"cache_dir": "/cache"})


@pytest.mark.parametrize(
    ("include_weights", "expected_extra_kwargs"),
    [(True, {}), (False, {"allow_patterns": ["*.json", "*.py"]})],
)
def test_resolve_automodel_checkpoint_path_uses_exact_snapshot(tmp_path, include_weights, expected_extra_kwargs):
    snapshot_root = tmp_path / "snapshot"
    expected_path = snapshot_root / "weights"
    expected_path.mkdir(parents=True)

    with patch("huggingface_hub.snapshot_download", return_value=str(snapshot_root)) as snapshot_download:
        result = pretrained._resolve_automodel_checkpoint_path(
            "private-checkpoint",
            {
                "cache_dir": str(tmp_path / "cache"),
                "local_files_only": True,
                "revision": "exact-revision",
                "subfolder": "weights",
                "token": "secret-token",
            },
            include_weights=include_weights,
        )

    assert result == str(expected_path)
    snapshot_download.assert_called_once_with(
        repo_id="private-checkpoint",
        cache_dir=str(tmp_path / "cache"),
        local_files_only=True,
        revision="exact-revision",
        token="secret-token",
        **expected_extra_kwargs,
    )


def test_automodel_mtp_depth_supports_non_nemotron_config_fields():
    assert (
        pretrained._automodel_config_mtp_depth(
            SimpleNamespace(
                num_nextn_predict_layers=2,
                mtp_hybrid_override_pattern=None,
                mtp_layers_block_type=None,
            )
        )
        == 2
    )
    assert (
        pretrained._automodel_config_mtp_depth(SimpleNamespace(num_nextn_predict_layers=None, mtp_num_hidden_layers=2))
        == 2
    )


def test_automodel_base_load_skips_checkpoint_mtp_for_fresh_head(tmp_path):
    model = SimpleNamespace(config=SimpleNamespace(model_type="nemotron_h"), backbone=object())

    with (
        patch("nemo_automodel.components.checkpoint.checkpointing.Checkpointer") as checkpointer_cls,
        patch.object(torch.cuda, "is_available", return_value=False),
    ):
        pretrained._load_automodel_base_checkpoint_without_mtp(
            model,
            str(tmp_path),
            {},
        )

    checkpoint_config = checkpointer_cls.call_args.args[0]
    assert checkpoint_config.skip_task_head_prefixes_for_base_model == ["mtp."]
    checkpointer_cls.return_value.load_base_model.assert_called_once_with(
        model,
        torch.device("cpu"),
        None,
        str(tmp_path),
        load_base_model=True,
    )
    checkpointer_cls.return_value.load_model.assert_not_called()


@pytest.mark.parametrize("checkpoint_format", ["bin", "safetensors"])
def test_automodel_base_load_keeps_fresh_mtp_on_direct_fast_paths(tmp_path, checkpoint_format):
    class IdentityStateDictAdapter:
        def __init__(self):
            self.loaded_keys = None

        def from_hf(self, state_dict, **_kwargs):
            self.loaded_keys = set(state_dict)
            return state_dict

        def to_hf(self, state_dict, **_kwargs):
            return state_dict

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(2, 2, bias=False)
            self.mtp = torch.nn.Linear(2, 2, bias=False)
            self.config = SimpleNamespace(model_type="tiny_test", tie_word_embeddings=False)
            self.state_dict_adapter = IdentityStateDictAdapter()

    model = TinyModel()
    fresh_mtp = model.mtp.weight.detach().clone()
    checkpoint_state = {
        "base.weight": torch.full_like(model.base.weight, 3.0),
        "mtp.weight": torch.full_like(model.mtp.weight, 9.0),
    }
    if checkpoint_format == "bin":
        torch.save(checkpoint_state, tmp_path / "pytorch_model.bin")
    else:
        # Exercise Automodel's single-device custom-model safetensors branch.
        TinyModel.__module__ = "nemo_automodel.components.models.test"
        save_file(checkpoint_state, tmp_path / "model.safetensors")

    pretrained._load_automodel_base_checkpoint_without_mtp(model, str(tmp_path), {})

    torch.testing.assert_close(model.base.weight, torch.full_like(model.base.weight, 3.0))
    torch.testing.assert_close(model.mtp.weight, fresh_mtp)
    assert model.state_dict_adapter.loaded_keys == {"base.weight"}


def test_exclude_mtp_checkpoint_state_restores_hook_and_adapter_after_error():
    class IdentityStateDictAdapter:
        def from_hf(self, state_dict, **_kwargs):
            return state_dict

    def fail_checkpoint_load():
        raise RuntimeError("checkpoint load failed")

    model = torch.nn.Linear(2, 2)
    model.state_dict_adapter = IdentityStateDictAdapter()

    with pytest.raises(RuntimeError, match="checkpoint load failed"):
        with pretrained._exclude_mtp_checkpoint_state(model):
            assert model._load_state_dict_pre_hooks
            assert "from_hf" in model.state_dict_adapter.__dict__
            fail_checkpoint_load()

    assert not model._load_state_dict_pre_hooks
    assert "from_hf" not in model.state_dict_adapter.__dict__
    assert model.state_dict_adapter.from_hf.__func__ is IdentityStateDictAdapter.from_hf


# ---------------------------------------------------------------------------------------------
# ParallelExpertEncoder mounts: the perception encoder's sync flag reaches both PE branches
# ---------------------------------------------------------------------------------------------
@pytest.fixture
def cpu_default_device():
    """Pin CPU, and restore. Sibling modules set a CUDA default at import and never put it back."""
    previous = torch.get_default_device()
    torch.set_default_device("cpu")
    yield
    torch.set_default_device(previous)


def _pe_mount_model(cfg: dict) -> SimpleNamespace:
    """A model stand-in with just what the PE mounts read: ``cfg`` and a 32-wide perception stack."""
    return SimpleNamespace(
        cfg=DictConfig(cfg),
        perception=SimpleNamespace(
            encoder=SimpleNamespace(d_model=32),
            proj=torch.nn.Linear(32, 8),
            preprocessor=SimpleNamespace(featurizer=SimpleNamespace(normalize="per_feature")),
        ),
    )


def _perception_cfg(sync_flag) -> dict:
    encoder = {} if sync_flag is None else {"sync_max_audio_length": sync_flag}
    return {"encoder": encoder, "preprocessor": {"features": 128, "normalize": "per_feature"}}


def _branch_sync_flags(model) -> tuple:
    encoder = model.perception.encoder
    return encoder.asr_encoder.sync_max_audio_length, encoder.diarization_model.encoder.sync_max_audio_length


# (perception.encoder flag, PE-level flag) -> expected (ASR branch, diarizer encoder) flags. The toy
# branches default to True, so every False below is a value that reached them.
_SYNC_FLAG_CASES = [
    pytest.param(False, None, (False, False), id="perception_encoder"),
    pytest.param(None, False, (False, False), id="pe_level"),
    pytest.param(True, False, (False, False), id="pe_level_wins_over_perception_encoder"),
    pytest.param(None, None, (True, True), id="unset_keeps_branch_defaults"),
]


@pytest.mark.unit
@pytest.mark.parametrize("perception_flag, pe_flag, expected", _SYNC_FLAG_CASES)
def test_two_checkpoint_mount_honours_sync_flag(monkeypatch, cpu_default_device, perception_flag, pe_flag, expected):
    import nemo.collections.asr.modules.parallel_expert_encoder as pe_module
    from tests.collections.asr.test_parallel_expert_encoder import toy_diarization_model_cfg
    from tests.collections.asr.test_parallel_expert_encoder_streaming import (
        build_toy_streaming_pe_encoder,
        streaming_asr_encoder_cfg,
    )

    source = build_toy_streaming_pe_encoder()
    sources = {
        "toy/asr": (
            {"encoder": OmegaConf.to_container(streaming_asr_encoder_cfg())},
            {f"encoder.{k}": v for k, v in source.asr_encoder.state_dict().items()},
        ),
        "toy/diar": (OmegaConf.to_container(toy_diarization_model_cfg()), source.diarization_model.state_dict()),
    }
    monkeypatch.setattr(pe_module, "_resolve_branch_source", lambda name, model_cls, map_location: sources[name])
    pe_cfg = {"asr_model": "toy/asr", "diar_model": "toy/diar", "asr_normalize_type": None}
    if pe_flag is not None:
        pe_cfg["sync_max_audio_length"] = pe_flag
    model = _pe_mount_model({"parallel_expert_encoder": pe_cfg, "perception": _perception_cfg(perception_flag)})

    pretrained.setup_parallel_expert_encoder_from_checkpoints(model)

    assert _branch_sync_flags(model) == expected


def _toy_branch_sources(checkpoint_normalize) -> dict:
    """``_resolve_branch_source`` results for a toy ASR + diarizer pair; the ASR config states
    ``preprocessor.normalize: checkpoint_normalize``."""
    from tests.collections.asr.test_parallel_expert_encoder import toy_diarization_model_cfg
    from tests.collections.asr.test_parallel_expert_encoder_streaming import (
        build_toy_streaming_pe_encoder,
        streaming_asr_encoder_cfg,
    )

    source = build_toy_streaming_pe_encoder()
    asr_cfg = {
        "encoder": OmegaConf.to_container(streaming_asr_encoder_cfg()),
        "preprocessor": {"normalize": checkpoint_normalize},
    }
    return {
        "toy/asr": (asr_cfg, {f"encoder.{k}": v for k, v in source.asr_encoder.state_dict().items()}),
        "toy/diar": (OmegaConf.to_container(toy_diarization_model_cfg()), source.diarization_model.state_dict()),
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "checkpoint_normalize, persisted, effective", [("NA", "NA", None), ("per_feature", "per_feature", "per_feature")]
)
def test_two_checkpoint_mount_persists_resolved_auto_normalization(
    monkeypatch, cpu_default_device, checkpoint_normalize, persisted, effective
):
    """`asr_normalize_type: auto` is resolved once, at mount, and the resolved value replaces `auto`
    in `model.cfg` (HF export) and in the saved hyperparameters (`.ckpt`). A reload therefore
    builds the same encoder even if the ASR checkpoint's preprocessor changes later."""
    import nemo.collections.asr.modules.parallel_expert_encoder as pe_module

    sources = _toy_branch_sources(checkpoint_normalize)
    monkeypatch.setattr(pe_module, "_resolve_branch_source", lambda name, model_cls, map_location: sources[name])
    cfg = {
        "parallel_expert_encoder": {"asr_model": "toy/asr", "diar_model": "toy/diar", "asr_normalize_type": "auto"},
        "perception": _perception_cfg(None),
    }
    caller_cfg = copy.deepcopy(cfg)
    model = _pe_mount_model(cfg)
    # Lightning's `save_hyperparameters()` keeps the caller's own `cfg` object.
    model.hparams = {"cfg": caller_cfg}

    pretrained.setup_parallel_expert_encoder_from_checkpoints(model)

    assert model.perception.encoder.asr_normalize_type == effective
    assert model.cfg.parallel_expert_encoder.asr_normalize_type == persisted
    assert model.hparams["cfg"]["parallel_expert_encoder"]["asr_normalize_type"] == persisted
    assert caller_cfg["parallel_expert_encoder"]["asr_normalize_type"] == "auto", "the caller's config was edited"

    # Reload from the saved hyperparameters after the checkpoint changed its mind.
    sources["toy/asr"][0]["preprocessor"]["normalize"] = "all_features"
    reloaded = _pe_mount_model(model.hparams["cfg"])
    pretrained.setup_parallel_expert_encoder_from_checkpoints(reloaded)
    assert reloaded.perception.encoder.asr_normalize_type == effective


@pytest.mark.unit
def test_two_checkpoint_mount_leaves_an_absent_normalization_key_absent(monkeypatch, cpu_default_device):
    """Compatibility: an absent key still means `per_feature` and is not written back."""
    import nemo.collections.asr.modules.parallel_expert_encoder as pe_module

    sources = _toy_branch_sources("NA")
    monkeypatch.setattr(pe_module, "_resolve_branch_source", lambda name, model_cls, map_location: sources[name])
    model = _pe_mount_model(
        {
            "parallel_expert_encoder": {"asr_model": "toy/asr", "diar_model": "toy/diar"},
            "perception": _perception_cfg(None),
        }
    )

    pretrained.setup_parallel_expert_encoder_from_checkpoints(model)

    assert model.perception.encoder.asr_normalize_type == "per_feature"
    assert "asr_normalize_type" not in model.cfg.parallel_expert_encoder


@pytest.mark.unit
@pytest.mark.parametrize("perception_flag, pe_flag, expected", _SYNC_FLAG_CASES)
def test_bundle_mount_honours_sync_flag(tmp_path, cpu_default_device, perception_flag, pe_flag, expected):
    from tests.collections.asr.test_parallel_expert_encoder import write_toy_bundle

    cfg = {
        "pe_encoder_path": write_toy_bundle(tmp_path / "pe.nemo"),
        "perception": _perception_cfg(perception_flag),
    }
    if pe_flag is not None:
        cfg["pe_encoder_overrides"] = {"sync_max_audio_length": pe_flag}
    model = _pe_mount_model(cfg)

    pretrained.setup_parallel_expert_encoder(model)

    assert _branch_sync_flags(model) == expected


@pytest.mark.unit
def test_parallel_expert_encoder_cfg_keys_only_when_a_bundle_is_mounted():
    """The bundle mount's raw ``model.cfg`` keys are exempt from the dataclass warning only when
    that mount runs; without a bundle, the warning is still the truth for them."""
    keys = pretrained.parallel_expert_encoder_cfg_keys(DictConfig({"pe_encoder_path": "/tmp/pe.nemo"}))
    assert {"pe_encoder_overrides", "encoder_chunk_size_seconds", "spk_kernel_scale"} <= set(keys)
    # An HF export's embedded bundle config is mounted by the same function.
    assert pretrained.parallel_expert_encoder_cfg_keys(DictConfig({"pe_encoder_config": {"target": "x"}})) == keys
    assert pretrained.parallel_expert_encoder_cfg_keys(DictConfig({"pe_encoder_overrides": {}})) == ()


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs, expected",
    [({}, "ParallelExpertEncoder"), ({"streaming": True}, "StreamingParallelExpertEncoder")],
    ids=["default_plain", "streaming"],
)
def test_bundle_mount_class_follows_the_streaming_flag(tmp_path, cpu_default_device, kwargs, expected):
    """StreamingSTT asks for the streaming class; the default (SALM) keeps the plain encoder."""
    from tests.collections.asr.test_parallel_expert_encoder import write_toy_bundle

    model = _pe_mount_model({"pe_encoder_path": write_toy_bundle(tmp_path / "pe.nemo"), "perception": {}})

    pretrained.setup_parallel_expert_encoder(model, **kwargs)

    assert type(model.perception.encoder).__name__ == expected


def _hf_export_config(model) -> dict:
    """``examples/speechlm2/to_hf.py::_hf_export_config``; the script is loaded as ``test_to_hf.py`` does."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "to_hf_for_pretrained_test", Path(__file__).parents[3] / "examples" / "speechlm2" / "to_hf.py"
    )
    to_hf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(to_hf)
    return to_hf._hf_export_config(model, "float32")


@pytest.mark.unit
@pytest.mark.parametrize(
    "kwargs, expected",
    [({}, "ParallelExpertEncoder"), ({"streaming": True}, "StreamingParallelExpertEncoder")],
    ids=["default_plain", "streaming"],
)
def test_embedded_bundle_config_mounts_the_exported_encoder(tmp_path, cpu_default_device, kwargs, expected):
    """An HF export replaces ``pe_encoder_path`` with ``pe_encoder_config``. Mounting that config
    builds the same architecture, with the exported runtime values, as the class the caller asks for;
    the exported weights then load strictly. No bundle file is needed."""
    from tests.collections.asr.test_parallel_expert_encoder import write_toy_bundle

    bundle = write_toy_bundle(tmp_path / "pe.nemo")
    source = _pe_mount_model(
        {
            "pe_encoder_path": bundle,
            "pe_encoder_overrides": {"missing_rttm_target": -2.0},
            "encoder_chunk_size_seconds": 2.0,
            "spk_kernel_scale": 0.5,
            "perception": {},
        }
    )
    pretrained.setup_parallel_expert_encoder(source, **kwargs)
    exported = _hf_export_config(source)
    assert exported["pe_encoder_path"] is None and "pe_encoder_overrides" not in exported
    (tmp_path / "pe.nemo").unlink()

    model = _pe_mount_model({**exported, "perception": {}})
    pretrained.setup_parallel_expert_encoder(model, **kwargs)

    encoder = model.perception.encoder
    assert type(encoder).__name__ == expected
    incompatible = encoder.load_state_dict(source.perception.encoder.state_dict(), strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    assert encoder.chunk_size_seconds == 2.0
    assert encoder.spk_kernel_scale == 0.5
    assert encoder.missing_rttm_target == -2.0
    assert model.perception.preprocessor.featurizer.normalize is None


@pytest.mark.unit
def test_embedded_bundle_config_rejects_a_second_source(tmp_path, cpu_default_device):
    """``pe_encoder_config`` already carries the resolved values: it cannot be combined with a
    bundle path, and ``pe_encoder_overrides`` (which only apply to a bundle path) are refused."""
    from tests.collections.asr.test_parallel_expert_encoder import toy_bundle_config, write_toy_bundle

    inline = OmegaConf.to_container(toy_bundle_config())
    both = _pe_mount_model({"pe_encoder_path": write_toy_bundle(tmp_path / "pe.nemo"), "pe_encoder_config": inline})
    with pytest.raises(ValueError, match="Set only one of model.pe_encoder_path"):
        pretrained.setup_parallel_expert_encoder(both)

    overridden = _pe_mount_model(
        {"pe_encoder_config": inline, "pe_encoder_overrides": {"missing_rttm_target": -2.0}, "perception": {}}
    )
    with pytest.raises(ValueError, match="pe_encoder_overrides applies to model.pe_encoder_path only"):
        pretrained.setup_parallel_expert_encoder(overridden)


@pytest.mark.unit
@pytest.mark.parametrize("key", ["pe_encoder_path", "pe_encoder_config"])
def test_setup_speech_encoder_mounts_either_bundle_source(key):
    """SALM builds its perception through ``setup_speech_encoder``, which must mount the PE for an
    exported ``pe_encoder_config`` as it does for ``pe_encoder_path``."""
    value = "/tmp/pe.nemo" if key == "pe_encoder_path" else {"target": "ParallelExpertEncoderPT"}
    model = SimpleNamespace(
        cfg=DictConfig(
            {
                "pretrained_asr": "fake-asr",
                key: value,
                "perception": {
                    "target": "nemo.collections.speechlm2.modules.perception.AudioPerceptionModule",
                    "preprocessor": {"_target_": "fake.Preprocessor"},
                    "encoder": {"d_model": 4},
                    "output_dim": 1,
                    "modality_adapter": {"output_dim": 1},
                },
            }
        ),
        llm=None,
    )

    with (
        patch.object(pretrained, "AudioPerceptionModule"),
        patch.object(pretrained, "setup_parallel_expert_encoder") as mount,
    ):
        pretrained.setup_speech_encoder(model, pretrained_weights=False)

    mount.assert_called_once_with(model)
