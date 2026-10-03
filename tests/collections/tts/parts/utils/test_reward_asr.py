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

import sys
import textwrap

import pytest
import torch

from nemo.collections.tts.parts.utils import reward_asr
from nemo.collections.tts.parts.utils.reward_asr import ProcessRewardASRBackend, RewardASRBackend, RewardASRRouter

pytestmark = pytest.mark.unit

GPU_UUID = "GPU-0f3a6b2c-1234-5678-9abc-def012345678"
MIG_UUID = f"MIG-{GPU_UUID}/1/0"


class FakeBackend(RewardASRBackend):
    def __init__(self, cfg, device_getter):
        del device_getter
        self.name = cfg["name"]
        self.closed = False

    def transcribe(self, audio_paths, languages):
        return [f"{self.name}:{language}:{audio_path}" for audio_path, language in zip(audio_paths, languages)]

    def close(self):
        self.closed = True


def test_router_preserves_order_across_language_routes():
    router = RewardASRRouter(
        {
            "default_backend": "qwen",
            "language_routes": {"hi": "whisper"},
            "backends": {
                "qwen": {"type": "fake", "name": "qwen"},
                "whisper": {"type": "fake", "name": "whisper"},
            },
        },
        device_getter=lambda: torch.device("cpu"),
        backend_types={"fake": FakeBackend},
    )

    assert router.transcribe(["a.wav", "b.wav", "c.wav"], ["en", "hi", "de"]) == [
        "qwen:en:a.wav",
        "whisper:hi:b.wav",
        "qwen:de:c.wav",
    ]
    router.close()
    assert all(backend.closed for backend in router.backends.values())


def test_qwen_backend_requires_qwen_enabled_runtime(tmp_path):
    with pytest.raises(RuntimeError, match="Qwen-enabled training container"):
        ProcessRewardASRBackend(
            {
                "type": "qwen",
                "python_executable": str(tmp_path / "missing-python"),
                "worker_script": str(tmp_path / "worker.py"),
            },
            device_getter=lambda: torch.device("cpu"),
        )


def test_qwen_backend_worker_protocol(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text(
        textwrap.dedent(
            """
            import argparse
            import json
            import sys

            parser = argparse.ArgumentParser()
            parser.add_argument("--backend")
            parser.add_argument("--model")
            parser.add_argument("--device")
            parser.add_argument("--batch-size")
            parser.add_argument("--max-new-tokens")
            parser.add_argument("--language-map")
            args = parser.parse_args()
            print(json.dumps({"status": "ready", "model": args.model, "device": args.device}), flush=True)
            for line in sys.stdin:
                request = json.loads(line)
                if request["command"] == "shutdown":
                    print(json.dumps({"status": "stopped"}), flush=True)
                    break
                transcripts = [
                    f"{language}:{path}"
                    for path, language in zip(request["audio_paths"], request["languages"])
                ]
                print(json.dumps({"status": "ok", "transcripts": transcripts}), flush=True)
            """
        )
    )
    backend = ProcessRewardASRBackend(
        {
            "type": "qwen",
            "python_executable": sys.executable,
            "worker_script": str(worker),
            "timeout_seconds": 10,
        },
        device_getter=lambda: torch.device("cpu"),
    )

    assert backend.transcribe(["a.wav", "b.wav"], ["en", "de"]) == ["en:a.wav", "de:b.wav"]
    backend.close()
    assert backend.process is None


@pytest.mark.parametrize(
    ("mask", "index", "expected"),
    [
        ("2,3", 1, "3"),
        ("2,3", 0, "2"),
        ("2,3", None, "2"),
        (" 2 , 3 ,", 1, "3"),
        (f"{GPU_UUID},{MIG_UUID}", 0, GPU_UUID),
        (f"{GPU_UUID},{MIG_UUID}", 1, MIG_UUID),
        (None, 1, "1"),
        (None, None, "0"),
        ("", 1, "1"),
    ],
)
def test_visible_device_for_worker_resolves_logical_index_through_parent_mask(mask, index, expected):
    environ = {} if mask is None else {"CUDA_VISIBLE_DEVICES": mask}
    device = torch.device("cuda") if index is None else torch.device("cuda", index)

    assert reward_asr._visible_device_for_worker(device, environ) == expected


def test_visible_device_for_worker_rejects_index_outside_parent_mask():
    with pytest.raises(ValueError, match=r"index 1 is outside the parent CUDA_VISIBLE_DEVICES='2'"):
        reward_asr._visible_device_for_worker(torch.device("cuda", 1), {"CUDA_VISIBLE_DEVICES": "2"})


def test_visible_device_for_worker_leaves_non_cuda_devices_untouched():
    assert reward_asr._visible_device_for_worker(torch.device("cpu"), {"CUDA_VISIBLE_DEVICES": "2,3"}) is None
    assert reward_asr._visible_device_for_worker(torch.device("cpu"), {}) is None


def test_process_backend_pins_worker_to_physical_gpu(tmp_path, monkeypatch):
    worker = tmp_path / "worker.py"
    worker.write_text(
        textwrap.dedent(
            """
            import argparse
            import json
            import os
            import sys

            parser = argparse.ArgumentParser()
            parser.add_argument("--backend")
            parser.add_argument("--model")
            parser.add_argument("--device")
            parser.add_argument("--batch-size")
            parser.add_argument("--max-new-tokens")
            parser.add_argument("--language-map")
            args = parser.parse_args()
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            print(json.dumps({"status": "ready", "model": args.model, "device": args.device}), flush=True)
            for line in sys.stdin:
                request = json.loads(line)
                if request["command"] == "shutdown":
                    print(json.dumps({"status": "stopped"}), flush=True)
                    break
                transcripts = [f"{visible_devices}:{path}" for path in request["audio_paths"]]
                print(json.dumps({"status": "ok", "transcripts": transcripts}), flush=True)
            """
        )
    )
    # Rank 1 of a job restricted to physical GPUs 2 and 3 trains on logical cuda:1, i.e. physical GPU 3.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    messages = []
    monkeypatch.setattr(reward_asr.logging, "info", lambda message, *args, **kwargs: messages.append(message))
    backend = ProcessRewardASRBackend(
        {
            "type": "qwen",
            "python_executable": sys.executable,
            "worker_script": str(worker),
            "timeout_seconds": 10,
        },
        device_getter=lambda: torch.device("cuda", 1),
    )

    try:
        assert backend.transcribe(["a.wav"], ["en"]) == ["3:a.wav"]
    finally:
        backend.close()
    ready_messages = [message for message in messages if "ASR worker ready" in message]
    assert len(ready_messages) == 1
    assert "CUDA_VISIBLE_DEVICES=3" in ready_messages[0]
