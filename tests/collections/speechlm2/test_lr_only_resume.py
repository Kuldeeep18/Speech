# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate historical LR-only continuation without discarding optimizer state."""
import copy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig

from nemo.collections.speechlm2.parts.lr_only_resume import ScaleRestoredLearningRate
from nemo.core.classes.common import _is_target_allowed, safe_instantiate
from nemo.core.optim.lr_scheduler import CosineAnnealing


def _restored_trainer():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    scheduler = CosineAnnealing(optimizer, max_steps=100, min_lr=1e-5)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()
    trainer = SimpleNamespace(
        global_step=50,
        ckpt_path='/checkpoints/step=50.ckpt',
        optimizers=[optimizer],
        lr_scheduler_configs=[SimpleNamespace(scheduler=scheduler)],
        is_global_zero=False,
    )
    return trainer, optimizer, scheduler


def test_scale_restored_lr_preserves_moments_and_schedule_position_once():
    trainer, optimizer, scheduler = _restored_trainer()
    before_state = copy.deepcopy(optimizer.state_dict()['state'])
    old_lr = optimizer.param_groups[0]['lr']
    old_last_lr = scheduler._last_lr[:]
    old_epoch = scheduler.last_epoch
    callback = ScaleRestoredLearningRate(0.75, 50, 1e-3, 1e-5)
    callback.on_train_start(trainer, None)

    torch.testing.assert_close(optimizer.state_dict()['state'], before_state)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(old_lr * 0.75)
    assert optimizer.param_groups[0]['initial_lr'] == pytest.approx(0.00075)
    assert scheduler.base_lrs == pytest.approx([0.00075])
    assert scheduler.min_lr == pytest.approx(0.0000075)
    assert scheduler._last_lr == pytest.approx([lr * 0.75 for lr in old_last_lr])
    assert scheduler.last_epoch == old_epoch

    restored = ScaleRestoredLearningRate(0.75, 50, 1e-3, 1e-5)
    restored.load_state_dict(callback.state_dict())
    trainer.global_step = 51
    restored.on_train_start(trainer, None)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(old_lr * 0.75)
    assert restored.state_key == callback.state_key


@pytest.mark.parametrize('wrong', ['step', 'checkpoint', 'base_lr', 'minimum_lr'])
def test_scale_restored_lr_rejects_wrong_source_before_mutating(wrong):
    trainer, optimizer, scheduler = _restored_trainer()
    if wrong == 'step':
        trainer.global_step = 49
    elif wrong == 'checkpoint':
        trainer.ckpt_path = '/checkpoints/step=50-last.ckpt'
    elif wrong == 'base_lr':
        scheduler.base_lrs = [2e-3]
    else:
        scheduler.min_lr = 2e-5
    old_lr = optimizer.param_groups[0]['lr']
    callback = ScaleRestoredLearningRate(0.75, 50, 1e-3, 1e-5)
    with pytest.raises(RuntimeError):
        callback.on_train_start(trainer, None)
    assert optimizer.param_groups[0]['lr'] == old_lr
    assert callback.state_dict() == {'applied': False}


def test_historical_callback_is_safely_instantiable():
    target = 'nemo.collections.speechlm2.parts.lr_only_resume.ScaleRestoredLearningRate'
    assert _is_target_allowed(target)
    callback = safe_instantiate(
        DictConfig(
            {
                '_target_': target,
                'factor': 0.75,
                'source_step': 50,
                'original_base_lr': 1e-3,
                'original_min_lr': 1e-5,
            }
        )
    )
    assert isinstance(callback, ScaleRestoredLearningRate)
