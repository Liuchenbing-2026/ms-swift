# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
from transformers.trainer_callback import TrainerControl, TrainerState
from types import SimpleNamespace
from unittest.mock import patch

from swift.trainers.patcher import PrinterCallbackNew, ProgressCallbackNew


@pytest.mark.parametrize('callback_cls', [ProgressCallbackNew, PrinterCallbackNew])
def test_evaluation_before_training_does_not_require_training_clock(callback_cls, tmp_path):
    callback = callback_cls()
    state = TrainerState(is_world_process_zero=False)
    logs = {'eval_loss': 0.5}
    callback.on_log(SimpleNamespace(output_dir=str(tmp_path)), state, TrainerControl(), logs=logs)
    assert logs == {'eval_loss': 0.5}


@pytest.mark.parametrize('callback_cls', [ProgressCallbackNew, PrinterCallbackNew])
def test_training_clock_starts_at_resume_step(callback_cls, tmp_path):
    callback = callback_cls()
    state = TrainerState(global_step=10, max_steps=20, is_world_process_zero=False)
    args = SimpleNamespace(output_dir=str(tmp_path))
    control = TrainerControl()
    with patch('swift.trainers.patcher.time.time', return_value=100):
        callback.on_train_begin(args, state, control)
    state.global_step = 12
    logs = {'loss': 0.4}
    with patch('swift.trainers.patcher.time.time', return_value=106):
        callback.on_log(args, state, control, logs=logs)
    assert logs['train_speed(s/it)'] == 3
    assert logs['global_step/max_steps'] == '12/20'
