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


@pytest.mark.parametrize('disable_tqdm', [False, True])
def test_trainer_evaluate_before_train(disable_tqdm, tmp_path):
    import torch
    from transformers import Trainer, TrainingArguments

    class TinyModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(2, 2)

        def forward(self, input_ids, labels):
            logits = self.projection(input_ids.float())
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return {'loss': loss, 'logits': logits}

    args = TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, report_to=[], disable_tqdm=disable_tqdm, per_device_eval_batch_size=2)
    dataset = [{'input_ids': [1, 0], 'labels': 0}, {'input_ids': [0, 1], 'labels': 1}]
    trainer = Trainer(model=TinyModel(), args=args, eval_dataset=dataset)
    expected_callback = PrinterCallbackNew if disable_tqdm else ProgressCallbackNew
    assert any(isinstance(callback, expected_callback) for callback in trainer.callback_handler.callbacks)
    metrics = trainer.evaluate()
    assert torch.isfinite(torch.tensor(metrics['eval_loss']))
    assert trainer.state.global_step == 0
