# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import pytest
import torch
from pathlib import Path


@pytest.fixture(scope='module')
def plugin():
    path = Path(__file__).resolve().parents[2] / 'examples/train/decision/startlux.py'
    spec = importlib.util.spec_from_file_location('startlux_decision_test_plugin', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_projection_and_gradient(plugin):
    torch.manual_seed(1)
    embeddings = torch.nn.Linear(8, 40, bias=False)
    embeddings.requires_grad_(False)
    head = plugin.StartLuxHead(list(range(26)), [1., 2., 3.])
    head.set_output_embeddings(embeddings)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    meta = {
        'record_index': [0, 1],
        'seq_lens': [3, 5],
        'n_options': [2, 4],
        'kinds': [0, 1],
        'counts': [1, 1],
        'max_opt': 4
    }
    actual = head(hidden, None, meta)
    expected = torch.nn.functional.linear(hidden[[0, 1], [2, 4]], embeddings.weight[:26])
    torch.testing.assert_close(actual.logits[0, :2], expected[0, :2])
    torch.testing.assert_close(actual.logits[1], expected[1, :4] / 2)
    actual.logits[actual.option_mask].sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert embeddings.weight.grad is None
    assert all('embedding' not in key for key in head.state_dict())


def test_head_save_restore(plugin, tmp_path):
    head = plugin.StartLuxHead(list(range(26)), [1., 2., 3.])
    head.save_pretrained(tmp_path)
    other = plugin.StartLuxHead(list(reversed(range(26))), [1., 1., 1.])
    assert other.load_pretrained(tmp_path)
    assert torch.equal(head.letter_ids, other.letter_ids)
    assert torch.equal(head.temperatures, other.temperatures)


@pytest.mark.parametrize('temperature', [0., -1., float('nan')])
def test_invalid_temperature(plugin, temperature):
    with pytest.raises(ValueError):
        plugin.StartLuxHead(list(range(26)), [1., 1., temperature])


@pytest.mark.parametrize('state', [{'image': 'a.png'}, ['text', {'image_url': 'a.png'}]])
def test_reject_nested_media(plugin, state):
    assert plugin._contains_startlux_media(state)


def test_decision_payload_survives_standard_column_filter(plugin, monkeypatch):
    import json
    import sys
    from datasets import Dataset
    from types import SimpleNamespace

    from swift.dataset import RowPreprocessor

    def render(state, spec, qid):
        return {'state': state, 'type': 'choice', 'options': [{'id': 'a'}, {'id': 'b'}]}

    monkeypatch.setitem(sys.modules, 'startlux_decision',
                        SimpleNamespace(jevfmt=SimpleNamespace(from_systemone=render)))
    source = {'state': 'evidence', 'questions': {'q': {}}, 'targets': {'q': {'a': 1., 'b': 0.}}}
    rows = plugin.StartLuxPreprocessor().preprocess(source)
    filtered = RowPreprocessor.remove_useless_columns(Dataset.from_list(rows))
    payload = json.loads(filtered[0]['messages'][0]['content'])
    assert payload['state'] == 'evidence'
    assert 'targets' not in payload
    assert filtered[0]['target_probs'] == [[1., 0.]]
