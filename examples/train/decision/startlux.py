# Copyright (c) ModelScope Contributors. All rights reserved.
"""StartLux checkpoint support through the standard decision task interfaces."""
import json
import math
import os
import torch
import weakref
from pathlib import Path
from torch.nn import functional as F

from swift.dataset import DatasetMeta, RowPreprocessor, register_dataset
from swift.model import Model, ModelGroup, ModelMeta, register_model
from swift.model.decision_head import QUESTION_TYPES, ScoringHead, ScoringOutput, pad_option_logits
from swift.model.models.decision import ScoringModelLoader
from swift.model.models.qwen import Qwen3_5Loader, Qwen3_5MoeLoader
from swift.template import Template, TemplateMeta, register_template


class StartLuxHead(ScoringHead):
    """Last-position FP32 vocabulary projection restricted to A-Z candidates."""

    needs_output_embeddings = True
    needs_lm_logits = False
    keep_fp32 = True

    def __init__(self, letter_ids, temperatures):
        super().__init__()
        if len(letter_ids) != 26 or len(set(letter_ids)) != 26:
            raise ValueError('Expected 26 distinct candidate token IDs')
        if any(not math.isfinite(t) or t <= 0 for t in temperatures):
            raise ValueError('Decision temperatures must be finite and positive')
        self.register_buffer('letter_ids', torch.tensor(letter_ids, dtype=torch.long))
        self.register_buffer('temperatures', torch.tensor(temperatures, dtype=torch.float32))

    def set_output_embeddings(self, embeddings):
        # The backbone owns the vocabulary head; it must stay outside adapter head saves.
        self._embedding_ref = weakref.ref(embeddings)

    @classmethod
    def from_pretrained(cls, model_dir, model=None, config=None):
        data = json.loads((Path(model_dir) / 'decision_config.json').read_text())
        return cls(data['letter_token_ids'], [data['temperature_by_type'][k] for k in QUESTION_TYPES])

    def extra_state_for_save(self):
        return {'letter_token_ids': self.letter_ids.tolist(), 'temperatures': self.temperatures.tolist()}

    def forward(self, hidden_states, lm_logits, meta):
        if meta is None:
            raise ValueError('StartLux requires decision metadata')
        counts = meta['n_options']
        if any(not 2 <= n <= 26 for n in counts):
            raise ValueError('Use an explicit wide-choice strategy for more than 26 options')
        rows = torch.tensor(meta['record_index'], device=hidden_states.device)
        columns = [
            hidden_states.shape[1] - 1 if meta.get('left_padding') else meta['seq_lens'][r] - 1
            for r in meta['record_index']
        ]
        columns = torch.tensor(columns, device=hidden_states.device)
        embeddings = self._embedding_ref()
        if embeddings is None:
            raise RuntimeError('Backbone vocabulary head is unavailable')
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            weight = embeddings.weight.index_select(0, self.letter_ids.to(embeddings.weight.device)).float()
            scores = F.linear(hidden_states[rows, columns].float(), weight)
            per_question = [scores[i, :n] / self.temperatures[int(meta['kinds'][i])] for i, n in enumerate(counts)]
        logits, mask = pad_option_logits(per_question, width=meta.get('max_opt'))
        return ScoringOutput(
            logits=logits,
            option_mask=mask,
            kinds=torch.tensor(meta['kinds'], device=logits.device),
            counts=torch.tensor(meta['counts'], device=logits.device))


class StartLuxLoader(ScoringModelLoader, Qwen3_5Loader):
    head_cls = StartLuxHead

    def get_model(self, model_dir, config, processor, model_kwargs):
        from startlux_decision import jevfmt
        tokenizer = getattr(processor, 'tokenizer', processor)
        data = json.loads((Path(model_dir) / 'decision_config.json').read_text())
        if jevfmt.check_tokenizer(tokenizer) != data['letter_token_ids']:
            raise ValueError('Checkpoint verbalizer does not match the tokenizer')
        from swift.model.decision_head import maybe_load_trained_scoring_head
        loader = Qwen3_5MoeLoader if config.model_type == 'qwen3_5_moe' else Qwen3_5Loader
        if config.model_type not in ('qwen3_5', 'qwen3_5_moe'):
            raise ValueError('Expected a Qwen3.5 dense or MoE checkpoint')
        model = loader.get_model(self, model_dir, config, processor, model_kwargs)
        self.attach_scoring_head(model, self.build_head(model, model_dir, config))
        maybe_load_trained_scoring_head(model, model_dir)
        return model


def _contains_startlux_media(value):
    if isinstance(value, dict):
        return bool(set(value) & {'image', 'images', 'image_url', 'video', 'videos'}) or any(
            _contains_startlux_media(v) for v in value.values())
    if isinstance(value, list):
        return any(_contains_startlux_media(v) for v in value)
    return False


class StartLuxPreprocessor(RowPreprocessor):

    def preprocess(self, record):
        from startlux_decision import jevfmt
        if record.get('images') or record.get('videos') or _contains_startlux_media(record.get('state')):
            raise ValueError('This StartLux training example accepts text records')
        rows = []
        for qid, spec in record['questions'].items():
            row = jevfmt.from_systemone(record['state'], spec, qid=qid)
            order = [o['id'] for o in row['options']]
            if not 2 <= len(order) <= 26:
                raise ValueError('Expected 2-26 legal candidates')
            result = {'messages': [{'role': 'user', 'content': json.dumps(row, ensure_ascii=False)}]}
            if 'targets' in record:
                target = record['targets'][qid]
                if set(target) - set(order):
                    raise ValueError('Unknown target option')
                values = [float(target.get(key, 0)) for key in order]
                if any(not math.isfinite(v) or v < 0 for v in values) or not math.isclose(
                        sum(values), 1., abs_tol=1e-6):
                    raise ValueError('Expected a finite normalized target distribution')
                result['target_probs'] = [values]
            rows.append(result)
        if not rows:
            raise ValueError('Record contains no questions')
        return rows


class StartLuxTemplate(Template):

    def _encode(self, inputs):
        from startlux_decision import jevfmt
        row = json.loads(inputs.messages[0]['content'])
        order = [o['id'] for o in row['options']]
        ids, rendered_order = jevfmt.render_ids(row, self.tokenizer, order, max_length=self.max_length or 8192)
        if rendered_order != order:
            raise ValueError('Renderer changed option ordering')
        return {'input_ids': ids, 'decision_meta': {'n_options': [len(order)], 'kinds': [QUESTION_TYPES[row['type']]]}}


register_template(
    TemplateMeta('startlux_decision', prefix=[], prompt=['{{QUERY}}'], chat_sep=None, template_cls=StartLuxTemplate))
register_model(
    ModelMeta(
        'startlux_decision', [
            ModelGroup([
                Model('StartLuxAI/StartLux-Decision-4B', 'StartLuxAI/StartLux-Decision-4B'),
                Model('StartLuxAI/StartLux-Decision-35B-A3B', 'StartLuxAI/StartLux-Decision-35B-A3B'),
            ])
        ],
        StartLuxLoader,
        template='startlux_decision',
        model_arch='qwen2_vl',
        task_type='decision',
        is_multimodal=True,
        requires=['transformers>=5.2.0', 'qwen_vl_utils>=0.0.14'],
        additional_saved_files=['decision_config.json']))
for name, variable in [('startlux_decision', 'STARTLUX_TRAIN_FILE'), ('startlux_decision_eval', 'STARTLUX_EVAL_FILE')]:
    if os.environ.get(variable):
        register_dataset(
            DatasetMeta(dataset_name=name, dataset_path=os.environ[variable], preprocess_func=StartLuxPreprocessor()))
