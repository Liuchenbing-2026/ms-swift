"""Independent per-source validation for a fixed multitask intervention."""
import argparse
import os
from pathlib import Path

import torch
from swift.callbacks import callbacks_map
from joint_validation import JointValidation, evaluate


def evaluate_sources(model, tokenizer, data_dir, output, prefix):
    for source in ('business', 'tool', 'safety'):
        evaluate(model, tokenizer, Path(data_dir) / ('validation-' + source + '.jsonl'),
                 Path(output) / (prefix + '-' + source + '.json'))


class MultitaskValidation(JointValidation):
    def run(self, args, name):
        evaluate_sources(self.trainer.model, self.trainer.template.tokenizer,
                         os.environ['MULTITASK_DATA'], args.output_dir, name)


callbacks_map['multitask_validation'] = MultitaskValidation


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output);out.mkdir(parents=True, exist_ok=False)
    from swift.model import get_model_processor
    torch.manual_seed(42);torch.npu.set_device(0)
    model, processor = get_model_processor(a.checkpoint, model_type='qwen3_5',
        torch_dtype=torch.bfloat16, device_map='npu:0', attn_impl='sdpa', new_special_tokens=['<decision>'])
    evaluate_sources(model, processor.tokenizer, a.data_dir, out, 'validation')
