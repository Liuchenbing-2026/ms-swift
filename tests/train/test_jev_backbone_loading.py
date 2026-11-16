"""Native checkpoint-to-scoring-head regression with generated tiny CPU weights."""
import json
import pytest
import torch
from transformers import (Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5MoeConfig,
                          Qwen3_5MoeForConditionalGeneration)

from swift.model.decision_head import JevVerbalizerHead
from swift.model.model_meta import MODEL_MAPPING, ModelInfo
from swift.model.models.decision import JevModelLoader


@pytest.mark.parametrize('moe', [False, True])
def test_jev_loads_checkpoint_and_scoring_head(moe, tmp_path):
    text = dict(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=['full_attention'],
        max_position_embeddings=64)
    if moe:
        text.update(num_experts=2, num_experts_per_tok=1, moe_intermediate_size=16, shared_expert_intermediate_size=16)
    else:
        text['intermediate_size'] = 64
    vision = dict(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        out_hidden_size=32,
        patch_size=2,
        temporal_patch_size=1,
        spatial_merge_size=1,
        num_position_embeddings=16)
    config_cls = Qwen3_5MoeConfig if moe else Qwen3_5Config
    model_cls = Qwen3_5MoeForConditionalGeneration if moe else Qwen3_5ForConditionalGeneration
    config = config_cls(text_config=text, vision_config=vision, tie_word_embeddings=False)
    config._attn_implementation = 'eager'
    model_cls(config).save_pretrained(tmp_path)
    (tmp_path / 'decision_head.json').write_text(json.dumps({
        'bias': [0.] * 24,
        'verbalizer_ids': list(range(24)),
    }))
    info = ModelInfo(
        model_type='jev',
        model_dir=str(tmp_path),
        torch_dtype=torch.float32,
        max_model_len=64,
        quant_method=None,
        quant_bits=None,
        is_moe_model=moe,
        is_multimodal=True,
        config=config,
        task_type='decision')
    loader = JevModelLoader(info, MODEL_MAPPING['jev'], model_kwargs={})
    model = loader.get_model(str(tmp_path), config, None, {'dtype': torch.float32})
    assert isinstance(model, model_cls)
    assert isinstance(model.scoring_head, JevVerbalizerHead)
    output = model(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.ones(1, 3),
        decision_meta={
            'kinds': [0],
            'n_options': [2],
            'record_index': [0],
            'seq_lens': [3],
            'counts': [1]
        })
    assert output.logits.shape == (1, 2)
    assert torch.isfinite(output.logits).all()
