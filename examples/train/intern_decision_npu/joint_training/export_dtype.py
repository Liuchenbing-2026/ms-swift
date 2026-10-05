"""Convert inference parameters while preserving positional-frequency buffers."""
import torch


def cast_parameters_for_export(model):
    # Module.to also rounds nonpersistent FP32 RoPE buffers. Those are regenerated
    # in FP32 at reload, so rounding them causes a false save/reload mismatch.
    buffers = [(module, name, value) for module in model.modules()
               for name, value in module._buffers.items()]
    model.to(dtype=torch.bfloat16)
    for module, name, value in buffers:
        module._buffers[name] = value
    return model
