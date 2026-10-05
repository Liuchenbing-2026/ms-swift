"""NPU parameter updates must match CPU AdamW for supplied FP32 gradients."""
import torch
import torch_npu
from cpu_adamw import CPUAdamW


def main():
    torch.manual_seed(42)
    reference = [torch.nn.Parameter(torch.randn(7, 11)) for _ in range(2)]
    device = [torch.nn.Parameter(p.detach().to('npu:0')) for p in reference]
    expected = torch.optim.AdamW([{'params': [reference[0]], 'weight_decay': 0.1},
                                  {'params': [reference[1]], 'weight_decay': 0.0}], lr=1e-3, foreach=False)
    actual = CPUAdamW([{'params': [device[0]], 'weight_decay': 0.1},
                      {'params': [device[1]], 'weight_decay': 0.0}], expected.defaults)
    for step in range(5):
        for left, right in zip(expected.param_groups, actual.param_groups):
            left['lr'] = right['lr'] = 1e-3 / (step + 1)
        for index, (left, right) in enumerate(zip(reference, device)):
            grad = None if step == 2 and index == 1 else torch.randn_like(left)
            left.grad = grad
            right.grad = None if grad is None else grad.to('npu:0')
        expected.step()
        actual.step()
        for left, right in zip(reference, device):
            torch.testing.assert_close(left, right.cpu(), rtol=0, atol=0)
        state = actual.state_dict()
        actual.load_state_dict(state)
        assert all(value.device.type == 'cpu' for values in actual.cpu_optimizer.state.values()
                   for value in values.values() if isinstance(value, torch.Tensor))
    print('PASS: five FP32 NPU updates, parameter groups, LR changes, missing gradients, state roundtrip')


if __name__ == '__main__':
    main()
