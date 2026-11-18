"""Independent leaf tensors for unit-test fixtures, not model graph execution."""


def fresh_leaf(values, *, device=None, dtype=None, requires_grad=False):
    """Copy tensor values without sharing storage, grad state, or graph history.

    Importing this helper does not initialize a device. Unsupported requires_grad
    and dtype combinations raise the normal PyTorch error instead of being cast.
    """
    import torch

    if not isinstance(values, torch.Tensor):
        raise TypeError('values must be a torch.Tensor')
    kwargs = {}
    if device is not None:
        kwargs['device'] = device
    if dtype is not None:
        kwargs['dtype'] = dtype
    result = values.detach().to(**kwargs).clone().requires_grad_(requires_grad)
    assert result.is_leaf and result.requires_grad == requires_grad
    assert result.grad is None
    return result
