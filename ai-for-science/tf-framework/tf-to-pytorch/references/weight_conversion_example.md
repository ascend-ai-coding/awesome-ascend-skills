# TF To PyTorch Weight Conversion Example

This reference file gives a compact example for mapping common TensorFlow weights
into a PyTorch state dict during AI4S model migration.

## Dense Layer

```python
pt_state["encoder.proj.weight"] = torch.from_numpy(tf_kernel.T)
pt_state["encoder.proj.bias"] = torch.from_numpy(tf_bias)
```

## Conv1D Layer

```python
pt_state["conv.weight"] = torch.from_numpy(np.transpose(tf_kernel, (2, 1, 0)))
pt_state["conv.bias"] = torch.from_numpy(tf_bias)
```

## LayerNorm

```python
module = torch.nn.LayerNorm(hidden_dim, eps=1e-3)
module.weight.copy_(torch.from_numpy(tf_gamma))
module.bias.copy_(torch.from_numpy(tf_beta))
```

## CuDNNLSTM

```python
state["lstm.weight_ih_l0"] = torch.from_numpy(tf_kernel.T)
state["lstm.weight_hh_l0"] = torch.from_numpy(tf_recurrent_kernel.T)
state["lstm.bias_ih_l0"] = torch.from_numpy(tf_bias[: 4 * hidden_dim])
state["lstm.bias_hh_l0"] = torch.from_numpy(tf_bias[4 * hidden_dim :])
```

## Validation Pattern

1. Save one or more intermediate outputs from TensorFlow.
2. Run the PyTorch implementation with the same input.
3. Compare arrays with `../scripts/compare_arrays.py`.
4. Fix one layer at a time instead of debugging the full model all at once.
