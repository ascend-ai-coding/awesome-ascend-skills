# Domain 6: VAE Patch Parallel

## Overview

VAE decoding is a memory-intensive bottleneck in video generation. The Ascend version introduces spatial patch parallelism that distributes the VAE decode (and I2V encode) across multiple NPUs by slicing the H×W spatial dimensions.

## 6.1 Core Concept

```
Input video latent: [B, C, T, H, W]

Split into grid of patches across NPUs:
┌──────┬──────┐
│ NPU0 │ NPU1 │  h_split=2
├──────┼──────┤  w_split=2
│ NPU2 │ NPU3 │
└──────┴──────┘

Each NPU processes its local patch.
Boundary data exchanged via P2P communication.
```

## 6.2 Parallel_VAE_SP Class

```python
class Parallel_VAE_SP:
    def __init__(self, h_split=1, w_split=1, all_pp_group_ranks=None):
        """
        h_split: Number of splits along height
        w_split: Number of splits along width
        all_pp_group_ranks: List of rank groups for PP communication
        """
        # Create row communication group (same row of patches)
        # Create column communication group (same column of patches)
        self.h_rank = rank // w_split  # Row index in grid
        self.w_rank = rank % w_split   # Column index in grid
```

### Key Methods

- **`patch(x)`**: Split full tensor into local patches
- **`dispatch(local_patch)`**: Reconstruct full tensor from all patches via all_gather
- **`exchange_rows(data, pad_h)`**: Exchange boundary rows with vertical neighbors via P2P
- **`exchange_columns(data, pad_w)`**: Exchange boundary columns with horizontal neighbors via P2P

## 6.3 Monkey-Patching Strategy

The VAE parallel works by temporarily replacing PyTorch functions during decode:

```python
class VAE_patch_parallel:
    """Context manager that patches PyTorch functions for distributed VAE."""

    def __enter__(self):
        # Replace standard functions with distributed versions
        F.conv3d = wraps_f_conv3d
        F.conv2d = wraps_f_conv2d
        F.interpolate = wraps_f_interpolate
        F.pad = wraps_f_pad
        F.scaled_dot_product_attention = wraps_fa
        decoder.forward = wraps_decoder_fw  # Wrap the decoder entry point

    def __exit__(self, ...):
        # Restore original functions
        F.conv3d = original_conv3d
        ...
```

## 6.4 Wrapped Function Details

### `wraps_f_conv3d` — 3D Convolution

```
For each conv3d call:
1. Extract padding from conv3d args
2. Instead of local padding, exchange boundary data with neighbors:
   - exchange_rows() for H-dimension boundaries
   - exchange_columns() for W-dimension boundaries
3. Call original conv3d with padding=(temporal_pad, 0, 0)
```

The H and W padding is replaced by actual data from neighboring patches.

### `wraps_f_conv2d` — 2D Convolution

Two modes based on stride:
- **stride != 1** (downsampling): `dispatch()` to gather full tensor, then split by channels
- **stride == 1** (normal): Exchange boundaries, local convolution

### `wraps_f_interpolate` — Upsampling

- **nearest mode**: Direct local computation (no boundary needed)
- **Other modes**: Exchange boundaries, interpolate, crop to correct size

### `wraps_f_pad` — Padding

Only apply padding on global boundary edges:
```python
# Left padding only if rank is at the left edge of the grid
# Right padding only if rank is at the right edge
# Top padding only if rank is at the top
# Bottom padding only if rank is at the bottom
```

### `wraps_fa` — Scaled Dot-Product Attention in VAE

Before computing attention, `all_gather` to reconstruct full K/V across all patches.

### `wraps_decoder_fw` — Decoder Entry Point

```python
def wraps_decoder_fw(self, x, **kwargs):
    x = parallel_vae.patch(x)        # Split input into patches
    output = original_decoder_fw(x)   # Run decoder on local patch
    output = parallel_vae.dispatch(output)  # Gather results
    return output
```

## 6.5 CausalConv3d Padding Adaptation

**Critical prerequisite:** The VAE's `CausalConv3d` padding must be modified to be compatible with patch parallel.

```python
# Original (all padding via F.pad)
class CausalConv3d(nn.Conv3d):
    def __init__(self, ...):
        self._padding = (self.padding[2], self.padding[2],   # W
                         self.padding[1], self.padding[1],    # H
                         2 * self.padding[0], 0)               # T
        self.padding = (0, 0, 0)  # Remove all conv3d native padding

# Ascend (only T via F.pad, H/W via conv3d native)
class CausalConv3d(nn.Conv3d):
    def __init__(self, ...):
        self._padding = (0, 0,                    # W: handled by conv3d
                         0, 0,                     # H: handled by conv3d
                         2 * self.padding[0], 0)   # T: still via F.pad
        self.padding = (0, self.padding[1], self.padding[2])  # H,W via conv3d
```

**Why:** The `wraps_f_conv3d` wrapper intercepts `F.conv3d` and handles H/W padding through boundary exchange. If H/W padding was done in `F.pad` beforehand, the wrapper cannot intercept it.

## 6.6 Pipeline Integration

### Initialization

```python
# In pipeline __init__:
if use_vae_parallel:
    from .vae_patch_parallel import VAE_patch_parallel, set_vae_patch_parallel

    if dist.get_world_size() < 8:
        # All ranks in one group
        all_pp_group_ranks = [list(range(0, dist.get_world_size()))]
        set_vae_patch_parallel(self.vae.model, dist.get_world_size(), 1,
            all_pp_group_ranks=all_pp_group_ranks, decoder_decode="decoder.forward")
    else:
        # Groups of 8 ranks each
        all_pp_group_ranks = [list(range(8*i, 8*(i+1)))
                              for i in range(dist.get_world_size() // 8)]
        set_vae_patch_parallel(self.vae.model, 4, 2,  # 4×2 grid
            all_pp_group_ranks=all_pp_group_ranks, decoder_decode="decoder.forward")
```

### Usage

```python
# VAE decode (all 3 pipelines)
if self.rank < 8:  # Only first 8 ranks participate in VAE
    with VAE_patch_parallel():
        videos = self.vae.decode(x0)

# VAE encode (I2V pipeline only)
with VAE_patch_parallel():
    y = self.vae.encode([encode_input])[0]
```

**Note:** The decode condition changed from `rank == 0` (Original) to `rank < 8` (Ascend) because all ranks in the VAE parallel group must participate.

## Pitfalls

1. **World size must be power of 2**: Required for clean H×W splitting.
2. **CausalConv3d padding change is required**: Without this, boundary exchange will produce incorrect results for H/W dimensions.
3. **I2V needs both encode and decode wrapping**: Unlike T2V and TI2V which only decode, I2V must also wrap the encoder.
4. **vae2_2.py already has correct padding**: Only `vae2_1.py` needs the CausalConv3d padding fix.
