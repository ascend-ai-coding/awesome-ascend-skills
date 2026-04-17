# Domain 9: Inference Pipeline Integration

## Overview

This domain covers the end-to-end changes needed in the inference pipeline (text2video, image2video, textimage2video) to integrate all Ascend adaptations into a working system.

## 9.1 Pipeline Constructor Changes

### New Parameters

All three pipelines gain these constructor parameters:

```python
class WanT2V:
    def __init__(self, config, checkpoint_dir, device_rank=0,
                 t5_fsdp=False, dit_fsdp=False, use_usp=False,
                 t5_cpu=False, use_sp=False,
                 quant_dit_path=None,        # NEW: Quantized model path
                 use_vae_parallel=False,      # NEW: Enable VAE patch parallel
                 ):
```

### Model Loading with dtype

```python
# Original
self.model = WanModel.from_pretrained(checkpoint_dir, subfolder=config.checkpoint)

# Ascend: Specify dtype at load time
self.model = WanModel.from_pretrained(
    checkpoint_dir, subfolder=config.checkpoint,
    torch_dtype=self.param_dtype  # e.g., torch.bfloat16
)
```

### T5 Loading Device

```python
# Original: Always load T5 on CPU
self.text_encoder = T5EncoderModel(..., device=torch.device('cpu'))

# Ascend: Configurable via environment variable
self.text_encoder = T5EncoderModel(
    ...,
    device=torch.device('cpu') if int(os.getenv('T5_LOAD_CPU', 0)) else self.device
)
```

### Model Configuration Commenting

```python
# Original
if not self.init_on_cpu:
    model.to(self.device)

# Ascend: Commented out (device placement handled elsewhere)
# if not self.init_on_cpu:
#     model.to(self.device)
```

### FSDP Instance Attribute

```python
# Ascend: Save dit_fsdp as instance attribute for freqs_list cleanup
self.dit_fsdp = dit_fsdp
```

## 9.2 Quantization Integration

```python
if quant_dit_path:
    quant_dit_path = os.path.abspath(quant_dit_path)

    # For MoE models (T2V, I2V): Quantize both sub-models
    quant_path = os.path.join(quant_dit_path, config.low_noise_checkpoint)
    quant_desc, use_nz = find_quant_config_file(quant_path)
    quantize(model=self.low_noise_model, quant_des_path=quant_desc, use_nz=use_nz)

    quant_path = os.path.join(quant_dit_path, config.high_noise_checkpoint)
    quant_desc, use_nz = find_quant_config_file(quant_path)
    quantize(model=self.high_noise_model, quant_des_path=quant_desc, use_nz=use_nz)

    # For single model (TI2V): Quantize once
    quant_path = os.path.join(quant_dit_path, config.checkpoint)
    quant_desc, use_nz = find_quant_config_file(quant_path)
    quantize(model=self.model, quant_des_path=quant_desc, use_nz=use_nz)
```

## 9.3 VAE Integration

### VAE Constructor dtype

```python
# Original
self.vae = Wan2_1_VAE(..., device=self.device)

# Ascend: Pass dtype
self.vae = Wan2_1_VAE(..., device=self.device, dtype=self.param_dtype)
```

### VAE Parallel Setup

```python
if use_vae_parallel:
    from .vae_patch_parallel import VAE_patch_parallel, set_vae_patch_parallel

    if dist.get_world_size() < 8:
        groups = [list(range(dist.get_world_size()))]
        set_vae_patch_parallel(self.vae.model, dist.get_world_size(), 1,
            all_pp_group_ranks=groups, decoder_decode="decoder.forward")
    else:
        groups = [list(range(8*i, 8*(i+1))) for i in range(dist.get_world_size()//8)]
        set_vae_patch_parallel(self.vae.model, 4, 2,
            all_pp_group_ranks=groups, decoder_decode="decoder.forward")
```

### VAE Decode Condition

```python
# Original: Only rank 0 decodes
if self.rank == 0:
    videos = self.vae.decode(x0)

# Ascend: All ranks in VAE group participate
if self.rank < 8:
    with VAE_patch_parallel():
        videos = self.vae.decode(x0)
```

## 9.4 SP World Size Query

```python
# Original
from .distributed.util import get_world_size
self.sp_size = get_world_size()

# Ascend: Use SP-specific query
from wan.distributed.parallel_mgr import get_sequence_parallel_world_size
self.sp_size = get_sequence_parallel_world_size()
```

## 9.5 CFG Parallel in Sampling Loop

```python
from wan.distributed.parallel_mgr import (
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_cfg_group,
)

# Prepare unified args for CFG parallel
if get_classifier_free_guidance_world_size() == 2:
    arg_all = {
        'context': context if get_classifier_free_guidance_rank() == 0 else context_null,
        'seq_len': seq_len
    }

# In sampling loop
for t_idx, t in enumerate(tqdm(timesteps)):
    timestep = ...
    if get_classifier_free_guidance_world_size() == 2:
        # Single forward per rank
        noise_pred = model(x, t=timestep, **arg_all, t_idx=t_idx)[0]
        noise_pred_cond, noise_pred_uncond = get_cfg_group().all_gather(
            noise_pred, separate_tensors=True)
    else:
        # Standard: Two forwards per step
        noise_pred_cond = model(x, t=timestep, **arg_c, t_idx=t_idx)[0]
        noise_pred_uncond = model(x, t=timestep, **arg_null, t_idx=t_idx)[0]

    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
```

## 9.6 t_idx Passthrough

```python
# Original
for _, t in enumerate(tqdm(timesteps)):
    noise_pred = model(x, t=timestep, **args)[0]

# Ascend: Pass timestep index
for t_idx, t in enumerate(tqdm(timesteps)):
    noise_pred = model(x, t=timestep, **args, t_idx=t_idx)[0]
```

## 9.7 RoPE Cache Lifecycle

After each generation, clear the cached RoPE frequencies:

```python
# At end of generate() method:
if self.dit_fsdp:
    self.low_noise_model._fsdp_wrapped_module.freqs_list = None
    self.high_noise_model._fsdp_wrapped_module.freqs_list = None
else:
    self.low_noise_model.freqs_list = None
    self.high_noise_model.freqs_list = None
```

**Why:** The RoPE cache is resolution-dependent. If the next generation uses a different resolution, stale cache would cause incorrect positional encoding.

## 9.8 Random Number Precision

```python
precision_cpu = int(os.getenv('PRECISION', 0))
gen_device = torch.device("cpu") if precision_cpu else self.device

seed_g = torch.Generator(device=gen_device)
seed_g.manual_seed(seed)
noise = torch.randn(shape, dtype=torch.float32, device=gen_device,
                     generator=seed_g).to(self.device)
```

## 9.9 Warm-Up Generation

In `generate.py`, perform a warm-up generation before timing:

```python
# Warm-up (2 steps, discarded)
videos = wan_t2v.generate(
    prompt, size=(h, w), frame_num=f,
    seed=seed, sample_steps=2  # Only 2 steps for warm-up
)
torch.npu.synchronize()

# Actual generation with timing
start = time.time()
videos = wan_t2v.generate(
    prompt, size=(h, w), frame_num=f,
    seed=seed, sample_steps=actual_steps
)
torch.npu.synchronize()
elapsed = time.time() - start
```

**Why:** NPU operators are compiled/cached on first execution. Warm-up ensures the timed run reflects steady-state performance.

## 9.10 Explicit Model Device Placement

```python
# In generate.py, before generation:
wan_t2v.low_noise_model.to("npu")
wan_t2v.high_noise_model.to("npu")
```

This ensures models are on NPU even when loaded on CPU initially.

## 9.11 generate.py Finalization

```python
# Original
dist.barrier()
dist.destroy_process_group()

# Ascend
from wan.distributed.parallel_mgr import finalize_parallel_env
finalize_parallel_env()
```

## Summary: Pipeline Change Checklist

- [ ] Add `quant_dit_path` and `use_vae_parallel` constructor params
- [ ] Add `torch_dtype` to `from_pretrained()` calls
- [ ] Integrate `mindiesd.quantize()` after model loading
- [ ] Set up VAE patch parallel if enabled
- [ ] Replace SP world_size query with `get_sequence_parallel_world_size()`
- [ ] Import and configure CFG parallel functions
- [ ] Add `t_idx` to all model forward calls
- [ ] Implement CFG parallel branch in sampling loop
- [ ] Change VAE decode condition from `rank == 0` to `rank < 8`
- [ ] Wrap VAE decode/encode with `VAE_patch_parallel()`
- [ ] Add `PRECISION` env var for random number control
- [ ] Add `T5_LOAD_CPU` env var for T5 loading
- [ ] Clear `freqs_list` after generation
- [ ] Save `dit_fsdp` as instance attribute
- [ ] Replace `autocast('cuda')` with `autocast('npu')`
- [ ] Replace device type checks from `'cuda'` to `'npu'`
