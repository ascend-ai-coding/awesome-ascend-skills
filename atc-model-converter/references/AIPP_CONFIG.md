# AIPP Configuration Guide

AIPP (AI PreProcessing) handles image preprocessing on the Ascend NPU, improving inference performance by offloading preprocessing from CPU.

## What is AIPP

AIPP performs these operations on NPU:
- Color space conversion (YUV to RGB/BGR)
- Image normalization (mean subtraction, division)
- Image cropping and resizing
- Data type conversion

## Configuration File Format

AIPP config files use protocol buffer text format:

```protobuf
aipp_op {
    aipp_mode: static
    input_format: YUV420SP_U8
    src_image_size_w: 640
    src_image_size_h: 640
    
    # Color space conversion
    csc_switch: true
    matrix_r0c0: 298
    matrix_r0c1: 516
    matrix_r0c2: 0
    matrix_r1c0: 298
    matrix_r1c1: -100
    matrix_r1c2: -208
    matrix_r2c0: 298
    matrix_r2c1: 0
    matrix_r2c2: 409
    
    # Input bias for YUV
    input_bias_0: 16
    input_bias_1: 128
    input_bias_2: 128
    
    # Normalization
    mean_chn_0: 104
    mean_chn_1: 117
    mean_chn_2: 123
    min_chn_0: 0.0
    min_chn_1: 0.0
    min_chn_2: 0.0
    var_reci_chn_0: 1.0
    var_reci_chn_1: 1.0
    var_reci_chn_2: 1.0
}
```

## Common Configurations

### RGB Image Input (uint8 → float16)

```protobuf
aipp_op {
    aipp_mode: static
    input_format: RGB888_U8
    src_image_size_w: 224
    src_image_size_h: 224
    csc_switch: false
    
    # Normalize to [0, 1]
    min_chn_0: 0.0
    min_chn_1: 0.0
    min_chn_2: 0.0
    var_reci_chn_0: 0.003921568627  # 1/255
    var_reci_chn_1: 0.003921568627
    var_reci_chn_2: 0.003921568627
}
```

### BGR Image with ImageNet Normalization

```protobuf
aipp_op {
    aipp_mode: static
    input_format: BGR888_U8
    src_image_size_w: 224
    src_image_size_h: 224
    csc_switch: false
    
    # ImageNet normalization
    mean_chn_0: 104  # B
    mean_chn_1: 117  # G
    mean_chn_2: 123  # R
    min_chn_0: 0.0
    min_chn_1: 0.0
    min_chn_2: 0.0
    var_reci_chn_0: 1.0
    var_reci_chn_1: 1.0
    var_reci_chn_2: 1.0
}
```

### YUV420SP (NV12) Input with CSC

```protobuf
aipp_op {
    aipp_mode: static
    input_format: YUV420SP_U8
    src_image_size_w: 640
    src_image_size_h: 640
    csc_switch: true
    
    # YUV to RGB conversion matrix
    matrix_r0c0: 298
    matrix_r0c1: 516
    matrix_r0c2: 0
    matrix_r1c0: 298
    matrix_r1c1: -100
    matrix_r1c2: -208
    matrix_r2c0: 298
    matrix_r2c1: 0
    matrix_r2c2: 409
    
    input_bias_0: 16
    input_bias_1: 128
    input_bias_2: 128
    
    mean_chn_0: 0.0
    mean_chn_1: 0.0
    mean_chn_2: 0.0
    var_reci_chn_0: 0.003921568627
    var_reci_chn_1: 0.003921568627
    var_reci_chn_2: 0.003921568627
}
```

## Parameter Reference

### Mode Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `aipp_mode` | enum | `static` (compile-time) or `dynamic` (runtime) |
| `input_format` | enum | Input image format |
| `src_image_size_w` | int | Source image width |
| `src_image_size_h` | int | Source image height |

### Input Formats

| Format | Description |
|--------|-------------|
| `YUV420SP_U8` | YUV420 semi-planar (NV12) |
| `YUV422SP_U8` | YUV422 semi-planar |
| `RGB888_U8` | RGB 24-bit |
| `BGR888_U8` | BGR 24-bit |
| `XRGB8888_U8` | XRGB 32-bit |
| `XBGR8888_U8` | XBGR 32-bit |

### Color Space Conversion (CSC)

| Parameter | Type | Description |
|-----------|------|-------------|
| `csc_switch` | bool | Enable CSC |
| `matrix_r0c0-2` | int | Row 0 of CSC matrix |
| `matrix_r1c0-2` | int | Row 1 of CSC matrix |
| `matrix_r2c0-2` | int | Row 2 of CSC matrix |
| `input_bias_0-2` | int | Bias for YUV inputs |
| `output_bias_0-2` | int | Output bias |

### Normalization

| Parameter | Type | Description |
|-----------|------|-------------|
| `mean_chn_0-3` | float | Mean values per channel |
| `min_chn_0-3` | float | Min values per channel |
| `var_reci_chn_0-3` | float | 1/std per channel |

### Cropping and Resizing

| Parameter | Type | Description |
|-----------|------|-------------|
| `crop` | bool | Enable cropping |
| `load_start_pos_w` | int | Crop start X |
| `load_start_pos_h` | int | Crop start Y |
| `crop_size_w` | int | Crop width |
| `crop_size_h` | int | Crop height |
| `resize` | bool | Enable resize |
| `resize_output_w` | int | Output width |
| `resize_output_h` | int | Output height |

### Padding

| Parameter | Type | Description |
|-----------|------|-------------|
| `padding` | bool | Enable padding |
| `left_padding_size` | int | Left padding |
| `right_padding_size` | int | Right padding |
| `top_padding_size` | int | Top padding |
| `bottom_padding_size` | int | Bottom padding |
| `padding_value` | int | Pad value (0-255) |

## Usage with ATC

```bash
# Create AIPP config file
cat > aipp.cfg << 'EOF'
aipp_op {
    aipp_mode: static
    input_format: RGB888_U8
    src_image_size_w: 224
    src_image_size_h: 224
    csc_switch: false
    mean_chn_0: 123.68
    mean_chn_1: 116.78
    mean_chn_2: 103.94
    var_reci_chn_0: 0.017
    var_reci_chn_1: 0.017
    var_reci_chn_2: 0.017
}
EOF

# Convert model with AIPP
atc --model=model.onnx --framework=5 --output=model_aipp \
    --soc_version=Ascend310P3 \
    --insert_op_conf=aipp.cfg
```

## Tips

1. **Format Conversion:** AIPP handles YUV→RGB conversion in hardware, saving CPU cycles
2. **Normalization:** Do normalization in AIPP rather than in application code
3. **Memory Layout:** AIPP outputs NCHW format directly
4. **Performance:** Using AIPP typically improves throughput by 20-30%
5. **Debugging:** Set `--log=debug` to verify AIPP configuration
