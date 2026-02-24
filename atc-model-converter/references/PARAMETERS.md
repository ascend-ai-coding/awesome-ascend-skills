# ATC Parameters Reference

Complete reference for ATC (Ascend Tensor Compiler) command-line parameters.

## Required Parameters

### --model
Input model file path.
- **Type:** String
- **Example:** `--model=resnet50.onnx`

### --framework
Source framework type.
- **Type:** Integer
- **Values:**
  - `0`: Caffe
  - `3`: TensorFlow
  - `5`: ONNX
  - `1`: MindSpore AIR
- **Example:** `--framework=5`

### --output
Output model path (without extension).
- **Type:** String
- **Example:** `--output=resnet50_om`
- **Result:** Generates `resnet50_om.om`

### --soc_version
Target Ascend chip version.
- **Type:** String
- **Values:** Ascend310B4, Ascend310P1, Ascend310P3, Ascend910, Ascend910B, etc.
- **Example:** `--soc_version=Ascend310P3`

## Input/Output Parameters

### --input_shape
Specify input tensor shapes.
- **Type:** String
- **Format:** `"name1:dim1,dim2,dim3;name2:dim1,dim2"`
- **Example:** `--input_shape="input:1,3,224,224"`
- **Dynamic shape:** Use `-1` or `-2` for dynamic dimensions

### --input_format
Input data format.
- **Type:** String
- **Values:** NCHW, NHWC, ND
- **Default:** NCHW
- **Example:** `--input_format=NCHW`

### --output_type
Output data type.
- **Type:** String
- **Values:** FP16, FP32, UINT8, INT8
- **Default:** FP32
- **Example:** `--output_type=FP16`

## Precision Parameters

### --precision_mode
Global precision mode.
- **Type:** String
- **Values:**
  - `force_fp32`: Force FP32 computation
  - `force_fp16`: Force FP16 computation
  - `allow_mix_precision`: Allow mixed precision (default)
  - `allow_fp32_to_fp16`: Allow FP32 to FP16 conversion
  - `must_keep_origin_dtype`: Keep original dtype
- **Example:** `--precision_mode=force_fp16`

### --op_precision_mode
Operator-level precision configuration file.
- **Type:** String
- **Example:** `--op_precision_mode=op_precision.cfg`

### --modify_mixlist
Modify mixed precision list.
- **Type:** String
- **Example:** `--modify_mixlist=mixlist.json`

### --keep_dtype
Keep specific operators' dtype unchanged.
- **Type:** String
- **Format:** Comma-separated operator names
- **Example:** `--keep_dtype=Conv2D,MatMul`

## AIPP Parameters

### --insert_op_conf
AIPP (AI PreProcessing) configuration file.
- **Type:** String
- **Example:** `--insert_op_conf=aipp.cfg`

### --fusion_switch_file
Fusion rule configuration file.
- **Type:** String
- **Example:** `--fusion_switch_file=fusion.cfg`

## Optimization Parameters

### --compression_optimize_conf
Compression and optimization config.
- **Type:** String
- **Example:** `--compression_optimize_conf=compress.cfg`

### --optypelist_for_implmode
Operator implementation mode list.
- **Type:** String
- **Values:** `high_precision`, `high_performance`
- **Example:** `--optypelist_for_implmode=Conv2D:high_performance`

### --op_debug_level
Operator debug level.
- **Type:** Integer
- **Values:** 0-3
- **Default:** 0
- **Example:** `--op_debug_level=1`

## Advanced Parameters

### --log
Log level.
- **Type:** String
- **Values:** debug, info, warning, error, null
- **Default:** null
- **Example:** `--log=info`

### --save_original_model
Save intermediate models.
- **Type:** Boolean (no value needed)
- **Example:** `--save_original_model`

### --dynamic_batch_size
Dynamic batch size support.
- **Type:** String
- **Format:** Comma-separated values
- **Example:** `--dynamic_batch_size="1,2,4,8"`

### --dynamic_image_size
Dynamic image size support.
- **Type:** String
- **Format:** "width1,height1;width2,height2"
- **Example:** `--dynamic_image_size="224,224;448,448"`

### --singleop
Single operator mode.
- **Type:** String (JSON file path)
- **Example:** `--singleop=op_desc.json`

### --buffer_optimize
Buffer optimization level.
- **Type:** String
- **Values:** `l1`, `l2`, `off`
- **Default:** `l2`
- **Example:** `--buffer_optimize=l2`

### --enable_small_channel
Enable small channel optimization.
- **Type:** Integer
- **Values:** 0, 1
- **Default:** 0
- **Example:** `--enable_small_channel=1`

## Debugging Parameters

### --op_select_implmode
Operator implementation selection mode.
- **Type:** String
- **Values:** `high_precision`, `high_performance`
- **Example:** `--op_select_implmode=high_performance`

### --op_compiler_cache_mode
Operator compiler cache mode.
- **Type:** String
- **Values:** `enable`, `disable`, `force`
- **Example:** `--op_compiler_cache_mode=enable`

### --op_compiler_cache_dir
Operator compiler cache directory.
- **Type:** String
- **Example:** `--op_compiler_cache_dir=./cache`

### --debug_dir
Debug directory for intermediate files.
- **Type:** String
- **Example:** `--debug_dir=./debug`

### --op_params
Additional operator parameters.
- **Type:** String
- **Format:** JSON string
- **Example:** `--op_params='{"param1":value1}'`

## Multi-Device Parameters

### --virtual_type
Virtual device type.
- **Type:** Integer
- **Values:** 0, 1
- **Example:** `--virtual_type=0`

### --hccl_config
HCCL configuration file.
- **Type:** String
- **Example:** `--hccl_config=hccl.json`

## Usage Examples

### Basic Conversion
```bash
atc --model=model.onnx --framework=5 --output=out --soc_version=Ascend310P3
```

### With Input Shape
```bash
atc --model=model.onnx --framework=5 --output=out --soc_version=Ascend310P3 \
    --input_shape="input:1,3,224,224"
```

### With AIPP
```bash
atc --model=model.onnx --framework=5 --output=out --soc_version=Ascend310P3 \
    --insert_op_conf=aipp.cfg
```

### FP32 Precision
```bash
atc --model=model.onnx --framework=5 --output=out --soc_version=Ascend310P3 \
    --precision_mode=force_fp32
```

### Dynamic Batch
```bash
atc --model=model.onnx --framework=5 --output=out --soc_version=Ascend310P3 \
    --dynamic_batch_size="1,2,4,8"
```

### Debug Mode
```bash
atc --model=model.onnx --framework=5 --output=out --soc_version=Ascend310P3 \
    --log=debug --op_debug_level=2 --debug_dir=./debug
```
