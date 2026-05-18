# MLLM JSON Data Format Specification

MindSpeed-MM VLM training uses the **MLLM JSON format** as the unified data input format. This document provides a detailed specification of the format, field definitions, and conversion methods.

## Format Overview

MLLM JSON is a JSON file containing an array of objects, where each object represents a single training sample (single-turn or multi-turn conversation).

## Basic Structure

The MLLM format uses `messages` (with `role`/`content`) and `images` fields. This is the format output by the conversion script and expected by MindSpeed-MM training.

```json
[
    {
        "images": ["path/to/image.jpg"],
        "messages": [
            {
                "role": "user",
                "content": "<image>\nDescribe the contents of this image."
            },
            {
                "role": "assistant",
                "content": "This image shows a cat sitting on a sofa."
            }
        ]
    }
]
```

> **Important**: The field names (`messages`, `images`, `role`, `content`) must match the `attr` section in `data.json`. The default `data_3b.json` maps `"messages": "messages"` and `"images": "images"`. Using different field names (e.g., `conversations` instead of `messages`) will cause `KeyError` at runtime.

## Field Descriptions

### Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `images` | array[string] | Conditional | List of image file paths (relative to `dataset_dir`) |
| `messages` | array | Yes | List of conversation turns |
| `videos` | array[string] | Conditional | List of video file paths (if the model supports video input) |

> At least one of `images` or `videos` should be present (plain text conversations may omit both, but VLM training typically requires visual input).

### messages Field

Each conversation turn contains:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | string | Yes | Role identifier: `"user"` or `"assistant"` |
| `content` | string | Yes | Conversation content |

#### Special Tokens

| Token | Description |
|-------|-------------|
| `<image>` | Image placeholder; marks the image insertion position within `content` |
| `<video>` | Video placeholder (if applicable) |

> The number of `<image>` tokens must match the length of the `images` array. In multi-image scenarios, each `<image>` corresponds in order to the paths in the `images` array.

## Examples

### Single Image, Single-Turn Conversation

```json
{
    "images": ["COCO2017/train2017/000000000009.jpg"],
    "messages": [
        {
            "role": "user",
            "content": "<image>\nWhat is in this image?"
        },
        {
            "role": "assistant",
            "content": "The image shows a golden Labrador Retriever running on a grassy field. The background features a blue sky with white clouds."
        }
    ]
}
```

### Single Image, Multi-Turn Conversation

```json
{
    "images": ["COCO2017/train2017/000000000025.jpg"],
    "messages": [
        {
            "role": "user",
            "content": "<image>\nPlease describe this image."
        },
        {
            "role": "assistant",
            "content": "This is an indoor photo showing a modern-style living room."
        },
        {
            "role": "user",
            "content": "How many chairs are in the living room?"
        },
        {
            "role": "assistant",
            "content": "There are two chairs in the living room: a gray armchair and a white dining chair."
        }
    ]
}
```

### Multi-Image Conversation

```json
{
    "images": [
        "COCO2017/train2017/000000000030.jpg",
        "COCO2017/train2017/000000000036.jpg"
    ],
    "messages": [
        {
            "role": "user",
            "content": "<image>\n<image>\nPlease compare the similarities and differences between these two images."
        },
        {
            "role": "assistant",
            "content": "The first image is a daytime city street scene, and the second is a nighttime view of the same location. Both images show the same street, but the lighting and atmosphere are completely different."
        }
    ]
}
```

### Plain Text Conversation (No Images)

```json
{
    "messages": [
        {
            "role": "user",
            "content": "Please explain what a convolutional neural network is."
        },
        {
            "role": "assistant",
            "content": "A convolutional neural network (CNN) is a type of deep learning model..."
        }
    ]
}
```

## Image Path Resolution

The image path concatenation rule:

```
Final path = dataset_dir from data.json + image path from MLLM JSON
```

Example:
- `dataset_dir`: `"./data"`
- `image`: `["COCO2017/train2017/000000000009.jpg"]`
- Final path: `./data/COCO2017/train2017/000000000009.jpg`

## Converting from LLaVA Format

The conversion script transforms LLaVA-Instruct-150K format into MLLM format:

| Field | LLaVA Format | MLLM Format |
|-------|-------------|-------------|
| Image | `"image": "000000033471.jpg"` (string) | `"images": ["./data/COCO2017/train2017/000000033471.jpg"]` (array with full path) |
| Conversations | `"conversations": [{"from": "human", "value": ...}]` | `"messages": [{"role": "user", "content": ...}]` |
| Role names | `"human"` / `"gpt"` | `"user"` / `"assistant"` |

Conversion script:

```bash
# Must run from MindSpeed-MM root — script uses hardcoded ./data/ paths
cd /root/workspace/MindSpeed-MM
python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py
```

> **Note**: The script reads from `./data/llava_instruct_150k.json` and checks image existence under `./data/COCO2017/train2017/`. Ensure data is at these paths or create symlinks before running.

## Custom Dataset Preparation

If using your own dataset, follow these steps:

1. **Organize image files**: Place them in a unified directory and ensure paths are accessible
2. **Write annotation JSON**: Create conversation annotations in MLLM format
3. **Validate format**: Ensure the number of `<image>` tokens matches the length of the `images` array
4. **Configure data.json**: Set the correct `dataset_dir` and `dataset` paths

### Format Validation Checklist

- [ ] JSON syntax is valid (verify with `python -m json.tool`)
- [ ] Field names match `attr` config: `messages` (not `conversations`), `images` (not `image`)
- [ ] Each turn in `messages` uses `role`/`content` (not `from`/`value`)
- [ ] The `messages` array alternates between `user`/`assistant` turns
- [ ] For samples containing images, the number of `<image>` tokens matches the `images` array length
- [ ] Image paths are correctly accessible relative to `dataset_dir`
- [ ] Image files are in JPEG or PNG format
