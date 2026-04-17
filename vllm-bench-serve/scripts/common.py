#!/usr/bin/env python3
"""common.py — Shared constants and utilities for vllm-bench-serve scripts."""

# Backend → required --endpoint mapping.
# vllm bench serve defaults --endpoint to /v1/completions, so backends that use
# a different endpoint MUST have it explicitly specified or the request will fail.
BACKEND_ENDPOINT_MAP = {
    "vllm": "/v1/completions",
    "openai": "/v1/completions",
    "openai-chat": "/v1/chat/completions",
    "openai-audio": "/v1/audio/transcriptions",
    "openai-embeddings": "/v1/embeddings",
    "openai-embeddings-chat": "/v1/embeddings",
    "openai-embeddings-clip": "/v1/embeddings",
    "openai-embeddings-vlm2vec": "/v1/embeddings",
    "infinity-embeddings": "/v1/embeddings",
    "infinity-embeddings-clip": "/v1/embeddings",
    "vllm-pooling": "/pooling",
    "vllm-rerank": "/v1/rerank",
}

ALL_BACKENDS = list(BACKEND_ENDPOINT_MAP.keys())

# Dataset-backend compatibility matrix
DATASET_BACKEND_COMPAT = {
    "random": [
        "openai", "openai-chat", "vllm",
        "openai-embeddings", "openai-embeddings-chat",
        "openai-embeddings-clip", "openai-embeddings-vlm2vec",
        "infinity-embeddings", "infinity-embeddings-clip",
        "vllm-pooling",
    ],
    "random-mm": ["openai-chat"],
    "random-rerank": ["vllm-rerank"],
    "sharegpt": ["openai", "openai-chat", "vllm"],
    "burstgpt": ["openai", "openai-chat", "vllm"],
    "custom": ["openai", "openai-chat", "vllm"],
    "custom_mm": ["openai-chat"],
    "prefix_repetition": ["openai", "openai-chat", "vllm"],
    "spec_bench": ["openai", "openai-chat", "vllm"],
    "hf": None,  # varies by HF dataset, skip strict check
    "sonnet": ["openai", "openai-chat", "vllm"],
}

ALL_DATASETS = list(DATASET_BACKEND_COMPAT.keys())

# Datasets that require --dataset-path
DATASETS_REQUIRING_PATH = {"sharegpt", "burstgpt", "custom", "custom_mm", "spec_bench"}
