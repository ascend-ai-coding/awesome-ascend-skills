# Model Configuration Template
# Copy and modify for your specific model setup

from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import (
    extract_non_reasoning_content,
)

models = [
    dict(
        # Required: Service attribute
        attr="service",
        # Required: Model backend class
        # Options: VLLMCustomAPIChat, MindIEAPI, TritonAPI, TGIAPI, HuggingFace, VLLMOffline
        type=VLLMCustomAPIChat,
        # Required: Short name for results display
        abbr="vllm-api-custom",
        # Model tokenizer path (optional for API models, required for offline)
        path="",
        # Model name on server (empty string = auto-detect from service)
        model="",
        # Streaming mode: True for real-time, False for batch
        stream=False,
        # Request rate: 0 = send all at once, >0 = requests per second
        request_rate=0,
        # Max retry attempts per request
        retry=2,
        # API key (if required by service)
        api_key="",
        # === SERVICE CONNECTION ===
        # Service IP address
        host_ip="localhost",
        # Service port
        host_port=8080,
        # Custom URL (overrides host_ip:host_port if set)
        url="",
        # === INFERENCE SETTINGS ===
        # Maximum output token length
        max_out_len=512,
        # Concurrent request count (batch size)
        batch_size=1,
        # Trust remote code when loading tokenizer
        trust_remote_code=False,
        # Model-specific generation parameters
        generation_kwargs=dict(
            temperature=0.01,  # Lower = more deterministic
            top_p=0.9,  # Nucleus sampling
            top_k=50,  # Top-k sampling
            ignore_eos=False,  # Continue past EOS?
            # num_return_sequences=5,  # For pass@k evaluation
        ),
        # Output postprocessor
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
