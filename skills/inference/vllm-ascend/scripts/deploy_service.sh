#!/bin/bash
set -e

# Configuration (use placeholders)
MODEL_PATH="${MODEL_PATH:-/path/to/model}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TP_SIZE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model PATH           Path to model directory (default: $MODEL_PATH)"
            echo "  --port PORT            Service port (default: $PORT)"
            echo "  --tensor-parallel-size SIZE   Tensor parallel size (default: $TP_SIZE)"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=== vLLM-Ascend Service Deployment ==="
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Tensor parallel size: $TP_SIZE"
echo ""

# Check environment
echo "Step 1: Checking environment..."
if [ -f vllm-ascend/scripts/check_env.sh ]; then
    bash vllm-ascend/scripts/check_env.sh
else
    echo "Warning: check_env.sh not found, skipping environment check"
fi
echo ""

# Start service
echo "Step 2: Starting vLLM service..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --trust-remote-code \
    --disable-log-requests &

VLLM_PID=$!

echo "Service started with PID: $VLLM_PID"
echo ""

# Wait for service to be ready
echo "Step 3: Waiting for service to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Service is ready!"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Attempt $RETRY_COUNT/$MAX_RETRIES - Service not ready yet..."
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Error: Service did not become ready in time"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Health check
echo ""
echo "Step 4: Running health check..."
curl -s "http://localhost:$PORT/health" | python3 -m json.tool || echo "Health check failed"
echo ""

echo "=== Deployment Complete ==="
echo "Service running at: http://localhost:$PORT"
echo "Health check endpoint: http://localhost:$PORT/health"
echo "API endpoint: http://localhost:$PORT/v1/chat/completions"
echo ""
echo "To stop the service: kill $VLLM_PID"
echo "Or use: killall -9 python3"