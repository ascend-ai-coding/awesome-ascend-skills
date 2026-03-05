"""
Wan2.2 Distributed Training Setup

Complete example for setting up multi-device distributed training
on Ascend NPU with HCCL backend, tensor/sequence parallel,
and enhanced communication strategies.
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
import torch_npu
from torch.distributed import ProcessGroupNCCL, ProcessGroupHCCL

# ===================================
# 1. NPU Distributed Initialization
# ===================================


def init_npu_distributed(backend=None, init_method=None, timeout=None):
    """
    Initialize distributed training for NPU/HCCL.

    Args:
        backend: Force specific backend ("hccl", "nccl", "gloo")
        init_method: Distributed initialization method
        timeout: Communication timeout

    Returns:
        rank, world_size, local_rank
    """
    # Determine backend
    if backend is None:
        backend = "hccl" if torch_npu.is_available() else "nccl"

    print(f"Initializing distributed with backend: {backend}")

    # Initialize process group
    if timeout is None:
        timeout = torch.distributed.default_pg_timeout

    dist.init_process_group(backend=backend, init_method=init_method, timeout=timeout)

    # NPU-specific device setup
    if torch_npu.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch_npu.npu.set_device(local_rank)
        print(f"NPU device set: local_rank={local_rank}")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Distributed initialized: rank={rank}, world_size={world_size}")
    return rank, world_size, local_rank


# ===================================
# 2. Enhanced Tensor Parallel (NPU-Optimized)
# ===================================


class NPUTensorParallelLinear:
    """
    Enhanced tensor-parallel linear layer for NPU.

    Features:
        - npu_mm_all_reduce_base for efficient GEMM
        - Automatic device assignment
        - Memory-efficient communication
    """

    def __init__(self, input_size, output_size, world_size, rank, bias=True):
        self.input_size = input_size
        self.output_size = output_size
        self.world_size = world_size
        self.rank = rank

        # Partitioned dimensions
        self.input_per_partition = input_size // world_size
        self.output_per_partition = output_size // world_size

        # Create weight and bias
        self.weight = torch.nn.Parameter(
            torch.empty(output_per_partition, input_per_partition)
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(output_per_partition))
        else:
            self.register_parameter("bias", None)

        # Create process group
        self.tp_group = dist.new_group(ranks=list(range(world_size)))

    def forward(self, x):
        """Forward pass with NPU-optimized all-reduce."""
        # Partition input
        x_partition = x[
            :,
            self.rank * self.input_per_partition : (self.rank + 1)
            * self.input_per_partition,
        ]

        # Compute forward
        output = torch.matmul(x_partition, self.weight.t())
        if self.bias is not None:
            output = output + self.bias

        # All-reduce for tensor parallel
        output = self.npu_all_reduce(output)

        return output

    def npu_all_reduce(self, tensor):
        """NPU-optimized all-reduce."""
        if torch_npu.is_available():
            # Use NPU-specific all-reduce
            import torch_npu

            return torch_npu.npu_mm_all_reduce_base(
                tensor, reduction="sum", comm_group=self.tp_group
            )
        else:
            # Fallback to PyTorch
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.tp_group)
            return tensor


# ===================================
# 3. Sequence Parallel Strategies
# ===================================


class UlyssesSequenceParallel:
    """
    Ulysses sequence parallel strategy for long sequences.

    Features:
        - Sequence splitting across ranks
        - Gradient communication optimization
        - Memory-efficient attention
    """

    def __init__(self, world_size, max_seq_len=1024):
        self.world_size = world_size
        self.max_seq_len = max_seq_len

        # Calculate sequence length per rank
        self.seq_length_per_rank = max_seq_len // world_size

        # Create sequence group
        self.seq_group = dist.new_group(ranks=list(range(world_size)))

    def split_sequence(self, x):
        """
        Split sequence across ranks.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]

        Returns:
            x_partition: Partitioned sequence [batch_size, seq_len_per_rank, hidden_dim]
        """
        seq_len = x.size(1)

        # Ensure sequence can be evenly split
        if seq_len % self.world_size != 0:
            raise ValueError(
                f"Sequence length {seq_len} not divisible by world_size {self.world_size}"
            )

        # Partition sequence
        x_partition = x[
            :,
            self.rank * self.seq_length_per_rank : (self.rank + 1)
            * self.seq_length_per_rank,
            :,
        ]

        return x_partition

    def gather_gradients(self, grad_output):
        """
        Gather gradients across ranks.

        Args:
            grad_output: Gradient tensor [batch_size, seq_len_per_rank, hidden_dim]

        Returns:
            gathered_grad: Complete gradient [batch_size, seq_len, hidden_dim]
        """
        # Create empty tensor for gathering
        seq_len = grad_output.size(1)
        gathered_grad = torch.empty(
            grad_output.size(0),
            seq_len * self.world_size,
            grad_output.size(2),
            device=grad_output.device,
        )

        # Gather gradients
        dist.all_gather_into_tensor(gathered_grad, grad_output, group=self.seq_group)

        return gathered_grad


# ===================================
# 4. VAE Patch Parallel (NPU-Enhanced)
# ===================================


class VAEPatchParallel:
    """
    2D spatial patch parallel for VAE on NPU.

    Features:
        - Row/column communication groups
        - NPU-optimized communication
        - Memory-efficient VAE training
    """

    def __init__(self, patch_size, tp_size):
        self.patch_size = patch_size
        self.tp_size = tp_size

        # Create communication groups
        self.row_group = dist.new_group(ranks=list(range(tp_size)))
        self.col_group = dist.new_group(ranks=list(range(tp_size, 2 * tp_size)))

        # Create NPU streams
        self.compute_stream = torch_npu.Stream()
        self.comm_stream = torch_npu.Stream()

    def split_patches(self, x):
        """
        Split feature map into patches.

        Args:
            x: Input [B, C, H, W]

        Returns:
            patches: Reshaped patches [B, num_patches, C, patch_size, patch_size]
        """
        B, C, H, W = x.shape

        # Check if divisible by patch_size
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError("Feature map dimensions not divisible by patch_size")

        # Split into patches
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size

        # Reshape and permute
        patches = x.reshape(
            B, C, H_patches, self.patch_size, W_patches, self.patch_size
        )
        patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(
            B, H_patches * W_patches, C, -1
        )

        return patches

    def communicate_patches(self, patches):
        """
        Communicate patches across ranks.

        Args:
            patches: Patch tensor [B, num_patches, C * patch_size²]

        Returns:
            communicated_patches: After communication
        """
        # Row communication (first half of ranks)
        dist.all_reduce(patches, op=dist.ReduceOp.SUM, group=self.row_group)
        torch_npu.synchronize()

        # Column communication (second half of ranks)
        dist.all_reduce(patches, op=dist.ReduceOp.SUM, group=self.col_group)
        torch_npu.synchronize()

        return patches


# ===================================
# 5. Enhanced Communication Overlap
# ===================================


class AsyncCommManager:
    """
    Async communication manager for compute-communication overlap.

    Features:
        - Stream-based communication
        - NPU-optimized scheduling
        - Memory-efficient execution
    """

    def __init__(self, world_size):
        self.world_size = world_size

        # Create NPU streams
        self.comm_stream = torch_npu.Stream()
        self.compute_stream = torch_npu.Stream()

        # Create events for synchronization
        self.comm_event = torch_npu.Event()
        self.compute_event = torch_npu.Event()

    def async_all_reduce(self, tensor):
        """
        Asynchronous all-reduce with compute overlap.

        Args:
            tensor: Input tensor

        Returns:
            result: After communication
        """
        # Start communication in dedicated stream
        with torch_npu.stream(self.comm_stream):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            self.comm_event.record()

        # Continue computation in compute stream
        with torch_npu.stream(self.compute_stream):
            # Do other work while communication happens
            # This is where you'd compute other parts of the model
            pass

        # Wait for both to complete
        self.compute_stream.wait_event(self.comm_event)
        torch_npu.synchronize()

        return tensor

    def pipeline_attention(self, q, k, v):
        """
        Pipeline attention with async all-reduce.

        Args:
            q, k, v: Attention tensors

        Returns:
            output: Attention output
        """
        # Start attention computation
        with torch_npu.stream(self.compute_stream):
            output = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
            output = torch.softmax(output, dim=-1)
            output = torch.matmul(output, v)

            self.compute_event.record()

        # Start async communication
        with torch_npu.stream(self.comm_stream):
            self.comm_stream.wait_event(self.compute_event)
            dist.all_reduce(output, op=dist.ReduceOp.SUM)

        # Wait for completion
        torch_npu.synchronize()

        return output


# ===================================
# 6. Distributed Training Manager
# ===================================


class DistributedWanTrainer:
    """
    Complete distributed training manager for Wan2.2.

    Features:
        - NPU/HCCL initialization
        - Tensor parallel
        - Sequence parallel
        - VAE patch parallel
        - Async communication
    """

    def __init__(self, world_size, args):
        self.world_size = world_size
        self.rank = dist.get_rank()

        # Initialize distributed
        self.rank, self.world_size, self.local_rank = init_npu_distributed()

        # Setup parallel strategies
        self.tensor_parallel = None
        self.sequence_parallel = None
        self.vae_parallel = None
        self.async_manager = None

        # Select parallel strategies based on configuration
        if args.tensor_parallel > 1:
            self.tensor_parallel = NPUTensorParallelLinear(
                args.hidden_size,
                args.hidden_size,
                args.tensor_parallel,
                self.rank % args.tensor_parallel,
            )

        if args.sequence_parallel:
            self.sequence_parallel = UlyssesSequenceParallel(
                world_size, args.max_seq_len
            )

        if args.vae_patch_parallel:
            self.vae_parallel = VAEPatchParallel(args.patch_size, args.tensor_parallel)

        if args.async_comm:
            self.async_manager = AsyncCommManager(world_size)

    def setup_model_parallel(self, model):
        """
        Setup model parallel components.
        """
        if self.tensor_parallel:
            model = model.to(f"cuda:{self.local_rank}")

        return model

    def forward_attention(self, q, k, v):
        """
        Forward pass with parallel strategies.
        """
        if self.sequence_parallel:
            q = self.sequence_parallel.split_sequence(q)
            k = self.sequence_parallel.split_sequence(k)
            v = self.sequence_parallel.split_sequence(v)

        if self.tensor_parallel:
            q = self.tensor_parallel.forward(q)

        if self.async_manager:
            output = self.async_manager.pipeline_attention(q, k, v)
        else:
            output = self.attention_baseline(q, k, v)

        return output

    def attention_baseline(self, q, k, v):
        """Baseline attention computation."""
        output = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        output = torch.softmax(output, dim=-1)
        output = torch.matmul(output, v)
        return output


# ===================================
# Main Setup Function
# ===================================


def setup_distributed_training(args):
    """
    Complete distributed training setup.

    Args:
        args: Command line arguments

    Returns:
        trainer: DistributedWanTrainer instance
        model: Wrapped model
    """
    print("=== Wan2.2 Distributed Training Setup ===")
    print(f"World size: {args.world_size}")
    print(f"Tensor parallel: {args.tensor_parallel}")
    print(f"Sequence parallel: {args.sequence_parallel}")
    print(f"VAE patch parallel: {args.vae_patch_parallel}")
    print(f"Async communication: {args.async_comm}")

    # Initialize distributed trainer
    trainer = DistributedWanTrainer(args.world_size, args)

    # Create and wrap model
    # model = create_wan22_model()
    # model = trainer.setup_model_parallel(model)

    print("Setup complete!")
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 Distributed Training Setup")
    parser.add_argument("--world_size", type=int, default=4, help="World size")
    parser.add_argument(
        "--tensor_parallel", type=int, default=1, help="Tensor parallel size"
    )
    parser.add_argument(
        "--sequence_parallel", action="store_true", help="Enable sequence parallel"
    )
    parser.add_argument(
        "--vae_patch_parallel", action="store_true", help="Enable VAE patch parallel"
    )
    parser.add_argument(
        "--async_comm", action="store_true", help="Enable async communication"
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument("--patch_size", type=int, default=32, help="Patch size for VAE")
    parser.add_argument(
        "--hidden_size", type=int, default=1024, help="Hidden dimension"
    )

    args = parser.parse_args()

    # Setup distributed training
    trainer = setup_distributed_training(args)

    print("\nNext steps:")
    print("1. Load Wan2.2 model")
    print("2. Use DistributedWanTrainer for training")
    print("3. Monitor performance with:")
    print("   watch -n 1 npu-smi info")


if __name__ == "__main__":
    main()
