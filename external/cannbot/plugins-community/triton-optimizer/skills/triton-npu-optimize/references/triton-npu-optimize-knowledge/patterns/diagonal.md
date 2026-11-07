# Diagonal Block Traversal Pattern

## Summary

While it is good to access data from L2 cache as much as possible, having multiple
kernels accessing the *same* data from the L2 cache may cause bank conflicts that slow down operations.
One can use the diagonal access pattern to replace the usual swizzle pattern to alleviate this problem.
The example applies this technique to matrix multiplication, but it may be applicable in other contexts.

## Use When

- Large tiled matrix-style work shows poor locality or bank-conflict-like behavior even though the basic tiling is already reasonable.
- Many programs touch the same cache regions at the same time, so changing block traversal order may improve effective L2 use.

## Signals

### Code

- Traditional row-major or horizontal block assignment makes many cores touch the same left-matrix cache region at once.
- The matrix already spans many blocks along both `M` and `N`, so traversal order is a plausible performance lever rather than a cosmetic rewrite.
- The right-hand matrix is large enough that ordinary block traversal can churn L2 and lower reuse.

## Detail

Diagonal traversal changes the *order* of output tiles, not the tile compute body itself.
With ordinary row-major traversal, a `4 x 4` tile grid is visited like this:

```text
0  1  2  3
4  5  6  7
8  9 10 11
12 13 14 15
```

Inside a diagonal group, the same region is visited one diagonal stripe at a time instead:

```text
0  4  8 12
13  1  5  9
10 14  2  6
7 11 15  3
```

That small traversal change can reduce cases where many programs hammer the same cache region at once.
The example below keeps only the traversal logic and leaves the usual load / `tl.dot` / store body implied.

```python
@triton.jit
def matmul_kernel(
    mat_a,
    mat_b,
    mat_c,
    M,
    N,
    K,
    num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DIAGONAL_GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    num_blocks_m = triton.cdiv(M, BLOCK_M)
    num_blocks_n = triton.cdiv(N, BLOCK_N)
    num_blocks = num_blocks_m * num_blocks_n

    use_diagonal = (
        num_blocks_m >= DIAGONAL_GROUP
        and num_blocks_n >= DIAGONAL_GROUP
        and num_blocks_m % DIAGONAL_GROUP == 0
        and num_blocks_n % DIAGONAL_GROUP == 0
    )

    for block_idx in range(pid, num_blocks, num_cores):
        if use_diagonal:
            group_blocks = DIAGONAL_GROUP * DIAGONAL_GROUP
            groups_per_row = num_blocks_n // DIAGONAL_GROUP

            group_idx = block_idx // group_blocks
            local_idx = block_idx % group_blocks

            group_m0 = (group_idx // groups_per_row) * DIAGONAL_GROUP
            group_n0 = (group_idx % groups_per_row) * DIAGONAL_GROUP

            local_m = local_idx % DIAGONAL_GROUP
            diagonal_step = local_idx // DIAGONAL_GROUP
            local_n = (local_m + diagonal_step) % DIAGONAL_GROUP

            task_m_idx = group_m0 + local_m
            task_n_idx = group_n0 + local_n
        else:
            task_m_idx = block_idx // num_blocks_n
            task_n_idx = block_idx % num_blocks_n

        m_start = task_m_idx * BLOCK_M
        n_start = task_n_idx * BLOCK_N

        # Use (m_start, n_start) with the usual tiled matmul body here.
        # The optimization is the traversal order, not a different dot kernel.
```

This compact example assumes diagonal traversal only for full `DIAGONAL_GROUP x DIAGONAL_GROUP`
regions. Tail groups can fall back to ordinary traversal if partial-edge handling would make the
mapping harder to reason about than the expected cache benefit justifies.
