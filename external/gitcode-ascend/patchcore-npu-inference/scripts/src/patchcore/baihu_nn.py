#!/usr/bin/env python
"""BaihuNN — NPU (Ascend/Huawei) vector search engine.

Implements the same interface as FaissNN but uses torch_npu matrix
operations for L2 nearest-neighbor search. Falls back to CPU when
torch_npu is not available.

Features
--------
- FP16 mode on NPU with automatic FP32 re-ranking (two-stage search)
  to recover full precision while keeping fast FP16 matmul.
- FP32 reference mode (CPU or NPU).
"""

import logging
from typing import Optional

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


class BaihuNN:
    """NPU-accelerated L2 nearest neighbor search.

    Uses the identity  ||q-x||² = ||q||² + ||x||² - 2·q·xᵀ  to compute all
    pairwise distances efficiently on the NPU (or CPU fallback).

    Parameters
    ----------
    on_npu : bool
        Whether to attempt NPU device placement.  When False, always
        uses CPU regardless of torch_npu availability.
    device_id : int
        NPU device index (e.g. ``0`` for ``npu:0``).
    batch_size : int
        Number of query vectors processed in a single NPU operation.
    use_fp16 : bool
        Convert stored features to float16 (half precision) to save
        NPU memory and accelerate matmul.
        When enabled on NPU, a two-stage search is used:
        FP16 coarse pre-filter (top-``k * 4``) followed by
        FP32 exact re-ranking to restore full index accuracy.
    """

    _RERANK_MULTIPLIER = 8  # 从粗筛候选集中选取 4*K 个进行精排

    def __init__(
        self,
        on_npu: bool = False,
        device_id: int = 0,
        batch_size: int = 4096,
        use_fp16: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self._features: Optional[torch.Tensor] = None          # 存储在设备上的特征 (FP16 或 FP32)
        self._features_fp32: Optional[torch.Tensor] = None     # 存储在 CPU 上的 FP32 备份，用于精排

        # ---------- device selection ----------
        if on_npu:
            try:
                import torch_npu  # noqa: F401
                self._device = torch.device(f"npu:{device_id}")
                LOGGER.info("BaihuNN initialised on %s", self._device)
            except (ImportError, RuntimeError) as exc:
                LOGGER.warning(
                    "torch_npu not available (%s); falling back to CPU.", exc
                )
                self._device = torch.device("cpu")
        else:
            self._device = torch.device("cpu")
            LOGGER.debug("BaihuNN initialised on CPU (on_npu=False).")

    # -------- public interface (mirrors FaissNN) ----------

    def fit(self, features: np.ndarray) -> None:
        """Store index features as a NPU (or CPU) tensor.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(N, D)`` — ``N`` database vectors of dimension ``D``.
        """
        LOGGER.debug("BaihuNN.fit with %d vectors of dim %d", *features.shape)
        features = np.ascontiguousarray(features)
        feat = torch.from_numpy(features).to(self._device)

        # 如果启用 FP16 且设备是 NPU，则保存一份 CPU 上的 FP32 备份用于精排
        if self.use_fp16 and self._device.type != "cpu":
            self._features_fp32 = torch.from_numpy(features).cpu()  # 保留 FP32 精度在 CPU
            feat = feat.half()  # 在 NPU 上存储和使用 FP16
        else:
            self._features_fp32 = None  # FP32 模式或 CPU 模式下不需要备份

        self._features = feat

    def run(
        self,
        n_nearest_neighbours: int,
        query_features: np.ndarray,
        index_features: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the *k* nearest neighbours for every query vector.

        Parameters
        ----------
        n_nearest_neighbours : int
            Number of neighbours to retrieve per query.
        query_features : np.ndarray
            Shape ``(Q, D)``.
        index_features : np.ndarray | None
            Override the index (default: use the features from ``fit``).

        Returns
        -------
        distances : np.ndarray, shape ``(Q, K)``
            Sorted L2 distances (ascending).
        indices : np.ndarray, shape ``(Q, K)``
            Corresponding column indices into ``index_features``.
        """
        Q = (
            query_features.to(self._device)
            if isinstance(query_features, torch.Tensor)
            else torch.from_numpy(np.ascontiguousarray(query_features)).to(self._device)
        )

        if index_features is not None:
            # 临时索引模式：使用传入的 index_features，不进行两阶段检索
            X = (
                index_features.to(self._device)
                if isinstance(index_features, torch.Tensor)
                else torch.from_numpy(np.ascontiguousarray(index_features)).to(self._device)
            )
            if self._features is not None:
                Q = Q.to(self._features.dtype)
                X = X.to(self._features.dtype)
            use_rerank = False
            fp32_index = None
        else:
            # 使用 fit() 存储的索引
            if self._features is None:
                raise RuntimeError("No index features — call fit() first.")
            X = self._features
            Q = Q.to(X.dtype)
            # 判断是否需要并可以进行两阶段检索
            use_rerank = (
                self.use_fp16
                and self._device.type != "cpu"
                and self._features_fp32 is not None
            )
            fp32_index = self._features_fp32  # CPU 上的 FP32 特征备份

        n_nearest = min(n_nearest_neighbours, X.shape[0])
        nq = Q.shape[0]
        Q = Q.contiguous()  # ensure contiguous for NPU matmul
        X_t = X.T.contiguous()  # pre-transpose once (Ni, D) -> (D, Ni)

        # Precompute ‖x‖² once — stays on device, reused across batches
        x_norm = (X * X).sum(dim=1, keepdim=True)  # (Ni, 1)

        # Pre-allocate output buffers (avoids repeated concat)
        distances = np.empty((nq, n_nearest), dtype=np.float32)
        indices = np.empty((nq, n_nearest), dtype=np.int64)

        for start in range(0, nq, self.batch_size):
            end = min(start + self.batch_size, nq)
            q_batch = Q[start:end]  # (B, D)

            # ‖q-x‖² = ‖q‖² + ‖x‖² - 2·q·xᵀ   using pre-tranposed X
            q_norm = (q_batch * q_batch).sum(dim=1, keepdim=True)  # (B, 1)
            dot = torch.mm(q_batch, X_t)  # (B, Ni)
            dist = q_norm + x_norm.T - 2.0 * dot  # (B, Ni)

            # Numerical safety: clamp FP16 / FP32 rounding noise
            torch.clamp(dist, min=0.0, out=dist)

            if use_rerank:
                # --- 两阶段检索: FP16 粗筛 + FP32 精排 ---
                # 阶段1: 用 FP16 结果粗筛，选出更多候选 (例如 4*K 个)
                n_candidates = min(n_nearest * self._RERANK_MULTIPLIER, X.shape[0])
                _, idx_coarse = torch.topk(dist, n_candidates, dim=1, largest=False)  # (B, n_candidates)

                # 阶段2: 从 CPU 备份中取出对应候选的 FP32 特征，进行精确距离计算
                B = end - start
                # 将候选索引展平，用于从 CPU 备份中收集特征
                idx_flat_cpu = idx_coarse.reshape(-1).cpu()
                cand_fp32 = fp32_index[idx_flat_cpu]  # (B * n_candidates, D)
                cand_fp32 = cand_fp32.reshape(B, n_candidates, -1)     # (B, n_candidates, D)

                # 在 FP32 下精确计算查询与候选之间的距离
                q_batch_fp32 = q_batch.float().cpu()  # 转 FP32 计算
                q_norm_fp32 = (q_batch_fp32 * q_batch_fp32).sum(dim=1, keepdim=True)  # (B, 1)
                c_norm_fp32 = (cand_fp32 * cand_fp32).sum(dim=2)  # (B, n_candidates)
                # 批量矩阵乘法计算点积: (B, n_candidates, D) @ (B, D, 1) -> (B, n_candidates, 1)
                dot_exact = torch.bmm(cand_fp32, q_batch_fp32.unsqueeze(-1)).squeeze(-1)  # (B, n_candidates)
                dist_exact = q_norm_fp32 + c_norm_fp32 - 2.0 * dot_exact  # (B, n_candidates)
                torch.clamp(dist_exact, min=0.0, out=dist_exact)

                # 从精确距离中选出最终的 top-K
                d_batch, i_local = torch.topk(dist_exact, n_nearest, dim=1, largest=False)  # i_local: (B, K) 在候选中的局部索引
                # 将局部索引映射回原始索引
                i_batch = torch.gather(idx_coarse, 1, i_local.to(idx_coarse.device))  # (B, K)
            else:
                # --- 单阶段检索 (FP32 模式或 CPU 回退) ---
                d_batch, i_batch = torch.topk(dist, n_nearest, dim=1, largest=False)

            # Write directly into pre-allocated buffer (no concat overhead)
            distances[start:end] = d_batch.cpu().numpy()
            indices[start:end] = i_batch.cpu().numpy()

        return distances, indices

    def save(self, filename: str) -> None:
        """Persist the index tensor to disk.

        Parameters
        ----------
        filename : str
            Path to the output file (``.pt`` / ``.pth``).
        """
        payload = {
            "features": self._features,
            "features_fp32": self._features_fp32,
        }
        torch.save(payload, filename)
        LOGGER.debug("BaihuNN state saved to %s", filename)

    def load(self, filename: str) -> None:
        """Load a previously saved index tensor.

        Parameters
        ----------
        filename : str
            Path to the saved tensor file.
        """
        data = torch.load(filename, map_location="cpu")
        if isinstance(data, dict):
            # 新格式：包含两个特征张量
            feat = data.get("features")
            self._features_fp32 = data.get("features_fp32")
        else:
            # 旧格式：只有单个特征张量
            feat = data
            self._features_fp32 = None

        if feat is not None:
            feat = feat.to(self._device)
            if self.use_fp16 and self._device.type != "cpu":
                # 确保加载的特征是 FP16 (如果是 NPU 且启用 FP16)
                feat = feat.half()
        self._features = feat
        LOGGER.debug("BaihuNN state loaded from %s (%s)", filename, self._device)

    def reset_index(self) -> None:
        """Discard the stored index features."""
        self._features = None
        self._features_fp32 = None
        LOGGER.debug("BaihuNN index reset.")
