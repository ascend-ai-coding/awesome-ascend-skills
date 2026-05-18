# 论文筛选规则

## 搜索关键词

| 关键词 | 领域 |
|--------|------|
| recommendation | 推荐系统 |
| recommender system | 推荐系统 |
| collaborative filtering | 协同过滤 |
| CTR prediction | CTR 预估 |


## 目标会议/期刊

- RecSys - 推荐系统顶会
- SIGIR - 信息检索顶会
- KDD - 数据挖掘顶会
- WWW - Web 顶会
- ICML - 机器学习顶会
- NeurIPS - 神经信息处理顶会
- TOIS - ACM  TOIS 期刊

## 热门方向

- Sequential Recommendation (序列推荐)
- CTR Prediction (CTR 预估)
- Graph Neural Network (图神经网络)
- Multi-modal Recommendation (多模态推荐)
- Reinforcement Learning (强化学习)
- Debiasing (去偏)
- Explainable Recommendation (可解释推荐)
- Cold Start (冷启动)
- Federated Recommendation (联邦推荐)

## 源码检测规则

1. **优先检查** `github_url` 字段（enrich 阶段从论文 raw 内容提取）
2. **备用**：从论文 `comments` 和 `abstract` 字段提取 GitHub 链接
3. 调用 GitHub API **递归**验证仓库至少有 **3 个 .py 文件**（含子目录，最多 3 层）
4. 排除只有 README 的空仓库

## 去重逻辑

- 基于 arxiv_id 去重
- 保留首次出现的论文