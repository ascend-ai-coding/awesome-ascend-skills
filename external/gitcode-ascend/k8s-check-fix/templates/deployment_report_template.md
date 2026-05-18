# 📦 Deployment 分析：`{{deployment_name}}`

**命名空间**: {{namespace}}  
**创建时间**: {{creation_time}}  
**滚动策略**: {{strategy}}

## 副本状态
| 期望 | 就绪 | 可用 | 不可用 | 已更新 |
|------|------|------|--------|--------|
| {{desired}} | {{ready}} | {{available}} | {{unavailable}} | {{updated}} |

## 滚动状态
{{rollout_status}}


## 历史版本
| 版本 | ReplicaSet | 镜像 | 创建时间 |
|------|------------|------|----------|
{{history_rows}}

## 最近事件
| 类型 | 原因 | 消息 | 次数 | 最后时间 |
|------|------|------|------|----------|
{{events_rows}}

## 诊断结论
{{diagnosis}}

## 修复建议
{{recommendations}}