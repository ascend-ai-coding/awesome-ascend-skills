# 🏥 Kubernetes 诊断报告

**集群上下文**: {{context}}  
**命名空间**: {{namespace}}  
**检查时间**: {{timestamp}}  
**子命令**: {{subcommand}}

## 概览
{{overview}}

## 节点状态
| 节点 | 状态 | CPU 使用 | 内存使用 | 压力条件 |
|------|------|----------|----------|----------|
{{node_status_rows}}

## 问题 Pod 列表
| Pod | 命名空间 | 状态 | 重启次数 | 节点 |
|-----|----------|------|----------|------|
{{problem_pods_rows}}

## 告警事件摘要
| 类型 | 原因 | 对象 | 消息 | 次数 | 最后时间 |
|------|------|------|------|------|----------|
{{events_summary_rows}}

## 详细诊断
{{detailed_diagnosis}}

## 修复建议
{{recommendations}}