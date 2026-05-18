# 🏥 Pod 详细检查：`{{pod_name}}`

**命名空间**: {{namespace}}  
**节点**: {{node}}  
**状态**: {{phase}}  
**QoS 等级**: {{qos_class}}  
**创建时间**: {{creation_time}}  
**拥有者**: {{owner}}

## 容器状态
| 容器 | 镜像 | 就绪 | 重启次数 | 状态 |
|------|------|------|----------|------|
{{container_status_rows}}

## 镜像版本差异
{{#if image_mismatches}}
| 容器 | 期望镜像 | 运行镜像 |
|------|----------|----------|
{{image_mismatch_rows}}
{{else}}
✅ 所有容器镜像与规格一致
{{/if}}

## 事件时间线
| 类型 | 原因 | 消息 | 次数 | 最后时间 |
|------|------|------|------|----------|
{{events_rows}}

## 日志摘要（最近 {{tail}} 行）
{{current_logs}}



## 上一次容器日志（崩溃时）
{{previous_logs}}



## 诊断结论
{{diagnosis}}

## 修复建议
{{recommendations}}