#!/usr/bin/env bash
# subcommands/deploy.sh — Deployment 分析：状态、历史、ReplicaSet、事件

cmd_deploy() {
    local deploy_name="${SUBCOMMAND_ARGS[0]:-}"
    if [[ -z "$deploy_name" ]]; then
        err "用法: $0 deploy <deployment-name> [--namespace <ns>]"
        exit 1
    fi

    if [[ -z "$NAMESPACE" ]]; then
        NAMESPACE="default"
    fi

    # 1. Deployment 详细信息
    local deploy_json
    deploy_json=$(safe_kc kc get deployment "$deploy_name" -o json | jq '{
        name: .metadata.name,
        namespace: .metadata.namespace,
        replicas: {
            desired: .spec.replicas,
            ready: (.status.readyReplicas // 0),
            available: (.status.availableReplicas // 0),
            unavailable: (.status.unavailableReplicas // 0),
            updated: (.status.updatedReplicas // 0)
        },
        strategy: .spec.strategy,
        selector: .spec.selector,
        template_labels: .spec.template.metadata.labels,
        containers: [.spec.template.spec.containers[] | {name: .name, image: .image, resources: .resources}],
        conditions: [(.status.conditions // [])[] | {
            type: .type,
            status: .status,
            reason: (.reason // ""),
            message: (.message // ""),
            last_update: .lastUpdateTime
        }],
        generation: .metadata.generation,
        observed_generation: (.status.observedGeneration // 0),
        creation_timestamp: .metadata.creationTimestamp
    }')

    # 2. 滚动状态（文本）
    local rollout_status
    rollout_status=$(safe_kc kc rollout status "deployment/$deploy_name" --timeout=5s 2>&1) || true
    rollout_status=$(jq -n --arg s "$rollout_status" '$s')

    # 3. 滚动历史
    local rollout_history
    rollout_history=$(safe_kc kc rollout history "deployment/$deploy_name" -o json 2>/dev/null) || rollout_history='{}'
    if ! echo "$rollout_history" | jq . &>/dev/null 2>&1; then
        local history_text
        history_text=$(safe_kc kc rollout history "deployment/$deploy_name" 2>/dev/null) || history_text="(unavailable)"
        rollout_history=$(jq -n --arg h "$history_text" '{text: $h}')
    fi

    # 4. 关联的 ReplicaSet
    local selector
    selector=$(safe_kc kc get deployment "$deploy_name" -o json | jq -r '[.spec.selector.matchLabels | to_entries[] | "\(.key)=\(.value)"] | join(",")')
    local replicasets
    replicasets=$(safe_kc kc get replicasets -l "$selector" -o json 2>/dev/null | jq '{
        replicasets: [.items[] | {
            name: .metadata.name,
            desired: (.spec.replicas // 0),
            ready: (.status.readyReplicas // 0),
            available: (.status.availableReplicas // 0),
            revision: (.metadata.annotations["deployment.kubernetes.io/revision"] // ""),
            containers: [.spec.template.spec.containers[] | {name: .name, image: .image}],
            creation_timestamp: .metadata.creationTimestamp
        }] | sort_by(.revision) | reverse
    }') || replicasets='{"replicasets": []}'

    # 5. Deployment 事件
    local deploy_events
    deploy_events=$(safe_kc kc get events --field-selector="involvedObject.name=$deploy_name,involvedObject.kind=Deployment" --sort-by='.lastTimestamp' -o json 2>/dev/null | jq '{
        events: [.items[] | {
            type: .type,
            reason: .reason,
            message: .message,
            count: .count,
            last_timestamp: .lastTimestamp
        }]
    }') || deploy_events='{"events": []}'

    local data
    data=$(jq -n \
        --argjson deploy "$deploy_json" \
        --argjson rollout_status "$rollout_status" \
        --argjson rollout_history "$rollout_history" \
        --argjson replicasets "$replicasets" \
        --argjson events "$deploy_events" \
        '{deployment: $deploy, rollout_status: $rollout_status, rollout_history: $rollout_history, replicasets: $replicasets, events: $events}')

    json_envelope "ok" "deploy" "$data"
}