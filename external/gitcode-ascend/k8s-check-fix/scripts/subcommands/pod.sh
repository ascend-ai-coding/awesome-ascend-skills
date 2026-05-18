#!/usr/bin/env bash
# subcommands/pod.sh — Pod 深入排查：describe + logs + events + 镜像差异

cmd_pod() {
    local pod_name="${SUBCOMMAND_ARGS[0]:-}"
    if [[ -z "$pod_name" ]]; then
        err "用法: $0 pod <pod-name> [--namespace <ns>]"
        exit 1
    fi

    # 自动检测命名空间（如果未指定）
    if [[ -z "$NAMESPACE" ]]; then
        local detected_ns
        detected_ns=$(run_kubectl get pods --all-namespaces -o json 2>/dev/null \
            | jq -r --arg name "$pod_name" '.items[] | select(.metadata.name == $name) | .metadata.namespace' | head -1)
        if [[ -n "$detected_ns" ]]; then
            NAMESPACE="$detected_ns"
        else
            err "Pod '$pod_name' 在所有命名空间中均未找到。请指定 --namespace。"
            exit 1
        fi
    fi

    # 1. Pod 详细信息（JSON）
    local pod_json
    pod_json=$(safe_kc kc get pod "$pod_name" -o json | jq '{
        name: .metadata.name,
        namespace: .metadata.namespace,
        phase: .status.phase,
        node: (.spec.nodeName // "unscheduled"),
        start_time: .status.startTime,
        labels: .metadata.labels,
        owner: ((.metadata.ownerReferences // [])[0] | {kind: .kind, name: .name} // null),
        containers: [.spec.containers[] | {
            name: .name,
            image: .image,
            resources: .resources,
            ports: [(.ports // [])[] | {containerPort: .containerPort, protocol: .protocol}]
        }],
        container_statuses: [(.status.containerStatuses // [])[] | {
            name: .name,
            ready: .ready,
            restart_count: .restartCount,
            image: .image,
            image_id: .imageID,
            state: .state,
            last_state: .lastState
        }],
        init_container_statuses: [(.status.initContainerStatuses // [])[] | {
            name: .name,
            ready: .ready,
            state: .state
        }],
        conditions: [(.status.conditions // [])[] | {type: .type, status: .status, reason: (.reason // ""), message: (.message // "")}],
        tolerations: (.spec.tolerations // []),
        node_selector: (.spec.nodeSelector // {}),
        qos_class: (.status.qosClass // "unknown")
    }')

    # 2. 镜像版本差异检测
    local image_mismatch
    image_mismatch=$(safe_kc kc get pod "$pod_name" -o json | jq '{
        mismatches: [
            . as $pod |
            ($pod.spec.containers // [])[] as $spec |
            ($pod.status.containerStatuses // [])[] |
            select(.name == $spec.name) |
            select(.image != $spec.image) |
            {container: .name, spec_image: $spec.image, running_image: .image}
        ]
    }')

    # 3. 当前日志
    local current_logs
    current_logs=$(safe_kc kc logs "$pod_name" --tail="$TAIL" --all-containers=true 2>&1) || current_logs="(no logs available)"
    current_logs=$(jq -n --arg logs "$current_logs" '$logs')

    # 4. 上一次日志（用于崩溃分析）
    local previous_logs
    previous_logs=$(safe_kc kc logs "$pod_name" --previous --tail="$TAIL" --all-containers=true 2>&1) || previous_logs="(no previous logs)"
    previous_logs=$(jq -n --arg logs "$previous_logs" '$logs')

    # 5. Pod 相关事件
    local pod_events
    pod_events=$(safe_kc kc get events --field-selector="involvedObject.name=$pod_name" --sort-by='.lastTimestamp' -o json 2>/dev/null | jq '{
        events: [.items[] | {
            type: .type,
            reason: .reason,
            message: .message,
            count: .count,
            first_timestamp: .firstTimestamp,
            last_timestamp: .lastTimestamp,
            source: (.source.component // "")
        }]
    }') || pod_events='{"events": []}'

    # 组装 JSON
    local data
    data=$(jq -n \
        --argjson pod "$pod_json" \
        --argjson image_mismatch "$image_mismatch" \
        --argjson current_logs "$current_logs" \
        --argjson previous_logs "$previous_logs" \
        --argjson events "$pod_events" \
        '{pod: $pod, image_mismatch: $image_mismatch, current_logs: $current_logs, previous_logs: $previous_logs, events: $events}')

    json_envelope "ok" "pod" "$data"
}