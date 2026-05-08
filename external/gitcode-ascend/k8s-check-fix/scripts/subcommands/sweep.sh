# subcommands/sweep.sh — 全集群健康检查

cmd_sweep() {
    # 1. 获取节点状态 JSON
    local nodes_json
    nodes_json=$(safe_kc kc get nodes -o json | jq '{
        total: (.items | length),
        ready: [.items[] | select(.status.conditions[] | select(.type=="Ready" and .status=="True"))] | length,
        not_ready: [.items[] | select(.status.conditions[] | select(.type=="Ready" and .status!="True"))] | length,
        nodes: [.items[] | {name: .metadata.name, status: (if (.status.conditions[] | select(.type=="Ready")).status == "True" then "Ready" else "NotReady" end)}]
    }')

    # 2. 获取问题 Pod
    local problem_pods
    problem_pods=$(safe_kc kc_all get pods -o json | jq '{
        total_pods: (.items | length),
        problem_pods: [.items[] | select(.status.phase != "Running" and .status.phase != "Succeeded") | {
            name: .metadata.name,
            namespace: .metadata.namespace,
            phase: .status.phase
        }]
    }')

    # ... 其他检查（事件、组件状态等）...

    local data
    data=$(jq -n --argjson nodes "$nodes_json" --argjson pods "$problem_pods" '{nodes: $nodes, pods: $pods}')
    json_envelope "ok" "sweep" "$data"
}