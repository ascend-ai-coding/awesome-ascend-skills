#!/usr/bin/env bash
# subcommands/resources.sh — 节点和 Pod 资源使用分析

cmd_resources() {
    # 1. 节点资源使用（top）
    local node_top
    node_top=$(safe_kc kc top nodes --no-headers 2>/dev/null) || node_top=""
    local node_resources="[]"
    if [[ -n "$node_top" ]]; then
        local node_arr="[]"
        while read -r n_name n_cpu n_cpu_pct n_mem n_mem_pct _; do
            [[ -z "$n_name" ]] && continue
            node_arr=$(jq -n --argjson arr "$node_arr" \
                --arg name "$n_name" --arg cpu_cores "$n_cpu" --arg cpu_pct "$n_cpu_pct" \
                --arg memory "$n_mem" --arg memory_pct "$n_mem_pct" \
                '$arr + [{name: $name, cpu_cores: $cpu_cores, cpu_pct: $cpu_pct, memory: $memory, memory_pct: $memory_pct}]')
        done <<< "$node_top"
        node_resources="$node_arr"
    fi

    # 2. 节点压力条件（MemoryPressure, DiskPressure, PIDPressure）
    local node_conditions
    node_conditions=$(safe_kc kc get nodes -o json | jq '[.items[] | {
        name: .metadata.name,
        pressure_conditions: [.status.conditions[] | select(
            (.type == "MemoryPressure" or .type == "DiskPressure" or .type == "PIDPressure") and .status == "True"
        ) | {type: .type, message: .message}]
    } | select(.pressure_conditions | length > 0)]')

    # 3. Top Pods by CPU
    local pod_top_cpu
    pod_top_cpu=$(safe_kc kc_all top pods --sort-by=cpu --no-headers 2>/dev/null | head -20) || pod_top_cpu=""
    local top_pods_cpu="[]"
    if [[ -n "$pod_top_cpu" ]]; then
        local cpu_arr="[]"
        while read -r f1 f2 f3 f4 _; do
            [[ -z "$f1" ]] && continue
            if [[ -n "$f4" ]]; then
                # 4 字段：namespace name cpu memory
                cpu_arr=$(jq -n --argjson arr "$cpu_arr" \
                    --arg ns "$f1" --arg name "$f2" --arg cpu "$f3" --arg memory "$f4" \
                    '$arr + [{namespace: $ns, name: $name, cpu: $cpu, memory: $memory}]')
            else
                # 3 字段：name cpu memory
                cpu_arr=$(jq -n --argjson arr "$cpu_arr" \
                    --arg ns "default" --arg name "$f1" --arg cpu "$f2" --arg memory "$f3" \
                    '$arr + [{namespace: $ns, name: $name, cpu: $cpu, memory: $memory}]')
            fi
        done <<< "$pod_top_cpu"
        top_pods_cpu="$cpu_arr"
    fi

    # 4. Top Pods by Memory
    local pod_top_mem
    pod_top_mem=$(safe_kc kc_all top pods --sort-by=memory --no-headers 2>/dev/null | head -20) || pod_top_mem=""
    local top_pods_memory="[]"
    if [[ -n "$pod_top_mem" ]]; then
        local mem_arr="[]"
        while read -r f1 f2 f3 f4 _; do
            [[ -z "$f1" ]] && continue
            if [[ -n "$f4" ]]; then
                mem_arr=$(jq -n --argjson arr "$mem_arr" \
                    --arg ns "$f1" --arg name "$f2" --arg cpu "$f3" --arg memory "$f4" \
                    '$arr + [{namespace: $ns, name: $name, cpu: $cpu, memory: $memory}]')
            else
                mem_arr=$(jq -n --argjson arr "$mem_arr" \
                    --arg ns "default" --arg name "$f1" --arg cpu "$f2" --arg memory "$f3" \
                    '$arr + [{namespace: $ns, name: $name, cpu: $cpu, memory: $memory}]')
            fi
        done <<< "$pod_top_mem"
        top_pods_memory="$mem_arr"
    fi

    # 5. 缺少资源限制的 Pod（requests/limits）
    local missing_limits
    missing_limits=$(safe_kc kc_all get pods -o json | jq '[.items[] | select(.status.phase == "Running") | {
        name: .metadata.name,
        namespace: .metadata.namespace,
        containers: [.spec.containers[] | {
            name: .name,
            requests: (.resources.requests // {}),
            limits: (.resources.limits // {}),
            no_limits: ((.resources.limits // {}) == {}),
            no_requests: ((.resources.requests // {}) == {})
        }]
    } | select(.containers | any(.no_limits or .no_requests))] | .[0:30]')

    local data
    data=$(jq -n \
        --argjson node_resources "$node_resources" \
        --argjson node_conditions "$node_conditions" \
        --argjson top_cpu "$top_pods_cpu" \
        --argjson top_memory "$top_pods_memory" \
        --argjson missing_limits "$missing_limits" \
        '{node_usage: $node_resources, node_pressure: $node_conditions, top_pods_by_cpu: $top_cpu, top_pods_by_memory: $top_memory, pods_missing_limits: $missing_limits}')

    json_envelope "ok" "resources" "$data"
}