#!/usr/bin/env bash
# subcommands/events.sh — 集群事件汇总

cmd_events() {
    # 可选命名空间参数
    local ns="${SUBCOMMAND_ARGS[0]:-}"
    if [[ -n "$ns" && -z "$NAMESPACE" ]]; then
        NAMESPACE="$ns"
    fi

    local events_json
    events_json=$(safe_kc kc_all get events --sort-by='.lastTimestamp' -o json 2>/dev/null | jq --arg since "$SINCE" '{
        all_events: [.items[] | {
            namespace: .metadata.namespace,
            type: .type,
            reason: .reason,
            object_kind: .involvedObject.kind,
            object_name: .involvedObject.name,
            message: .message,
            count: .count,
            first_timestamp: .firstTimestamp,
            last_timestamp: .lastTimestamp,
            source: (.source.component // "")
        }] | sort_by(.last_timestamp) | reverse,
        summary: {
            total: ([.items[]] | length),
            warnings: ([.items[] | select(.type == "Warning")] | length),
            normals: ([.items[] | select(.type == "Normal")] | length),
            top_reasons: ([.items[] | .reason] | group_by(.) | map({reason: .[0], count: length}) | sort_by(.count) | reverse | .[0:10])
        }
    }') || events_json='{"all_events": [], "summary": {"total": 0, "warnings": 0, "normals": 0, "top_reasons": []}}'

    # 限制输出数量，避免过大
    events_json=$(echo "$events_json" | jq '.all_events = (.all_events | .[0:100])')

    json_envelope "ok" "events" "$events_json"
}