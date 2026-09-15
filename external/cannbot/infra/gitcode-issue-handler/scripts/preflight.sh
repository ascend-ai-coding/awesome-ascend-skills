#!/bin/bash
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# GitCode issue-handler operation-scoped environment capability check.
# Checks are selected just before the operation that needs them. With no
# --checks argument the full handler check remains available.
# Output: JSON report to stdout. User-provided inputs and environment blockers are
# separated so orchestrators may aggregate them without changing legacy callers.
# Exit 0 = ready, exit 1 = user input or environment repair is required.
#
# Usage:
#   bash preflight.sh                    # all checks including git author
#   bash preflight.sh --skip-git-author  # skip git user.name/email check
#   bash preflight.sh --checks api       # token, curl and python3
#   bash preflight.sh --checks git,tmp   # git and a writable temp root
#   bash preflight.sh --checks author    # git and git author
#   bash preflight.sh --work-dir <path>  # inspect this repo and use it for temp fallback
#   bash preflight.sh --token-available  # token is held in session, not the environment

set -euo pipefail

SKIP_GIT_AUTHOR=false
TOKEN_AVAILABLE=false
EXTENDED_PREFLIGHT=false
WORK_DIR="$PWD"
REQUESTED_CHECKS="full"
CHECKS_SEEN=false
while [ "$#" -gt 0 ]; do
    case "$1" in
        --skip-git-author)
            SKIP_GIT_AUTHOR=true
            shift
            ;;
        --token-available)
            TOKEN_AVAILABLE=true
            shift
            ;;
        --work-dir)
            if [ "$#" -lt 2 ]; then
                echo "--work-dir requires a path" >&2
                exit 2
            fi
            WORK_DIR="$2"
            EXTENDED_PREFLIGHT=true
            shift 2
            ;;
        --checks)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                echo "--checks requires a comma-separated value" >&2
                exit 2
            fi
            if [ "$CHECKS_SEEN" = true ]; then
                echo "--checks may only be specified once" >&2
                exit 2
            fi
            case "$2" in
                ,*|*,|*,,*)
                    echo "--checks contains an empty check group" >&2
                    exit 2
                    ;;
            esac
            REQUESTED_CHECKS="$2"
            CHECKS_SEEN=true
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

WANT_TOKEN=false
WANT_GIT=false
WANT_CURL=false
WANT_PYTHON=false
WANT_TMP=false
WANT_GIT_AUTHOR=false
IFS=',' read -r -a CHECK_GROUPS <<< "$REQUESTED_CHECKS"
for check_group in "${CHECK_GROUPS[@]}"; do
    case "$check_group" in
        full)
            WANT_TOKEN=true
            WANT_GIT=true
            WANT_CURL=true
            WANT_PYTHON=true
            WANT_TMP=true
            WANT_GIT_AUTHOR=true
            ;;
        api)
            WANT_TOKEN=true
            WANT_CURL=true
            WANT_PYTHON=true
            ;;
        git)
            WANT_GIT=true
            ;;
        tmp)
            WANT_TMP=true
            ;;
        author)
            WANT_GIT=true
            WANT_GIT_AUTHOR=true
            ;;
        *)
            echo "unknown check group: $check_group" >&2
            exit 2
            ;;
    esac
done

# --- Check helpers ---

json_escape() {
    local value="$1"
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    value=${value//$'\n'/\\n}
    value=${value//$'\r'/\\r}
    value=${value//$'\t'/\\t}
    printf '%s' "$value"
}

check_token() {
    if [ -n "${GITCODE_TOKEN:-}" ]; then
        printf '{"item":"token","status":"pass","detail":"env GITCODE_TOKEN (length=%d)","source":"env"}' "${#GITCODE_TOKEN}"
    elif [ "$TOKEN_AVAILABLE" = true ]; then
        printf '{"item":"token","status":"pass","detail":"available in current session","source":"session"}'
    else
        printf '{"item":"token","status":"fail","detail":"GITCODE_TOKEN not set"}'
    fi
}

check_binary() {
    local cmd="$1" label="${2:-$1}"
    if command -v "$cmd" >/dev/null 2>&1; then
        local ver
        ver=$("$cmd" --version 2>&1 | head -1 || true)
        printf '{"item":"%s","status":"pass","detail":"%s"}' \
            "$(json_escape "$label")" "$(json_escape "$ver")"
    else
        printf '{"item":"%s","status":"fail","detail":"%s not found"}' \
            "$(json_escape "$label")" "$(json_escape "$cmd")"
    fi
}

check_tmp() {
    local candidates=() candidate probe selected=""
    if [ "$EXTENDED_PREFLIGHT" = true ]; then
        [ -n "${ISSUE_HANDLER_TMP_DIR:-}" ] && candidates+=("$ISSUE_HANDLER_TMP_DIR")
        [ -n "${TMPDIR:-}" ] && candidates+=("$TMPDIR")
        candidates+=("/tmp" "$WORK_DIR/.cannbot/gitcode-issue-handler/tmp")
    else
        # Preserve the original no-argument contract for existing callers.
        candidates+=("/tmp")
    fi

    for candidate in "${candidates[@]}"; do
        [ -n "$candidate" ] || continue
        mkdir -p "$candidate" >/dev/null 2>&1 || continue
        probe="$candidate/.gitcode-issue-handler-write-test-$$"
        if : >"$probe" 2>/dev/null; then
            rm -f "$probe"
            selected="$candidate"
            break
        fi
    done

    if [ -n "$selected" ]; then
        if [ "$EXTENDED_PREFLIGHT" = true ]; then
            printf '{"item":"tmp","status":"pass","detail":"writable fallback selected","selected_path":"%s"}' \
                "$(json_escape "$selected")"
        else
            printf '{"item":"tmp","status":"pass","detail":"writable"}'
        fi
    else
        printf '{"item":"tmp","status":"fail","detail":"no writable temp root found"}'
    fi
}

check_git_author() {
    if [ "$SKIP_GIT_AUTHOR" = true ]; then
        printf '{"item":"git_author","status":"skip","detail":"--skip-git-author"}'
        return
    fi
    if ! command -v git >/dev/null 2>&1; then
        printf '{"item":"git_author","status":"skip","detail":"git unavailable; author not inspected"}'
        return
    fi
    local name email source

    # A repository-local identity takes precedence over the global identity.
    # The previous implementation only inspected --global, which incorrectly
    # stopped repositories that intentionally configure their author locally.
    name=$(git -C "$WORK_DIR" config --local user.name 2>/dev/null || true)
    email=$(git -C "$WORK_DIR" config --local user.email 2>/dev/null || true)
    source="local"
    if [ -z "$name" ] || [ -z "$email" ]; then
        name=$(git -C "$WORK_DIR" config --global user.name 2>/dev/null || true)
        email=$(git -C "$WORK_DIR" config --global user.email 2>/dev/null || true)
        source="global"
    fi
    if [ -n "$name" ] && [ -n "$email" ]; then
        printf '{"item":"git_author","status":"pass","detail":"%s <%s>","source":"%s"}' \
            "$(json_escape "$name")" "$(json_escape "$email")" "$(json_escape "$source")"
    else
        printf '{"item":"git_author","status":"fail","detail":"local/global user.name/email not configured"}'
    fi
}

# --- Run checks ---

RESULTS="["
RESULT_COUNT=0
add_result() {
    local result="$1"
    if [ "$RESULT_COUNT" -gt 0 ]; then
        RESULTS+=","
    fi
    RESULTS+="$result"
    RESULT_COUNT=$((RESULT_COUNT + 1))
}

[ "$WANT_TOKEN" = true ] && add_result "$(check_token)"
[ "$WANT_GIT" = true ] && add_result "$(check_binary git)"
[ "$WANT_CURL" = true ] && add_result "$(check_binary curl)"
[ "$WANT_PYTHON" = true ] && add_result "$(check_binary python3 python3)"
[ "$WANT_TMP" = true ] && add_result "$(check_tmp)"
[ "$WANT_GIT_AUTHOR" = true ] && add_result "$(check_git_author)"
RESULTS+="]"

# Build a machine-readable routing summary without relying on Python. This is
# important because Python itself is one of the tools being checked.
FAIL_COUNT=0
REMAINDER="$RESULTS"
while [[ "$REMAINDER" == *'"status":"fail"'* ]]; do
    FAIL_COUNT=$((FAIL_COUNT + 1))
    REMAINDER="${REMAINDER#*\"status\":\"fail\"}"
done
ALL_COUNT=$RESULT_COUNT
PASS_COUNT=$((ALL_COUNT - FAIL_COUNT))

NEEDS_USER=()
BLOCKERS=()
[[ "$RESULTS" == *'"item":"token","status":"fail"'* ]] && NEEDS_USER+=("token")
[[ "$RESULTS" == *'"item":"git_author","status":"fail"'* ]] && NEEDS_USER+=("git_author")
[[ "$RESULTS" == *'"item":"git","status":"fail"'* ]] && BLOCKERS+=("git")
[[ "$RESULTS" == *'"item":"curl","status":"fail"'* ]] && BLOCKERS+=("curl")
[[ "$RESULTS" == *'"item":"python3","status":"fail"'* ]] && BLOCKERS+=("python3")
[[ "$RESULTS" == *'"item":"tmp","status":"fail"'* ]] && BLOCKERS+=("tmp")

json_array() {
    local first=true item
    printf '['
    for item in "$@"; do
        if [ "$first" = false ]; then printf ','; fi
        printf '"%s"' "$item"
        first=false
    done
    printf ']'
}

if [ "${#NEEDS_USER[@]}" -gt 0 ] && [ "${#BLOCKERS[@]}" -gt 0 ]; then
    ACTION="request_inputs_and_report_blockers"
elif [ "${#NEEDS_USER[@]}" -gt 0 ]; then
    ACTION="request_inputs"
elif [ "${#BLOCKERS[@]}" -gt 0 ]; then
    ACTION="report_blockers"
else
    ACTION="continue"
fi

printf '{"ready":%s,"action":"%s","requested_checks":"%s","needs_user":' \
    "$([ "$FAIL_COUNT" -eq 0 ] && printf true || printf false)" "$ACTION" \
    "$(json_escape "$REQUESTED_CHECKS")"
json_array "${NEEDS_USER[@]}"
printf ',"blockers":'
json_array "${BLOCKERS[@]}"
printf ',"results":%s,"summary":{"pass":%d,"fail":%d,"total":%d}}\n' \
    "$RESULTS" "$PASS_COUNT" "$FAIL_COUNT" "$ALL_COUNT"

if [ "$FAIL_COUNT" -eq 0 ]; then
    exit 0
fi
exit 1
