#!/usr/bin/env bash
# =============================================================================
# ATB CSV Tester - TDD 测试
# 验证 SKILL.md 和 scripts/run-csv-tests.sh 的质量
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/../scripts/run-csv-tests.sh"
SKILL_MD_PATH="$SCRIPT_DIR/../SKILL.md"

if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "ERROR: Script not found at $SCRIPT_PATH"
    exit 1
fi

if [[ ! -f "$SKILL_MD_PATH" ]]; then
    echo "ERROR: SKILL.md not found at $SKILL_MD_PATH"
    exit 1
fi

PASS_COUNT=0
FAIL_COUNT=0
RESULTS=()

assert_match_in_file() {
    local name="$1"
    local file="$2"
    local pattern="$3"
    local should_match="${4:-true}"

    if grep -qP "$pattern" "$file"; then
        matched=true
    else
        matched=false
    fi

    if [[ "$matched" == "$should_match" ]]; then
        PASS_COUNT=$((PASS_COUNT + 1))
        RESULTS+=("  [PASS] $name")
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        RESULTS+=("  [FAIL] $name")
        if [[ "$should_match" == "true" ]]; then
            RESULTS+=("         Expected match for pattern: $pattern")
        else
            RESULTS+=("         Expected NO match for pattern: $pattern")
        fi
    fi
}

assert_true() {
    local name="$1"
    local value="$2"

    if [[ "$value" == "true" ]] || [[ "$value" == "0" ]]; then
        PASS_COUNT=$((PASS_COUNT + 1))
        RESULTS+=("  [PASS] $name")
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        RESULTS+=("  [FAIL] $name")
    fi
}

echo ""
echo "========================================"
echo "  TDD Tests for atb-csv-tester"
echo "========================================"

echo ""
echo "--- SKILL.md Documentation Quality ---"
assert_match_in_file "Has YAML frontmatter" "$SKILL_MD_PATH" '^---\s*$'
assert_match_in_file "Declares name" "$SKILL_MD_PATH" 'name:\s*\S+'
assert_match_in_file "Declares description" "$SKILL_MD_PATH" 'description:\s*\S+'
assert_match_in_file "Declares allowed-tools" "$SKILL_MD_PATH" 'allowed-tools:'
assert_match_in_file "Declares keywords" "$SKILL_MD_PATH" 'keywords:'
assert_match_in_file "Declares metadata" "$SKILL_MD_PATH" 'metadata:'
assert_match_in_file "Contains prerequisite section" "$SKILL_MD_PATH" '(前置条件|Prerequisites)'
assert_match_in_file "Contains execution steps" "$SKILL_MD_PATH" '(执行步骤|第\s*\d+\s*步|Step\s*\d+)'
assert_match_in_file "Contains CSV format section" "$SKILL_MD_PATH" '(CSV 文件格式|CSV File Format)'
assert_match_in_file "Contains troubleshooting section" "$SKILL_MD_PATH" '(故障排查|Troubleshooting)'
assert_match_in_file "References run-csv-tests.sh" "$SKILL_MD_PATH" 'run-csv-tests\.sh'
assert_match_in_file "Mentions Docker" "$SKILL_MD_PATH" 'Docker|docker'
assert_match_in_file "Mentions CANN" "$SKILL_MD_PATH" 'CANN'
assert_match_in_file "Mentions libatb_test_framework" "$SKILL_MD_PATH" 'libatb_test_framework'
assert_match_in_file "Mentions linear.csv" "$SKILL_MD_PATH" 'linear\.csv'

echo ""
echo "--- Script Structure & Parameters ---"
assert_match_in_file "Accepts --atb-repo-path" "$SCRIPT_PATH" '\-\-atb-repo-path'
assert_match_in_file "Accepts --csv-file" "$SCRIPT_PATH" '\-\-csv-file'
assert_match_in_file "Accepts --case-range" "$SCRIPT_PATH" '\-\-case-range'
assert_match_in_file "Accepts --docker-name" "$SCRIPT_PATH" '\-\-docker-name'
assert_match_in_file "Has help option" "$SCRIPT_PATH" '(\-h|\-\-help)'
assert_match_in_file "Has --dry-run option" "$SCRIPT_PATH" '\-\-dry-run'
assert_match_in_file "Uses set -eo pipefail" "$SCRIPT_PATH" 'set\s+-eo\s+pipefail'

echo ""
echo "--- Script Functionality ---"
assert_match_in_file "Checks Docker running" "$SCRIPT_PATH" 'docker ps'
assert_match_in_file "Checks container running" "$SCRIPT_PATH" 'docker exec'
assert_match_in_file "Checks ATB repo path" "$SCRIPT_PATH" 'ATB_REPO_PATH'
assert_match_in_file "Checks CSV file exists" "$SCRIPT_PATH" 'CSV_FILE_PATH'
assert_match_in_file "Checks libatb_test_framework" "$SCRIPT_PATH" 'libatb_test_framework'
assert_match_in_file "Sources CANN set_env.sh" "$SCRIPT_PATH" 'set_env\.sh'
assert_match_in_file "Sources ATB set_env.sh" "$SCRIPT_PATH" 'output/atb/set_env\.sh'
assert_match_in_file "Executes atb_csv_ops_test.py" "$SCRIPT_PATH" 'atb_csv_ops_test\.py'
assert_match_in_file "Supports case range -n" "$SCRIPT_PATH" '\-n.*CASE_RANGE'
assert_match_in_file "Has log_info function" "$SCRIPT_PATH" 'log_info\(\)'
assert_match_in_file "Has log_err function" "$SCRIPT_PATH" 'log_err\(\)'
assert_match_in_file "Has log_ok function" "$SCRIPT_PATH" 'log_ok\(\)'
assert_match_in_file "Shows elapsed time" "$SCRIPT_PATH" 'END_TIME|duration'

echo ""
echo "--- Parameter Validation ---"
assert_match_in_file "Validates --atb-repo-path required" "$SCRIPT_PATH" 'ATB_REPO_PATH'
assert_match_in_file "Validates --csv-file required" "$SCRIPT_PATH" 'CSV_FILE_PATH'
assert_match_in_file "Strips trailing slash" "$SCRIPT_PATH" 'ATB_REPO_PATH.*%/'

echo ""
echo "--- CSV Format Validation ---"
assert_match_in_file "Checks CSV column count" "$SCRIPT_PATH" 'COLUMN_COUNT'
assert_match_in_file "Validates pipe delimiter" "$SCRIPT_PATH" "'\\|'"

echo ""
echo "--- Error Handling ---"
assert_match_in_file "Handles Docker not running" "$SCRIPT_PATH" 'Docker.*未运行'
assert_match_in_file "Handles container not running" "$SCRIPT_PATH" '容器.*未运行'
assert_match_in_file "Handles ATB repo not found" "$SCRIPT_PATH" 'ATB.*目录不存在'
assert_match_in_file "Handles CSV file not found" "$SCRIPT_PATH" 'CSV.*文件不存在'
assert_match_in_file "Handles test framework not built" "$SCRIPT_PATH" 'libatb_test_framework'

echo ""
echo "--- CANN Detection ---"
assert_match_in_file "Detects ASCEND_TOOLKIT_HOME" "$SCRIPT_PATH" 'ASCEND_TOOLKIT_HOME'
assert_match_in_file "Detects ASCEND_HOME_PATH" "$SCRIPT_PATH" 'ASCEND_HOME_PATH'
assert_match_in_file "Falls back to common paths" "$SCRIPT_PATH" '/usr/local/Ascend'

echo ""
echo "--- Security Checks ---"
assert_match_in_file "No hardcoded credentials" "$SCRIPT_PATH" '(?i)(password|token|secret|apikey|api_key)\s*=' "false"
assert_match_in_file "Uses parameterized paths" "$SCRIPT_PATH" '\$ATB_REPO_PATH|\$CSV_FILE_PATH'

echo ""
echo "--- Idempotency ---"
assert_match_in_file "Checks existing files" "$SCRIPT_PATH" '\[ ! -f|\[ ! -d'
assert_match_in_file "Handles missing files gracefully" "$SCRIPT_PATH" 'log_err|exit 1'

echo ""
echo "--- Encoding Check ---"
BYTES=$(hexdump -C "$SKILL_MD_PATH" | head -1)
if echo "$BYTES" | grep -q "ef bb bf"; then
    assert_true "SKILL.md is UTF-8 without BOM" "false"
else
    assert_true "SKILL.md is UTF-8 without BOM" "true"
fi

echo ""
echo "========================================"
echo "  Test Results Summary"
echo "========================================"
echo "  PASS: $PASS_COUNT"
echo "  FAIL: $FAIL_COUNT"
echo "  TOTAL: $((PASS_COUNT + FAIL_COUNT))"
echo ""

for r in "${RESULTS[@]}"; do
    if echo "$r" | grep -q "PASS"; then
        echo -e "\033[0;32m$r\033[0m"
    elif echo "$r" | grep -q "FAIL"; then
        echo -e "\033[0;31m$r\033[0m"
    else
        echo "$r"
    fi
done

echo ""
if [[ "$FAIL_COUNT" -gt 0 ]]; then
    echo -e "\033[0;31m[RED] $FAIL_COUNT test(s) FAILED\033[0m"
    exit 1
else
    echo -e "\033[0;32m[GREEN] All tests PASSED\033[0m"
    exit 0
fi
