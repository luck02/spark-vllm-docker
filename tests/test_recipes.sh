#!/bin/bash
#
# test_recipes.sh - Integration tests for run-recipe.py and launch-cluster.sh
#
# These tests use --dry-run mode to verify compatibility without actually
# running containers. Suitable for CI/CD pipelines.
#
# Usage:
#   ./tests/test_recipes.sh          # Run all tests
#   ./tests/test_recipes.sh -v       # Verbose output
#

set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VERBOSE="${1:-}"

# Load expected commands for README verification
source "$SCRIPT_DIR/expected_commands.sh"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Helper functions
log_test() {
    echo -e "${YELLOW}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    TESTS_FAILED=$((TESTS_FAILED + 1))
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
}

log_verbose() {
    if [[ "$VERBOSE" == "-v" ]]; then
        echo "       $1"
    fi
}

# Check prerequisites
check_prerequisites() {
    log_test "Checking prerequisites..."
    
    if ! command -v python3 &> /dev/null; then
        log_fail "python3 not found"
        exit 1
    fi
    
    # Check Python version
    python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ $(echo "$python_version < 3.10" | bc -l) -eq 1 ]]; then
        log_fail "Python 3.10+ required, found $python_version"
        exit 1
    fi
    
    # Check PyYAML
    if ! python3 -c "import yaml" 2>/dev/null; then
        log_fail "PyYAML not installed"
        exit 1
    fi
    
    log_pass "Prerequisites OK (Python $python_version with PyYAML)"
}

# Test: run-recipe.py exists and is executable
test_run_recipe_exists() {
    log_test "run-recipe.py exists and is executable"
    
    if [[ -x "$PROJECT_DIR/run-recipe.py" ]]; then
        log_pass "run-recipe.py is executable"
    else
        log_fail "run-recipe.py not found or not executable"
    fi
}

# Test: launch-cluster.sh exists and is executable
test_launch_cluster_exists() {
    log_test "launch-cluster.sh exists and is executable"
    
    if [[ -x "$PROJECT_DIR/launch-cluster.sh" ]]; then
        log_pass "launch-cluster.sh is executable"
    else
        log_fail "launch-cluster.sh not found or not executable"
    fi
}

# Test: run-recipe.py --list works
test_list_recipes() {
    log_test "run-recipe.py --list"
    
    output=$("$PROJECT_DIR/run-recipe.py" --list 2>&1)
    
    if [[ $? -eq 0 ]] && echo "$output" | grep -q "Available recipes"; then
        log_pass "--list shows available recipes"
        log_verbose "Found recipes in output"
    else
        log_fail "--list failed or no recipes found"
        log_verbose "$output"
    fi
}

# Test: All recipes have required recipe_version field
test_recipe_version_required() {
    log_test "All recipes have required recipe_version field"
    
    local all_valid=true
    for recipe in "$PROJECT_DIR/recipes/"*.yaml; do
        if [[ -f "$recipe" ]]; then
            recipe_name=$(basename "$recipe")
            if ! grep -q "^recipe_version:" "$recipe"; then
                log_verbose "$recipe_name missing recipe_version"
                all_valid=false
            fi
        fi
    done
    
    if [[ "$all_valid" == "true" ]]; then
        log_pass "All recipes have recipe_version field"
    else
        log_fail "Some recipes missing recipe_version field"
    fi
}

# Test: All recipes load without errors
test_all_recipes_load() {
    log_test "All recipes load without errors"
    
    local all_valid=true
    for recipe in "$PROJECT_DIR/recipes/"*.yaml; do
        if [[ -f "$recipe" ]]; then
            recipe_name=$(basename "$recipe" .yaml)
            # Try to load recipe with --dry-run (will fail early if recipe is invalid)
            if ! "$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo 2>&1 | grep -q "Error:"; then
                log_verbose "$recipe_name loads OK"
            else
                log_verbose "$recipe_name failed to load"
                all_valid=false
            fi
        fi
    done
    
    if [[ "$all_valid" == "true" ]]; then
        log_pass "All recipes load successfully"
    else
        log_fail "Some recipes failed to load"
    fi
}

# Test: Dry-run generates valid launch script
test_dry_run_generates_script() {
    log_test "Dry-run generates valid launch script"
    
    # Find first available recipe
    first_recipe=$(ls "$PROJECT_DIR/recipes/"*.yaml 2>/dev/null | head -1)
    if [[ -z "$first_recipe" ]]; then
        log_skip "No recipes found"
        return
    fi
    
    recipe_name=$(basename "$first_recipe" .yaml)
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo 2>&1)
    
    if echo "$output" | grep -q "#!/bin/bash" && echo "$output" | grep -q "vllm serve"; then
        log_pass "Dry-run generates bash script with vllm serve command"
    else
        log_fail "Dry-run output doesn't contain expected content"
        log_verbose "$output"
    fi
}

# Test: Solo mode sets tensor_parallel=1
test_solo_mode_tp1() {
    log_test "Solo mode sets tensor_parallel=1"
    
    first_recipe=$(ls "$PROJECT_DIR/recipes/"*.yaml 2>/dev/null | head -1)
    if [[ -z "$first_recipe" ]]; then
        log_skip "No recipes found"
        return
    fi
    
    recipe_name=$(basename "$first_recipe" .yaml)
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo 2>&1)
    
    # Check that -tp 1 is in the output (solo mode should set tp=1)
    if echo "$output" | grep -q "\-tp 1"; then
        log_pass "Solo mode correctly sets -tp 1"
    else
        log_fail "Solo mode did not set -tp 1"
        log_verbose "$output"
    fi
}

# Test: Solo mode removes --distributed-executor-backend ray
test_solo_mode_removes_ray() {
    log_test "Solo mode removes --distributed-executor-backend ray"
    
    first_recipe=$(ls "$PROJECT_DIR/recipes/"*.yaml 2>/dev/null | head -1)
    if [[ -z "$first_recipe" ]]; then
        log_skip "No recipes found"
        return
    fi
    
    recipe_name=$(basename "$first_recipe" .yaml)
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo 2>&1)
    
    # Check that --distributed-executor-backend is NOT in the output
    if ! echo "$output" | grep -q "\-\-distributed-executor-backend"; then
        log_pass "Solo mode correctly removes --distributed-executor-backend"
    else
        log_fail "Solo mode did not remove --distributed-executor-backend"
        log_verbose "$output"
    fi
}

# Test: Cluster mode preserves --distributed-executor-backend ray
test_cluster_mode_keeps_ray() {
    log_test "Cluster mode preserves --distributed-executor-backend ray"
    
    # Use minimax-m2-awq which explicitly has --distributed-executor-backend ray
    if [[ ! -f "$PROJECT_DIR/recipes/minimax-m2-awq.yaml" ]]; then
        log_skip "minimax-m2-awq.yaml not found"
        return
    fi
    
    output=$("$PROJECT_DIR/run-recipe.py" minimax-m2-awq --dry-run -n "192.168.1.1,192.168.1.2" 2>&1)
    
    # Check that --distributed-executor-backend IS in the output for cluster mode
    if echo "$output" | grep -q "\-\-distributed-executor-backend ray"; then
        log_pass "Cluster mode correctly preserves --distributed-executor-backend ray"
    else
        log_fail "Cluster mode did not preserve --distributed-executor-backend"
        log_verbose "$output"
    fi
}

# Test: CLI overrides work (--port)
test_cli_override_port() {
    log_test "CLI override --port works"
    
    first_recipe=$(ls "$PROJECT_DIR/recipes/"*.yaml 2>/dev/null | head -1)
    if [[ -z "$first_recipe" ]]; then
        log_skip "No recipes found"
        return
    fi
    
    recipe_name=$(basename "$first_recipe" .yaml)
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo --port 9999 2>&1)
    
    if echo "$output" | grep -q "\-\-port 9999"; then
        log_pass "--port override correctly applied"
    else
        log_fail "--port override not found in output"
        log_verbose "$output"
    fi
}

# Test: launch-cluster.sh --help works
test_launch_cluster_help() {
    log_test "launch-cluster.sh --help"
    
    output=$("$PROJECT_DIR/launch-cluster.sh" --help 2>&1 || true)
    
    if echo "$output" | grep -q "Usage:"; then
        log_pass "--help shows usage information"
    else
        log_fail "--help did not show usage"
        log_verbose "$output"
    fi
}

# Test: launch-cluster.sh references examples/ not profiles/
test_launch_cluster_examples_path() {
    log_test "launch-cluster.sh references examples/ directory"
    
    if grep -q "examples/" "$PROJECT_DIR/launch-cluster.sh"; then
        log_pass "launch-cluster.sh references examples/"
    else
        log_fail "launch-cluster.sh does not reference examples/"
    fi
    
    if grep -q "profiles/" "$PROJECT_DIR/launch-cluster.sh"; then
        log_fail "launch-cluster.sh still references profiles/"
    fi
}

# Test: Unsupported recipe version shows warning
test_unsupported_recipe_version() {
    log_test "Unsupported recipe_version shows warning"
    
    # Create a temporary recipe with unsupported version
    temp_recipe=$(mktemp)
    cat > "$temp_recipe" << 'EOF'
recipe_version: "999"
name: Test Recipe
container: test-container
command: echo "test"
EOF
    
    output=$("$PROJECT_DIR/run-recipe.py" "$temp_recipe" --dry-run --solo 2>&1)
    rm -f "$temp_recipe"
    
    if echo "$output" | grep -q "Warning.*schema version"; then
        log_pass "Unsupported recipe_version shows warning"
    else
        log_fail "No warning for unsupported recipe_version"
        log_verbose "$output"
    fi
}

# Test: Missing recipe_version fails
test_missing_recipe_version_fails() {
    log_test "Missing recipe_version field fails"
    
    # Create a temporary recipe without recipe_version
    temp_recipe=$(mktemp)
    cat > "$temp_recipe" << 'EOF'
name: Test Recipe
container: test-container
command: echo "test"
EOF
    
    output=$("$PROJECT_DIR/run-recipe.py" "$temp_recipe" --dry-run --solo 2>&1 || true)
    rm -f "$temp_recipe"
    
    if echo "$output" | grep -q "Error.*recipe_version"; then
        log_pass "Missing recipe_version correctly fails"
    else
        log_fail "Missing recipe_version did not fail as expected"
        log_verbose "$output"
    fi
}

# Test: cluster_only recipe fails in solo mode
test_cluster_only_fails_solo() {
    log_test "cluster_only recipe fails in solo mode"
    
    # Create a temporary cluster_only recipe
    temp_recipe=$(mktemp)
    cat > "$temp_recipe" << 'EOF'
recipe_version: "1"
name: Cluster Only Test
container: test-container
cluster_only: true
command: echo "test"
EOF
    
    output=$("$PROJECT_DIR/run-recipe.py" "$temp_recipe" --dry-run --solo 2>&1 || true)
    exit_code=$?
    rm -f "$temp_recipe"
    
    if echo "$output" | grep -q "requires cluster mode"; then
        log_pass "cluster_only recipe correctly fails in solo mode"
    else
        log_fail "cluster_only recipe did not fail in solo mode"
        log_verbose "$output"
    fi
}

# ==============================================================================
# Launch-cluster.sh Command Line Verification Tests
# ==============================================================================
# These tests verify that the dry-run output contains the expected
# launch-cluster.sh command line arguments matching the recipe configuration.

# Helper: Extract launch-cluster command from dry-run output
extract_launch_cmd() {
    echo "$1" | grep -A5 "launch-cluster.sh is called with:" | grep -v "launch-cluster.sh is called with:" | tr '\n' ' '
}

# Test: Solo mode generates --solo flag in launch-cluster command
test_launch_cmd_solo_flag() {
    log_test "Launch command includes --solo flag in solo mode"
    
    first_recipe=$(ls "$PROJECT_DIR/recipes/"*.yaml 2>/dev/null | head -1)
    if [[ -z "$first_recipe" ]]; then
        log_skip "No recipes found"
        return
    fi
    
    recipe_name=$(basename "$first_recipe" .yaml)
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo 2>&1)
    launch_cmd=$(extract_launch_cmd "$output")
    
    if echo "$launch_cmd" | grep -q "\-\-solo"; then
        log_pass "Launch command includes --solo flag"
    else
        log_fail "Launch command missing --solo flag"
        log_verbose "Launch cmd: $launch_cmd"
    fi
}

# Test: Cluster mode generates -n flag with nodes
test_launch_cmd_nodes_flag() {
    log_test "Launch command includes -n flag with nodes in cluster mode"
    
    output=$("$PROJECT_DIR/run-recipe.py" minimax-m2-awq --dry-run -n "10.0.0.1,10.0.0.2" 2>&1)
    launch_cmd=$(extract_launch_cmd "$output")
    
    if echo "$launch_cmd" | grep -q "\-n 10.0.0.1,10.0.0.2"; then
        log_pass "Launch command includes -n with correct nodes"
    else
        log_fail "Launch command missing or incorrect -n flag"
        log_verbose "Launch cmd: $launch_cmd"
    fi
}

# Test: Container image from recipe is passed to launch-cluster
test_launch_cmd_container_image() {
    log_test "Launch command includes correct container image (-t)"
    
    # Use openai-gpt-oss-120b which has a specific container name
    if [[ ! -f "$PROJECT_DIR/recipes/openai-gpt-oss-120b.yaml" ]]; then
        log_skip "openai-gpt-oss-120b.yaml not found"
        return
    fi
    
    output=$("$PROJECT_DIR/run-recipe.py" openai-gpt-oss-120b --dry-run --solo 2>&1)
    launch_cmd=$(extract_launch_cmd "$output")
    
    # Check the container is vllm-node-mxfp4 (from the recipe)
    if echo "$launch_cmd" | grep -q "\-t vllm-node-mxfp4"; then
        log_pass "Launch command includes correct container image"
    else
        log_fail "Launch command has wrong container image"
        log_verbose "Launch cmd: $launch_cmd"
    fi
}

# Test: Mods from recipe are passed as --apply-mod
test_launch_cmd_mods() {
    log_test "Launch command includes --apply-mod for recipe mods"
    
    # Use glm-4.7-flash-awq which has a mod
    if [[ ! -f "$PROJECT_DIR/recipes/glm-4.7-flash-awq.yaml" ]]; then
        log_skip "glm-4.7-flash-awq.yaml not found"
        return
    fi
    
    output=$("$PROJECT_DIR/run-recipe.py" glm-4.7-flash-awq --dry-run --solo 2>&1)
    launch_cmd=$(extract_launch_cmd "$output")
    
    if echo "$launch_cmd" | grep -q "\-\-apply-mod"; then
        log_pass "Launch command includes --apply-mod for mods"
    else
        log_fail "Launch command missing --apply-mod"
        log_verbose "Launch cmd: $launch_cmd"
    fi
}

# Test: Daemon mode flag is passed through
test_launch_cmd_daemon_flag() {
    log_test "Launch command includes -d flag in daemon mode"
    
    first_recipe=$(ls "$PROJECT_DIR/recipes/"*.yaml 2>/dev/null | head -1)
    if [[ -z "$first_recipe" ]]; then
        log_skip "No recipes found"
        return
    fi
    
    recipe_name=$(basename "$first_recipe" .yaml)
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo -d 2>&1)
    launch_cmd=$(extract_launch_cmd "$output")
    
    if echo "$launch_cmd" | grep -q "\-d"; then
        log_pass "Launch command includes -d flag"
    else
        log_fail "Launch command missing -d flag"
        log_verbose "Launch cmd: $launch_cmd"
    fi
}

# Test: NCCL debug level is passed through
test_launch_cmd_nccl_debug() {
    log_test "Launch command includes --nccl-debug when specified"
    
    first_recipe=$(ls "$PROJECT_DIR/recipes/"*.yaml 2>/dev/null | head -1)
    if [[ -z "$first_recipe" ]]; then
        log_skip "No recipes found"
        return
    fi
    
    recipe_name=$(basename "$first_recipe" .yaml)
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo --nccl-debug INFO 2>&1)
    launch_cmd=$(extract_launch_cmd "$output")
    
    if echo "$launch_cmd" | grep -q "\-\-nccl-debug INFO"; then
        log_pass "Launch command includes --nccl-debug INFO"
    else
        log_fail "Launch command missing --nccl-debug"
        log_verbose "Launch cmd: $launch_cmd"
    fi
}

# Test: --launch-script is always included
test_launch_cmd_launch_script() {
    log_test "Launch command includes --launch-script"
    
    first_recipe=$(ls "$PROJECT_DIR/recipes/"*.yaml 2>/dev/null | head -1)
    if [[ -z "$first_recipe" ]]; then
        log_skip "No recipes found"
        return
    fi
    
    recipe_name=$(basename "$first_recipe" .yaml)
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo 2>&1)
    launch_cmd=$(extract_launch_cmd "$output")
    
    if echo "$launch_cmd" | grep -q "\-\-launch-script"; then
        log_pass "Launch command includes --launch-script"
    else
        log_fail "Launch command missing --launch-script"
        log_verbose "Launch cmd: $launch_cmd"
    fi
}

# Test: Container override (-t CLI) takes precedence
test_launch_cmd_container_override() {
    log_test "CLI container override (-t) takes precedence"
    
    first_recipe=$(ls "$PROJECT_DIR/recipes/"*.yaml 2>/dev/null | head -1)
    if [[ -z "$first_recipe" ]]; then
        log_skip "No recipes found"
        return
    fi
    
    recipe_name=$(basename "$first_recipe" .yaml)
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo -t my-custom-image 2>&1)
    launch_cmd=$(extract_launch_cmd "$output")
    
    if echo "$launch_cmd" | grep -q "\-t my-custom-image"; then
        log_pass "Container override correctly applied"
    else
        log_fail "Container override not applied"
        log_verbose "Launch cmd: $launch_cmd"
    fi
}

# Test: Cluster mode does NOT include --solo flag
test_launch_cmd_no_solo_in_cluster() {
    log_test "Launch command does NOT include --solo in cluster mode"
    
    output=$("$PROJECT_DIR/run-recipe.py" minimax-m2-awq --dry-run -n "10.0.0.1,10.0.0.2" 2>&1)
    launch_cmd=$(extract_launch_cmd "$output")
    
    if echo "$launch_cmd" | grep -qv "\-\-solo" || ! echo "$launch_cmd" | grep -q "\-\-solo"; then
        log_pass "Cluster mode correctly omits --solo flag"
    else
        log_fail "Cluster mode incorrectly includes --solo flag"
        log_verbose "Launch cmd: $launch_cmd"
    fi
}

# ==============================================================================
# README Documentation Verification Tests
# ==============================================================================
# These tests verify that recipe dry-run output matches the expected commands
# documented in README.md. Expected values are defined in expected_commands.sh

# Helper: Extract the generated launch script from dry-run output
extract_vllm_command() {
    # Extract lines between "Generated Launch Script" and "What would be executed"
    echo "$1" | sed -n '/=== Generated Launch Script ===/,/=== What would be executed ===/p' | grep -v "===" | grep -v "^#" | grep -v "^$"
}

# Helper: Verify a recipe contains all expected arguments
verify_recipe_args() {
    local recipe_name="$1"
    local expected_model="$2"
    local expected_container="$3"
    shift 3
    local expected_args=("$@")
    
    log_test "README match: $recipe_name"
    
    if [[ ! -f "$PROJECT_DIR/recipes/${recipe_name}.yaml" ]]; then
        log_skip "${recipe_name}.yaml not found"
        return
    fi
    
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run --solo 2>&1)
    vllm_cmd=$(extract_vllm_command "$output")
    launch_cmd=$(extract_launch_cmd "$output")
    
    local all_passed=true
    local missing_args=()
    
    # Check model name
    if ! echo "$vllm_cmd" | grep -q "$expected_model"; then
        missing_args+=("model: $expected_model")
        all_passed=false
    fi
    
    # Check container
    if ! echo "$launch_cmd" | grep -q "\-t $expected_container"; then
        missing_args+=("container: $expected_container")
        all_passed=false
    fi
    
    # Check each expected argument
    for arg in "${expected_args[@]}"; do
        # Handle arguments that may have slight formatting differences
        # Extract the flag and value separately for flexible matching
        local flag=$(echo "$arg" | awk '{print $1}')
        local value=$(echo "$arg" | cut -d' ' -f2-)
        
        # Use grep -F for fixed string matching (avoids -- being treated as grep options)
        if ! echo "$vllm_cmd" | grep -qF -- "$flag"; then
            missing_args+=("$arg")
            all_passed=false
        elif [[ -n "$value" ]] && [[ "$value" != "$flag" ]]; then
            # Check if value is present (might be on next line due to formatting)
            if ! echo "$vllm_cmd" | grep -qF -- "$value"; then
                missing_args+=("$arg (flag present, value mismatch)")
                all_passed=false
            fi
        fi
    done
    
    if [[ "$all_passed" == "true" ]]; then
        log_pass "README match: $recipe_name - all expected arguments present"
    else
        log_fail "README match: $recipe_name - missing arguments"
        for missing in "${missing_args[@]}"; do
            log_verbose "  Missing: $missing"
        done
        log_verbose "  vLLM command: $vllm_cmd"
    fi
}

# Test: glm-4.7-flash-awq matches README documentation
test_readme_glm_flash_awq() {
    verify_recipe_args "glm-4.7-flash-awq" \
        "$GLM_FLASH_AWQ_MODEL" \
        "$GLM_FLASH_AWQ_CONTAINER" \
        "${GLM_FLASH_AWQ_ARGS[@]}"
}

# Test: openai-gpt-oss-120b matches README documentation
test_readme_gpt_oss() {
    verify_recipe_args "openai-gpt-oss-120b" \
        "$GPT_OSS_MODEL" \
        "$GPT_OSS_CONTAINER" \
        "${GPT_OSS_ARGS[@]}"
}

# Test: minimax-m2-awq matches expected configuration
test_readme_minimax() {
    verify_recipe_args "minimax-m2-awq" \
        "$MINIMAX_MODEL" \
        "$MINIMAX_CONTAINER" \
        "${MINIMAX_ARGS[@]}"
}

# Test: glm-4.7-flash-awq includes correct mod
test_readme_glm_flash_mod() {
    log_test "README match: glm-4.7-flash-awq mod path"
    
    if [[ ! -f "$PROJECT_DIR/recipes/glm-4.7-flash-awq.yaml" ]]; then
        log_skip "glm-4.7-flash-awq.yaml not found"
        return
    fi
    
    output=$("$PROJECT_DIR/run-recipe.py" glm-4.7-flash-awq --dry-run --solo 2>&1)
    launch_cmd=$(extract_launch_cmd "$output")
    
    if echo "$launch_cmd" | grep -q "$GLM_FLASH_AWQ_MOD"; then
        log_pass "README match: glm-4.7-flash-awq has correct mod path"
    else
        log_fail "README match: glm-4.7-flash-awq missing expected mod: $GLM_FLASH_AWQ_MOD"
        log_verbose "Launch cmd: $launch_cmd"
    fi
}

# Helper: Verify cluster mode specific arguments
verify_cluster_args() {
    local recipe_name="$1"
    local expected_tp="$2"
    shift 2
    local expected_args=("$@")
    
    log_test "README match (cluster): $recipe_name"
    
    if [[ ! -f "$PROJECT_DIR/recipes/${recipe_name}.yaml" ]]; then
        log_skip "${recipe_name}.yaml not found"
        return
    fi
    
    # Use fake nodes for cluster mode
    output=$("$PROJECT_DIR/run-recipe.py" "$recipe_name" --dry-run -n "10.0.0.1,10.0.0.2" 2>&1)
    vllm_cmd=$(extract_vllm_command "$output")
    
    local all_passed=true
    local missing_args=()
    
    # Check tensor parallel
    if ! echo "$vllm_cmd" | grep -qE "(--tensor-parallel-size|-tp) $expected_tp"; then
        missing_args+=("tensor_parallel: $expected_tp")
        all_passed=false
    fi
    
    # Check cluster-specific arguments
    for arg in "${expected_args[@]}"; do
        if ! echo "$vllm_cmd" | grep -qF -- "$arg"; then
            missing_args+=("$arg")
            all_passed=false
        fi
    done
    
    if [[ "$all_passed" == "true" ]]; then
        log_pass "README match (cluster): $recipe_name - cluster args correct"
    else
        log_fail "README match (cluster): $recipe_name - missing cluster arguments"
        for missing in "${missing_args[@]}"; do
            log_verbose "  Missing: $missing"
        done
        log_verbose "  vLLM command: $vllm_cmd"
    fi
}

# Test: openai-gpt-oss-120b cluster mode has correct tensor_parallel and ray backend
test_readme_gpt_oss_cluster() {
    verify_cluster_args "openai-gpt-oss-120b" \
        "$GPT_OSS_CLUSTER_TP" \
        "${GPT_OSS_CLUSTER_ARGS[@]}"
}

# Test: minimax-m2-awq cluster mode has correct tensor_parallel and ray backend
test_readme_minimax_cluster() {
    verify_cluster_args "minimax-m2-awq" \
        "$MINIMAX_CLUSTER_TP" \
        "${MINIMAX_CLUSTER_ARGS[@]}"
}

# Test: glm-4.7-flash-awq cluster mode stays at tp=1 (single GPU model)
test_readme_glm_flash_cluster() {
    log_test "README match (cluster): glm-4.7-flash-awq stays tp=1"
    
    if [[ ! -f "$PROJECT_DIR/recipes/glm-4.7-flash-awq.yaml" ]]; then
        log_skip "glm-4.7-flash-awq.yaml not found"
        return
    fi
    
    # Even in cluster mode, this model uses tp=1
    output=$("$PROJECT_DIR/run-recipe.py" glm-4.7-flash-awq --dry-run -n "10.0.0.1,10.0.0.2" 2>&1)
    vllm_cmd=$(extract_vllm_command "$output")
    
    if echo "$vllm_cmd" | grep -qE "(--tensor-parallel-size|-tp) 1"; then
        log_pass "README match (cluster): glm-4.7-flash-awq correctly keeps tp=1"
    else
        log_fail "README match (cluster): glm-4.7-flash-awq should have tp=1"
        log_verbose "  vLLM command: $vllm_cmd"
    fi
}

# Run all tests
main() {
    echo "=============================================="
    echo "  run-recipe.py Integration Tests"
    echo "=============================================="
    echo ""
    
    cd "$PROJECT_DIR"
    
    check_prerequisites
    echo ""
    
    # File existence tests
    test_run_recipe_exists
    test_launch_cluster_exists
    echo ""
    
    # Basic functionality tests
    test_list_recipes
    test_recipe_version_required
    test_all_recipes_load
    echo ""
    
    # Dry-run tests
    test_dry_run_generates_script
    test_solo_mode_tp1
    test_solo_mode_removes_ray
    test_cluster_mode_keeps_ray
    test_cli_override_port
    echo ""
    
    # launch-cluster.sh command line verification tests
    echo "--- Launch Command Verification ---"
    test_launch_cmd_solo_flag
    test_launch_cmd_nodes_flag
    test_launch_cmd_container_image
    test_launch_cmd_mods
    test_launch_cmd_daemon_flag
    test_launch_cmd_nccl_debug
    test_launch_cmd_launch_script
    test_launch_cmd_container_override
    test_launch_cmd_no_solo_in_cluster
    echo ""
    
    # README documentation verification tests
    echo "--- README Documentation Verification (Solo Mode) ---"
    test_readme_glm_flash_awq
    test_readme_gpt_oss
    test_readme_minimax
    test_readme_glm_flash_mod
    echo ""
    
    # Cluster mode documentation verification tests
    echo "--- README Documentation Verification (Cluster Mode) ---"
    test_readme_gpt_oss_cluster
    test_readme_minimax_cluster
    test_readme_glm_flash_cluster
    echo ""
    
    # launch-cluster.sh tests
    test_launch_cluster_help
    test_launch_cluster_examples_path
    echo ""
    
    # Validation tests
    test_unsupported_recipe_version
    test_missing_recipe_version_fails
    test_cluster_only_fails_solo
    echo ""
    
    # Summary
    echo "=============================================="
    echo "  Test Summary"
    echo "=============================================="
    echo -e "  ${GREEN}Passed:${NC}  $TESTS_PASSED"
    echo -e "  ${RED}Failed:${NC}  $TESTS_FAILED"
    echo -e "  ${YELLOW}Skipped:${NC} $TESTS_SKIPPED"
    echo "=============================================="
    
    if [[ $TESTS_FAILED -gt 0 ]]; then
        exit 1
    fi
    exit 0
}

main "$@"
