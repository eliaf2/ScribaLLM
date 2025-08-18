#!/bin/zsh

# Compilation script for ScribaLLM repository
# Usage: ./run.sh [OPTIONS]
# Options:
#   -dev, --dev         Run in development mode
#   -d, --detached      Run containers in detached mode
#   -h, --help          Show this help message
#   --no-cache          Build without using cache
#   -v, --verbose       Enable verbose output
#   --rebuild           Force rebuild images
#   --init              Reinitialize containers (fresh start)

set -euo pipefail

# Script configuration
readonly SCRIPT_NAME=$(basename "$0")
readonly SCRIPT_DIR=$(dirname "$(realpath "$0")")
readonly PROJECT_NAME="scriballm"
readonly COMPOSE_FILE="docker-compose.yaml"
readonly COMPOSE_DEV_FILE="docker-compose.dev.yaml"

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Globals
DEV_MODE=false
DETACHED_MODE=false
VERBOSE=false
NO_CACHE=false
REBUILD_IMAGES=false
INIT_CONTAINERS=false

# Print helpers
print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_debug() { {[[ "$VERBOSE" == true ]] && echo -e "${BLUE}[DEBUG]${NC} $1"; } }

# Help
show_help() {
    cat << EOF
$SCRIPT_NAME - Production compilation script for $PROJECT_NAME

USAGE:
    $SCRIPT_NAME [OPTIONS]

OPTIONS:
    -dev, --dev         Run in development mode using $COMPOSE_DEV_FILE
    -d, --detached      Run containers in detached mode
    -h, --help          Show this help message and exit
    --no-cache          Build without using Docker cache
    -v, --verbose       Enable verbose output and logging
    --rebuild           Force rebuild of Docker images
    --init              Reinitialize containers from images (remove and recreate)

EXAMPLES:
    $SCRIPT_NAME                  # Normal production start
    $SCRIPT_NAME --dev            # Development start
    $SCRIPT_NAME --rebuild        # Rebuild images before starting
    $SCRIPT_NAME --init           # Recreate containers fresh
EOF
}

# Parse CLI args
while [[ $# -gt 0 ]]; do
    case $1 in
        -dev|--dev)
            DEV_MODE=true
            ;;
        -d|--detached)
            DETACHED_MODE=true
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --no-cache)
            NO_CACHE=true
            ;;
        -v|--verbose)
            VERBOSE=true
            ;;
        --rebuild) 
            REBUILD_IMAGES=true
            INIT_CONTAINERS=true
            ;;
        --init)
            INIT_CONTAINERS=true
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use -h or --help for usage information."
            exit 1
            ;;
    esac
    shift
done

# Dependency checks
check_dependencies() {
    local missing=()
    if ! command -v docker &>/dev/null; then missing+=("docker"); fi
    if ! command -v docker-compose &>/dev/null && ! docker compose version &>/dev/null; then
        missing+=("docker-compose")
    fi
    if [[ ${#missing[@]} -ne 0 ]]; then
        print_error "Missing: ${missing[*]}"
        exit 1
    fi
    print_debug "All dependencies are satisfied."
}

check_docker_daemon() {
    if ! docker info &>/dev/null; then
        print_error "Docker daemon is not running."
        exit 1
    fi
    print_debug "Docker daemon is running."
}

validate_compose_files() {
    local file="$COMPOSE_FILE"
    [[ "$DEV_MODE" == true ]] && file="$COMPOSE_DEV_FILE"
    [[ ! -f "$file" ]] && { print_error "Missing compose file: $file"; exit 1; }
    print_debug "Validated compose file: $file"
}

get_compose_command() {
    if command -v docker-compose &>/dev/null; then echo "docker-compose"
    else echo "docker compose"; fi
}

containers_running() {
    local file="$1"
    local cmd=$(get_compose_command)
    local args=("-f" "$file" "-p" "$PROJECT_NAME")
    local running=$($cmd "${args[@]}" ps --services --filter "status=running" | wc -l | tr -d ' ')
    local defined=$($cmd "${args[@]}" config --services | wc -l | tr -d ' ')
    [[ "$running" -eq "$defined" && "$defined" -gt 0 ]]
}

# Compile
compile_repository() {
    local file="$COMPOSE_FILE"
    [[ "$DEV_MODE" == true ]] && file="$COMPOSE_DEV_FILE"

    local cmd=$(get_compose_command)
    local args=("-f" "$file" "-p" "$PROJECT_NAME")

    print_info "Using compose file: $file"

    # If --init: tear down existing containers
    if [[ "$INIT_CONTAINERS" == true ]]; then
        print_info "Reinitializing containers..."
        $cmd "${args[@]}" down -v || true
    fi

    # If --rebuild: rebuild images explicitly
    if [[ "$REBUILD_IMAGES" == true ]]; then
        print_info "Rebuilding images..."
        build_opts=("build")
        [[ "$NO_CACHE" == true ]] && build_opts+=("--no-cache")
        $cmd "${args[@]}" "${build_opts[@]}"
    fi

    # If already running and not init/rebuild → skip
    if [[ "$INIT_CONTAINERS" == false && "$REBUILD_IMAGES" == false ]] && containers_running "$file"; then
        print_info "Containers already running. Skipping startup."
        return 0
    fi

    # Start containers
    up_args=("up")
    [[ "$DETACHED_MODE" == true ]] && up_args+=("--detach")
    [[ "$REBUILD_IMAGES" == true ]] && up_args+=("--build")
    [[ "$NO_CACHE" == true ]] && up_args+=("--no-cache")

    print_info "Starting containers..."
    $cmd "${args[@]}" "${up_args[@]}"
}


# Main
main() {
    check_dependencies
    check_docker_daemon
    validate_compose_files
    compile_repository
    print_info "Done!"
}

if [[ $ZSH_EVAL_CONTEXT == 'toplevel' ]]; then
    main "$@"
fi
