#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.vnv}"
REQ_FILE="$ROOT_DIR/requirements.txt"
REQ_HASH_FILE="$VENV_DIR/.requirements.sha256"

PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
RELOAD="${RELOAD:-1}"

print_usage() {
    cat <<'EOF'
Usage:
  ./run_project.sh run      # setup/update env and start API
  ./run_project.sh install  # setup/update env only
  ./run_project.sh check    # verify core imports in env
    ./run_project.sh finetune [args]  # prepare dataset and fine-tune YOLO (CUDA only)
    ./run_project.sh finetune-fast [args]  # fast CUDA training defaults + optional overrides
  ./run_project.sh help

Optional environment variables:
  VENV_DIR=.vnv_path
  PYTHON_BIN=python3
  HOST=0.0.0.0
  PORT=8001
  RELOAD=1
EOF
}

ensure_venv() {
    if [[ ! -d "$VENV_DIR" ]]; then
        echo "[1/4] Creating virtual environment at: $VENV_DIR"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    else
        echo "[1/4] Virtual environment exists: $VENV_DIR"
    fi
}

activate_venv() {
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    echo "[2/4] Active python: $(command -v python)"
}

sync_requirements() {
    if [[ ! -f "$REQ_FILE" ]]; then
        echo "requirements.txt not found at: $REQ_FILE"
        exit 1
    fi

    local current_hash
    current_hash="$(sha256sum "$REQ_FILE" | awk '{print $1}')"

    local previous_hash=""
    if [[ -f "$REQ_HASH_FILE" ]]; then
        previous_hash="$(cat "$REQ_HASH_FILE")"
    fi

    if [[ "$current_hash" != "$previous_hash" ]]; then
        echo "[3/4] Installing/updating dependencies from requirements.txt"
        python -m pip install --upgrade pip
        python -m pip install -r "$REQ_FILE"
        echo "$current_hash" > "$REQ_HASH_FILE"
    else
        echo "[3/4] requirements.txt unchanged, skipping install"
    fi
}

run_api() {
    local reload_flags=()
    case "${RELOAD,,}" in
        1|true|yes|y) reload_flags=(--reload) ;;
        *) reload_flags=() ;;
    esac

    echo "[4/4] Starting API at http://$HOST:$PORT"
    exec uvicorn api.main:app --host "$HOST" --port "$PORT" "${reload_flags[@]}"
}

run_check() {
    python - <<'PY'
import importlib.util
modules = ["fastapi", "numpy", "cv2"]
missing = [name for name in modules if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing modules in env: {', '.join(missing)}")
print("Environment check passed")
PY
}

main() {
    cd "$ROOT_DIR"

    local command="${1:-run}"
    case "$command" in
        run)
            ensure_venv
            activate_venv
            sync_requirements
            run_api
            ;;
        install)
            ensure_venv
            activate_venv
            sync_requirements
            echo "Install complete"
            ;;
        check)
            ensure_venv
            activate_venv
            run_check
            ;;
        finetune)
            ensure_venv
            activate_venv
            sync_requirements
            echo "[4/4] Starting CUDA-only fine-tuning"
            python scripts/fine_tune_yolo.py "${@:2}"
            ;;
        finetune-fast)
            ensure_venv
            activate_venv
            sync_requirements
            echo "[4/4] Starting accelerated CUDA-only fine-tuning"
            local finetune_args=(
                --imgsz auto
                --batch auto
                --workers 0
                --amp
                --cache ram
                --cos-lr
                --fast-mode
                --no-plots
            )
            if (( $# > 1 )); then
                finetune_args+=("${@:2}")
            fi
            python scripts/fine_tune_yolo.py "${finetune_args[@]}"
            ;;
        help|-h|--help)
            print_usage
            ;;
        *)
            echo "Unknown command: $command"
            print_usage
            exit 1
            ;;
    esac
}

main "$@"
