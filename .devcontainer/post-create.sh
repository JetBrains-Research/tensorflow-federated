#!/usr/bin/env bash
set -e
exec 2>&1

PROJECT_DIR=$(pwd)

echo "Installing required python packages..."

if nvidia-smi &> /dev/null; then
  GPU_EXTRA="tensorflow[and-cuda]"
fi

# shellcheck disable=SC2086
python3 -m pip install --root-user-action=ignore -r $PROJECT_DIR/requirements.txt ${GPU_EXTRA}
