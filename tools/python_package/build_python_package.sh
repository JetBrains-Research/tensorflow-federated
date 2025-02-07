#!/usr/bin/env bash
# Copyright 2019, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Tool to build the TensorFlow Federated Python package.
set -e

echo "Current working directory: - $(pwd)"

usage() {
  local script_name=$(basename "${0}")
  echo "usage: ${script_name} --output_dir=<path>"
  echo "  --output_dir=<path>  An optional output directory (defaults to"
  echo "                       '{BUILD_WORKING_DIRECTORY}/dist')."
}

main() {
  local output_dir="${BUILD_WORKING_DIRECTORY}/dist"

  pwd

  while [[ "$#" -gt 0 ]]; do
    option="$1"
    case "${option}" in
      --output_dir=*)
        output_dir="${option#*=}"
        shift
        ;;
      *)
        echo "error: unrecognized option '${option}'" 1>&2
        usage
        exit 1
        ;;
    esac
  done

  if [[ -z "${output_dir}" ]]; then
    echo "error: expected an 'output_dir'" 1>&2
    usage
    exit 1
  fi


  # Create a temp directory.
  local temp_dir="$(mktemp --directory)"
  trap "rm -rf ${temp_dir}" EXIT

  # Create a Python environment.
  python3 -m venv "${temp_dir}/venv"
  source "${temp_dir}/venv/bin/activate"
  python --version
  pip install --upgrade "pip"
  pip --version

  # Build the Python package.
  pip install --upgrade "build"
  pip freeze
  python -m build --outdir "${output_dir}"
}

main "$@"
