#!/usr/bin/env bash

# This script reproduces all results reported in the paper 
# using both Torcalia and Dagger as the underlying AMT runtime systems.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

bash ${SCRIPT_DIR}/torcalia/run.sh
bash ${SCRIPT_DIR}/dagger/run.sh

echo
echo "All experiments completed successfully."