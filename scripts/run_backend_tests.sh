#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

bash "$SCRIPT_DIR/run_backend_integration_tests.sh"
bash "$SCRIPT_DIR/run_backend_unit_tests.sh"

echo "All backend tests completed."
