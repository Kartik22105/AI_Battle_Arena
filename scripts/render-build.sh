#!/usr/bin/env bash
set -euo pipefail

# Ensure Cargo writes to a writable location in ephemeral build environments
export CARGO_HOME="${CARGO_HOME:-/tmp/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-/tmp/.rustup}"

# Upgrade packaging tools so pip can find binary wheels instead of building from source
python -m pip install --upgrade pip setuptools wheel

# Install Python requirements
python -m pip install -r requirements.txt

# Exit with status of pip install
exit $?