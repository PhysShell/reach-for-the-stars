#!/usr/bin/env bash
# Run the smoke tests with a local MinIO + the chromium pinned by Nix.
# Requires: nix develop (provides minio + ungoogled-chromium).
# MinIO must be 2024-09 or newer for conditional-write headers (If-Match,
# If-None-Match) to be enforced; older versions silently ignore them and
# the s3_cas_semantics test will fail asserting "fail" on a successful put.

set -euo pipefail

require() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing: $1 (run from inside 'nix develop')" >&2
    exit 1
  }
}
require minio
require cargo

REPO_ROOT="$(git rev-parse --show-toplevel)"
HARNESS_DIR="$REPO_ROOT/harness"
cd "$HARNESS_DIR"

MINIO_DATA="$(mktemp -d -t harness-smoke-XXXXXX)"
MINIO_LOG="$(mktemp -t harness-smoke-XXXXXX.log)"

cleanup() {
  if [[ -n "${MINIO_PID:-}" ]] && kill -0 "$MINIO_PID" 2>/dev/null; then
    kill "$MINIO_PID" 2>/dev/null || true
    wait "$MINIO_PID" 2>/dev/null || true
  fi
  rm -rf "$MINIO_DATA" "$MINIO_LOG"
}
trap cleanup EXIT

PORT="${SMOKE_MINIO_PORT:-19000}"
echo "==> starting MinIO on :$PORT (data=$MINIO_DATA, log=$MINIO_LOG)"
MINIO_ROOT_USER=minioadmin MINIO_ROOT_PASSWORD=minioadmin \
  minio server "$MINIO_DATA" --address ":$PORT" --console-address ":$((PORT+1))" \
  >"$MINIO_LOG" 2>&1 &
MINIO_PID=$!

# Wait for MinIO health.
for _ in $(seq 1 40); do
  if curl -fsS "http://127.0.0.1:$PORT/minio/health/ready" >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done
if ! curl -fsS "http://127.0.0.1:$PORT/minio/health/ready" >/dev/null 2>&1; then
  echo "MinIO did not become ready; tail of log:"
  tail -n 50 "$MINIO_LOG" >&2
  exit 1
fi

export SMOKE_S3_ENDPOINT="http://127.0.0.1:$PORT"
export SMOKE_S3_ACCESS_KEY="minioadmin"
export SMOKE_S3_SECRET_KEY="minioadmin"

if [[ -z "${CHROMIUM_BIN:-}" ]]; then
  CHROMIUM_BIN="$(command -v chromium 2>/dev/null || command -v chromium-browser 2>/dev/null || true)"
fi
export CHROMIUM_BIN

if [[ -z "$CHROMIUM_BIN" ]]; then
  echo "==> CHROMIUM_BIN unset; the chromium-dependent test will skip."
fi

echo "==> running cargo test --test smoke"
cargo test --test smoke -- --nocapture --test-threads=1
