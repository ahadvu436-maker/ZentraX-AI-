#!/usr/bin/env sh
set -eu

: "${HEALTHCHECK_URL:=http://127.0.0.1:8000/health}"
: "${HEALTHCHECK_TIMEOUT:=5}"

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is required for healthcheck.sh" >&2
  exit 2
fi

curl --fail --silent --show-error \
  --max-time "$HEALTHCHECK_TIMEOUT" \
  "$HEALTHCHECK_URL" >/dev/null

echo "ZentraX health check passed: $HEALTHCHECK_URL"
