#!/usr/bin/env sh
set -eu

: "${APP_MODULE:=main:app}"
: "${APP_HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${WEB_CONCURRENCY:=2}"

case "$PORT" in
  *[!0-9]*|"") echo "ERROR: PORT must be an integer; received '$PORT'" >&2; exit 2 ;;
esac

if [ ! -r /app/backend ]; then
  echo "ERROR: expected application directory /app/backend is missing" >&2
  exit 1
fi

cd /app/backend
echo "Starting ZentraX API: module=${APP_MODULE}, host=${APP_HOST}, port=${PORT}"
exec uvicorn "$APP_MODULE" \
  --host "$APP_HOST" \
  --port "$PORT" \
  --workers "$WEB_CONCURRENCY"
