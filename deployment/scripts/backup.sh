#!/usr/bin/env sh
set -eu

# PostgreSQL backup. Credentials must come from environment/Kubernetes Secret.
: "${PGHOST:?PGHOST is required}"
: "${PGDATABASE:?PGDATABASE is required}"
: "${PGUSER:?PGUSER is required}"
: "${BACKUP_DIR:=/backups}"
: "${RETENTION_DAYS:=14}"

command -v pg_dump >/dev/null 2>&1 || {
  echo "ERROR: pg_dump is not installed" >&2
  exit 2
}

umask 077
mkdir -p "$BACKUP_DIR"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_file="${BACKUP_DIR}/${PGDATABASE}_${timestamp}.dump"

echo "Creating PostgreSQL backup: $backup_file"
pg_dump --format=custom --no-owner --no-acl \
  --file="$backup_file" \
  --host="$PGHOST" \
  --username="$PGUSER" \
  "$PGDATABASE"

test -s "$backup_file" || {
  echo "ERROR: backup file is empty" >&2
  exit 1
}

find "$BACKUP_DIR" -type f -name "${PGDATABASE}_*.dump" \
  -mtime "+${RETENTION_DAYS}" -delete

echo "Backup completed. Verify restore procedures regularly."
