#!/usr/bin/env bash
#
# blitzkrieg-deploy.sh — rapid, staged deploy to Google Cloud's global
# edge (Global External HTTPS Load Balancer + Cloud CDN + Cloud Run).
#
# Deliberately staged rather than instant-everywhere: a single bad deploy
# hitting 100% of global traffic at once is the single biggest availability
# risk in this whole pipeline, so this script always goes region-by-region
# with health checks and automatic rollback between hops. That's what makes
# "fast" and "safe" compatible here.
#
# Usage:
#   ./blitzkrieg-deploy.sh <image_tag> <project_id> <service_name>
#
# Requires: gcloud CLI authenticated with deploy permissions, jq.

set -euo pipefail

IMAGE_TAG="${1:?Usage: $0 <image_tag> <project_id> <service_name>}"
PROJECT_ID="${2:?Usage: $0 <image_tag> <project_id> <service_name>}"
SERVICE_NAME="${3:?Usage: $0 <image_tag> <project_id> <service_name>}"

REGIONS=("us-central1" "europe-west1" "asia-southeast1")
HEALTH_CHECK_RETRIES=10
HEALTH_CHECK_INTERVAL_SECONDS=6
ERROR_RATE_THRESHOLD="0.02"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

check_health() {
  local region="$1"
  local url
  url=$(gcloud run services describe "$SERVICE_NAME" \
    --region "$region" --project "$PROJECT_ID" \
    --format='value(status.url)')

  for i in $(seq 1 "$HEALTH_CHECK_RETRIES"); do
    if curl -fsS -o /dev/null -w '%{http_code}' "${url}/healthz" | grep -q '^200$'; then
      log "Health check passed for $region (attempt $i)"
      return 0
    fi
    log "Health check attempt $i failed for $region, retrying..."
    sleep "$HEALTH_CHECK_INTERVAL_SECONDS"
  done
  log "Health check FAILED for $region after $HEALTH_CHECK_RETRIES attempts"
  return 1
}

rollback_region() {
  local region="$1"
  log "Rolling back $region to previous stable revision"
  gcloud run services update-traffic "$SERVICE_NAME" \
    --region "$region" --project "$PROJECT_ID" \
    --to-revisions LATEST=0 --to-latest=false || true
  gcloud run services update-traffic "$SERVICE_NAME" \
    --region "$region" --project "$PROJECT_ID" \
    --to-tags stable=100
}

deploy_region() {
  local region="$1"
  log "Deploying $IMAGE_TAG to $SERVICE_NAME in $region"

  gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_TAG" \
    --region "$region" \
    --project "$PROJECT_ID" \
    --no-traffic \
    --tag "candidate"

  log "Shifting 10% traffic to candidate in $region"
  gcloud run services update-traffic "$SERVICE_NAME" \
    --region "$region" --project "$PROJECT_ID" \
    --to-tags candidate=10

  if ! check_health "$region"; then
    rollback_region "$region"
    log "ABORTING full rollout: $region failed health checks"
    exit 1
  fi

  log "Promoting candidate to 100% traffic in $region"
  gcloud run services update-traffic "$SERVICE_NAME" \
    --region "$region" --project "$PROJECT_ID" \
    --to-tags candidate=100

  log "Deploy to $region complete"
}

update_global_lb_backend() {
  log "Refreshing global external HTTPS load balancer backend services"
  gcloud compute backend-services update "${SERVICE_NAME}-backend" \
    --global --project "$PROJECT_ID" \
    --description "Updated by blitzkrieg-deploy.sh at $(date -u +%FT%TZ)"

  log "Invalidating Cloud CDN cache for updated paths"
  gcloud compute url-maps invalidate-cdn-cache "${SERVICE_NAME}-url-map" \
    --path "/*" --project "$PROJECT_ID" --async
}

main() {
  log "Starting staged global deploy of $SERVICE_NAME ($IMAGE_TAG)"
  for region in "${REGIONS[@]}"; do
    deploy_region "$region"
  done
  update_global_lb_backend
  log "Global deploy complete across: ${REGIONS[*]}"
}

main "$@"
