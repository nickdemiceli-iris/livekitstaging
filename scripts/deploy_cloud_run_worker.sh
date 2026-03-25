#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <gcp_project_id> <region> <artifact_repo> [service_name] [image_tag]"
  exit 1
fi

PROJECT_ID="$1"
REGION="$2"
ARTIFACT_REPO="$3"
SERVICE_NAME="${4:-outbound-agent-worker}"
IMAGE_TAG="${5:-$(date +%Y%m%d-%H%M%S)}"

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}/${SERVICE_NAME}:${IMAGE_TAG}"

echo "Building image: ${IMAGE_URI}"
gcloud builds submit \
  --project "${PROJECT_ID}" \
  --tag "${IMAGE_URI}"

echo "Deploying Cloud Run service: ${SERVICE_NAME}"
gcloud run deploy "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --image "${IMAGE_URI}" \
  --platform managed \
  --execution-environment gen2 \
  --concurrency 1 \
  --min-instances 1 \
  --max-instances 20 \
  --cpu 2 \
  --memory 2Gi \
  --timeout 900 \
  --no-allow-unauthenticated \
  --set-env-vars "PYTHONUNBUFFERED=1,WORKER_AGENT_NAME=verifacto-outbound-agent"

echo "Deployment complete."
echo "Image: ${IMAGE_URI}"
