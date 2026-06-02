#!/usr/bin/env bash
# deploy_sagemaker.sh — Build, push, and deploy the CheXpert inference container
# to Amazon SageMaker.
#
# Prerequisites:
#   - AWS CLI v2 authenticated with appropriate IAM permissions
#   - Docker daemon running
#   - Environment variables set (or pass as CLI args):
#       AWS_ACCOUNT_ID, AWS_REGION, MODEL_VERSION, SAGEMAKER_ROLE_ARN
#
# Usage:
#   ./scripts/deploy_sagemaker.sh [--model-version v2.1] [--instance-type ml.g4dn.xlarge]

set -euo pipefail

# ── Defaults (override via env or flags) ─────────────────────────────────────
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID must be set}"
AWS_REGION="${AWS_REGION:-us-east-1}"
MODEL_VERSION="${MODEL_VERSION:-v2.1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-ml.g4dn.xlarge}"
SAGEMAKER_ROLE_ARN="${SAGEMAKER_ROLE_ARN:?SAGEMAKER_ROLE_ARN must be set}"

IMAGE_NAME="chexpert-inference"
ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${IMAGE_NAME}"
IMAGE_TAG="${MODEL_VERSION}-$(date +%Y%m%d%H%M%S)"
FULL_IMAGE="${ECR_REPO}:${IMAGE_TAG}"

ENDPOINT_NAME="chexpert-${MODEL_VERSION//./-}-endpoint"
MODEL_NAME="chexpert-${MODEL_VERSION//./-}-$(date +%Y%m%d%H%M%S)"

log() { printf '\033[1;36m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
err() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# ── Parse flags ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-version)   MODEL_VERSION="$2"; shift 2 ;;
        --instance-type)   INSTANCE_TYPE="$2"; shift 2 ;;
        --region)          AWS_REGION="$2"; shift 2 ;;
        *) err "Unknown flag: $1" ;;
    esac
done

# ── 1. Authenticate Docker with ECR ───────────────────────────────────────────
log "Authenticating Docker with ECR (region: ${AWS_REGION})"
aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login --username AWS --password-stdin "${ECR_REPO}" \
    || err "ECR authentication failed"

# ── 2. Ensure ECR repository exists ───────────────────────────────────────────
log "Ensuring ECR repository '${IMAGE_NAME}' exists"
aws ecr describe-repositories \
    --repository-names "${IMAGE_NAME}" \
    --region "${AWS_REGION}" > /dev/null 2>&1 \
|| aws ecr create-repository \
    --repository-name "${IMAGE_NAME}" \
    --region "${AWS_REGION}" \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256

# ── 3. Build Docker image ──────────────────────────────────────────────────────
log "Building Docker image: ${FULL_IMAGE}"
docker build \
    --platform linux/amd64 \
    --build-arg MODEL_VERSION="${MODEL_VERSION}" \
    --tag "${IMAGE_NAME}:latest" \
    --tag "${FULL_IMAGE}" \
    --file "$(git rev-parse --show-toplevel)/Dockerfile.inference" \
    "$(git rev-parse --show-toplevel)"

# ── 4. Push image to ECR ───────────────────────────────────────────────────────
log "Pushing image to ECR"
docker push "${FULL_IMAGE}"
log "Image pushed: ${FULL_IMAGE}"

# ── 5. Create SageMaker Model ─────────────────────────────────────────────────
log "Creating SageMaker model: ${MODEL_NAME}"
aws sagemaker create-model \
    --model-name "${MODEL_NAME}" \
    --execution-role-arn "${SAGEMAKER_ROLE_ARN}" \
    --primary-container "{
        \"Image\": \"${FULL_IMAGE}\",
        \"Environment\": {
            \"MODEL_VERSION\": \"${MODEL_VERSION}\",
            \"LOG_LEVEL\": \"INFO\",
            \"SAGEMAKER_CONTAINER_LOG_LEVEL\": \"20\"
        }
    }" \
    --region "${AWS_REGION}"

# ── 6. Create or update endpoint configuration ────────────────────────────────
ENDPOINT_CONFIG_NAME="${MODEL_NAME}-config"
log "Creating endpoint configuration: ${ENDPOINT_CONFIG_NAME}"
aws sagemaker create-endpoint-config \
    --endpoint-config-name "${ENDPOINT_CONFIG_NAME}" \
    --production-variants "[
        {
            \"VariantName\": \"primary\",
            \"ModelName\": \"${MODEL_NAME}\",
            \"InitialInstanceCount\": 1,
            \"InstanceType\": \"${INSTANCE_TYPE}\",
            \"InitialVariantWeight\": 1,
            \"ManagedInstanceScaling\": {
                \"Status\": \"ENABLED\",
                \"MinInstanceCount\": 1,
                \"MaxInstanceCount\": 4
            }
        }
    ]" \
    --region "${AWS_REGION}"

# ── 7. Deploy or update endpoint ─────────────────────────────────────────────
EXISTING_ENDPOINT=$(aws sagemaker describe-endpoint \
    --endpoint-name "${ENDPOINT_NAME}" \
    --region "${AWS_REGION}" \
    --query 'EndpointName' \
    --output text 2>/dev/null || echo "")

if [[ -n "${EXISTING_ENDPOINT}" ]]; then
    log "Updating existing endpoint: ${ENDPOINT_NAME}"
    aws sagemaker update-endpoint \
        --endpoint-name "${ENDPOINT_NAME}" \
        --endpoint-config-name "${ENDPOINT_CONFIG_NAME}" \
        --region "${AWS_REGION}"
else
    log "Creating new endpoint: ${ENDPOINT_NAME}"
    aws sagemaker create-endpoint \
        --endpoint-name "${ENDPOINT_NAME}" \
        --endpoint-config-name "${ENDPOINT_CONFIG_NAME}" \
        --region "${AWS_REGION}"
fi

# ── 8. Wait for endpoint to be in service ─────────────────────────────────────
log "Waiting for endpoint to reach InService status (this may take 5-10 min)..."
aws sagemaker wait endpoint-in-service \
    --endpoint-name "${ENDPOINT_NAME}" \
    --region "${AWS_REGION}"

ENDPOINT_STATUS=$(aws sagemaker describe-endpoint \
    --endpoint-name "${ENDPOINT_NAME}" \
    --region "${AWS_REGION}" \
    --query 'EndpointStatus' \
    --output text)

if [[ "${ENDPOINT_STATUS}" == "InService" ]]; then
    log "Deployment successful!"
    log "Endpoint:      ${ENDPOINT_NAME}"
    log "Model:         ${MODEL_NAME}"
    log "Instance type: ${INSTANCE_TYPE}"
    log "Image:         ${FULL_IMAGE}"
else
    err "Endpoint deployment failed with status: ${ENDPOINT_STATUS}"
fi
