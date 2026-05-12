#!/usr/bin/env bash
# Deploy to Kubernetes with helm
set -euo pipefail

NAMESPACE="${NAMESPACE:-panoramic}"
HELM_RELEASE="${HELM_RELEASE:-360-engine}"
VALUES_FILE="${VALUES_FILE:-kubernetes/helm/360-engine/values-prod.yaml}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "🚀 Deploying 360° Engine to Kubernetes"
echo "   Namespace: $NAMESPACE"
echo "   Release:   $HELM_RELEASE"
echo "   Tag:       $IMAGE_TAG"

# Ensure namespace exists
kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || kubectl create namespace "$NAMESPACE"

# Apply secrets (from environment)
kubectl create secret generic panoramic-secrets \
  --namespace="$NAMESPACE" \
  --from-literal=database-url="${DATABASE_URL:-}" \
  --from-literal=redis-url="${REDIS_URL:-}" \
  --from-literal=secret-key="${SECRET_KEY:-change-me}" \
  --from-literal=s3-access-key="${S3_ACCESS_KEY_ID:-}" \
  --from-literal=s3-secret-key="${S3_SECRET_ACCESS_KEY:-}" \
  --dry-run=client -o yaml | kubectl apply -f -

# Helm deploy
helm upgrade --install "$HELM_RELEASE" ./kubernetes/helm/360-engine \
  --namespace="$NAMESPACE" \
  -f "$VALUES_FILE" \
  --set "image.tag=$IMAGE_TAG" \
  --timeout=10m \
  --wait

# Verify
kubectl rollout status deployment/api-server -n "$NAMESPACE" --timeout=5m
echo ""
echo "✅ Deployment successful!"
kubectl get pods -n "$NAMESPACE"
