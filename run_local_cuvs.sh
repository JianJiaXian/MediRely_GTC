#!/usr/bin/env bash
# MediRely — WSL2 GENUINE NVIDIA cuVS launcher (RTX 2060). Run from the bundle root.
set -euo pipefail
cd "$(dirname "$0")"
export MEDIRELY_ARTIFACT_ROOT="$PWD/artifacts_local"
export MEDIRELY_IMAGE_ROOT="$PWD/artifacts_local/bank_images"
export MEDIRELY_MODEL_ROOT="${MEDIRELY_MODEL_ROOT:-$PWD}"     # expects $PWD/medsiglip-448
export MEDIRELY_RETRIEVAL_BACKEND=cuvs                        # STRICT genuine cuVS (no fallback)
export MEDIRELY_MS_DTYPE=float32
echo "MedSigLIP: $MEDIRELY_MODEL_ROOT/medsiglip-448   backend: cuvs (strict)"
python -m gtc_demo.app.app --local-gpu
