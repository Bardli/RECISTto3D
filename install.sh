#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MEDSAM2_REPO="${MEDSAM2_REPO:-https://github.com/bowang-lab/MedSAM2.git}"
NNINTERACTIVE_REPO="${NNINTERACTIVE_REPO:-https://github.com/MIC-DKFZ/nnInteractive.git}"
MEDSAM2_COMMIT="${MEDSAM2_COMMIT:-332f30d420f1d1b08e2a79b3ae6a602458808383}"
NNINTERACTIVE_COMMIT="${NNINTERACTIVE_COMMIT:-eb1e2718431acae00953069cfa33199ee1cb8440}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found. Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

clone_or_update() {
  local repo="$1"
  local dir="$2"
  local commit="$3"
  if [ ! -d "$dir/.git" ]; then
    git clone "$repo" "$dir"
  fi
  git -C "$dir" fetch --all --tags
  git -C "$dir" checkout "$commit"
}

echo "==> Cloning model repos"
clone_or_update "$MEDSAM2_REPO" MedSAM2 "$MEDSAM2_COMMIT"
clone_or_update "$NNINTERACTIVE_REPO" nnInteractive "$NNINTERACTIVE_COMMIT"

echo "==> Creating uv environment"
uv python install "$PYTHON_VERSION"
uv venv --python "$PYTHON_VERSION" .venv

echo "==> Installing PyTorch"
uv pip install --python .venv/bin/python \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url "$TORCH_INDEX_URL"

echo "==> Installing local repos and runtime dependencies"
SAM2_BUILD_CUDA=0 uv pip install --python .venv/bin/python --no-build-isolation \
  -e ./MedSAM2 \
  -e ./nnInteractive \
  pandas matplotlib huggingface_hub SimpleITK scipy opencv-python-headless

echo "==> Downloading checkpoints"
.venv/bin/python - <<'PY'
from pathlib import Path
import shutil

from huggingface_hub import hf_hub_download, snapshot_download

root = Path.cwd()
medsam_ckpt = root / "MedSAM2" / "checkpoints"
medsam_ckpt.mkdir(parents=True, exist_ok=True)

for filename in [
    "medsam2_FLARE25_RECIST_baseline.pt",
    "eff_medsam2_small_FLARE25_RECIST_baseline.pt",
]:
    src = Path(
        hf_hub_download(
            repo_id="wanglab/MedSAM2",
            filename=filename,
            cache_dir=str(root / ".hf_cache"),
        )
    )
    dst = medsam_ckpt / filename
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    print(f"ready: {dst}")

snapshot_download(
    repo_id="nnInteractive/nnInteractive",
    allow_patterns=["nnInteractive_v1.0/*"],
    local_dir=str(root / "checkpoints" / "nnInteractive"),
)
print(f"ready: {root / 'checkpoints' / 'nnInteractive' / 'nnInteractive_v1.0'}")
PY

echo "==> Verifying imports"
.venv/bin/python -c "import sam2, efficient_track_anything, nnInteractive; print('imports ok')"

cat <<'EOF'

Install complete.

Try a CPU smoke test:

  .venv/bin/python recist_infer.py \
    --model eff-medsam2 \
    --image examples/eay_demo_lung_cancer_image.nii.gz \
    --recist-line 45,179,235,205,245 \
    --spacing 3.0,0.763671875,0.763671875 \
    --intensity window --window=-175,275 \
    --output examples/eff_medsam2_smoke.npz

EOF
