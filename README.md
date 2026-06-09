# RECISTto3D

`RECISTto3D` is a small, movable inference pack for turning a RECIST prompt into
a 3D lesion segmentation. It provides one root-level Python entrypoint,
`recist_infer.py`, while keeping each backend's native preprocessing pipeline.

Default RECIST conversions:

- `medsam2`: RECIST line -> EAY-style bounding box.
- `eff-medsam2`: RECIST line -> EAY-style bounding box, Efficient MedSAM2 small only.
- `nninteractive`: RECIST line -> 5 foreground points sampled from the RECIST line.

## What This GitHub Repo Contains

Committed:

- `recist_infer.py`
- `run_eay_cancer_recist_demo.py`
- setup and usage docs
- `install.sh`
- one small test image: `examples/eay_demo_lung_cancer_image.nii.gz`

Not committed:

- `MedSAM2/`
- `nnInteractive/`
- `.venv/`
- model checkpoints
- generated demo outputs

Those large/runtime pieces are created by `install.sh`.

## One-Command Setup

```bash
git clone https://github.com/Bardli/RECISTto3D.git
cd RECISTto3D
bash install.sh
```

The script will:

1. install `uv` if it is missing
2. clone MedSAM2 into `./MedSAM2`
3. clone nnInteractive into `./nnInteractive`
4. create `.venv` with Python 3.12
5. install PyTorch and both local repos with `uv`
6. download MedSAM2, Efficient MedSAM2 small, and nnInteractive checkpoints
7. run a lightweight import check

## Manual Setup

### 1. Clone This Repo

```bash
git clone https://github.com/Bardli/RECISTto3D.git
cd RECISTto3D
```

### 2. Clone The Two Backend Repos

```bash
git clone https://github.com/bowang-lab/MedSAM2.git MedSAM2
git -C MedSAM2 checkout 332f30d420f1d1b08e2a79b3ae6a602458808383

git clone https://github.com/MIC-DKFZ/nnInteractive.git nnInteractive
git -C nnInteractive checkout eb1e2718431acae00953069cfa33199ee1cb8440
```

### 3. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 4. Create The Environment

```bash
uv python install 3.12
uv venv --python 3.12 .venv
```

### 5. Install PyTorch

CUDA 12.4 wheel:

```bash
uv pip install --python .venv/bin/python \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

CPU-only fallback:

```bash
uv pip install --python .venv/bin/python \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

### 6. Install Local Projects And Utilities

```bash
SAM2_BUILD_CUDA=0 uv pip install --python .venv/bin/python --no-build-isolation \
  -e ./MedSAM2 \
  -e ./nnInteractive \
  pandas matplotlib huggingface_hub SimpleITK scipy opencv-python-headless
```

### 7. Download Checkpoints

```bash
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
    src = Path(hf_hub_download("wanglab/MedSAM2", filename=filename, cache_dir=str(root / ".hf_cache")))
    shutil.copy2(src, medsam_ckpt / filename)

snapshot_download(
    repo_id="nnInteractive/nnInteractive",
    allow_patterns=["nnInteractive_v1.0/*"],
    local_dir=str(root / "checkpoints" / "nnInteractive"),
)
PY
```

Expected local model files:

```text
MedSAM2/checkpoints/medsam2_FLARE25_RECIST_baseline.pt
MedSAM2/checkpoints/eff_medsam2_small_FLARE25_RECIST_baseline.pt
checkpoints/nnInteractive/nnInteractive_v1.0/fold_0/checkpoint_final.pth
```

## Smoke Tests

The repository includes one CT image for testing:

```text
examples/eay_demo_lung_cancer_image.nii.gz
```

Efficient MedSAM2 small on CPU:

```bash
.venv/bin/python recist_infer.py \
  --model eff-medsam2 \
  --image examples/eay_demo_lung_cancer_image.nii.gz \
  --recist-line 45,179,235,205,245 \
  --spacing 3.0,0.763671875,0.763671875 \
  --intensity window --window=-175,275 \
  --output examples/eff_medsam2_smoke.npz
```

MedSAM2 on CUDA:

```bash
.venv/bin/python recist_infer.py \
  --model medsam2 \
  --device cuda \
  --image examples/eay_demo_lung_cancer_image.nii.gz \
  --recist-line 45,179,235,205,245 \
  --spacing 3.0,0.763671875,0.763671875 \
  --intensity window --window=-175,275 \
  --output examples/medsam2_smoke.npz
```

nnInteractive on CPU:

```bash
.venv/bin/python recist_infer.py \
  --model nninteractive \
  --device cpu \
  --image examples/eay_demo_lung_cancer_image.nii.gz \
  --recist-line 45,179,235,205,245 \
  --spacing 3.0,0.763671875,0.763671875 \
  --output examples/nninteractive_smoke.npz
```

## Main Usage

```bash
.venv/bin/python recist_infer.py \
  --model eff-medsam2 \
  --image image.nii.gz \
  --recist-line z,x1,y1,x2,y2 \
  --spacing z_spacing,y_spacing,x_spacing \
  --intensity window --window=-175,275 \
  --output pred.npz \
  --output-nifti pred.nii.gz
```

Use `--recist recist.nii.gz` instead of `--recist-line` when you already have a
RECIST mask.

## Notes

- The pack resolves built-in model paths relative to `recist_infer.py`, so it can
  be moved as one directory after setup.
- The virtual environment itself is still path-sensitive. If you move the folder
  after creating `.venv`, rebuilding with `bash install.sh` is the cleanest path.
- Checkpoints are intentionally not stored in git.

More detailed inference options are in `RECIST_INFER_USAGE.md`.
