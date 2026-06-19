# RECISTto3D uv Environment

RECISTto3D uses one shared `uv` virtual environment at:

```bash
.venv
```

All commands are relative to the pack root. Replace `/path/to/RECISTto3D` with
wherever the repo is cloned.

## One-Step Setup

```bash
git clone https://github.com/Bardli/RECISTto3D.git
cd RECISTto3D
bash install.sh
```

`install.sh` clones the two backend repositories, creates `.venv` with Python
3.12, installs dependencies with `uv`, downloads checkpoints, and verifies
imports.

## Manual uv Setup

```bash
cd /path/to/RECISTto3D

# Clone backend repos.
git clone https://github.com/bowang-lab/MedSAM2.git MedSAM2
git -C MedSAM2 checkout 332f30d420f1d1b08e2a79b3ae6a602458808383

git clone https://github.com/MIC-DKFZ/nnInteractive.git nnInteractive
git -C nnInteractive checkout eb1e2718431acae00953069cfa33199ee1cb8440

# Create env.
uv python install 3.12
uv venv --python 3.12 .venv
export UV_LINK_MODE=copy

# CUDA 12.4 PyTorch wheels.
uv pip install --python .venv/bin/python \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124

# Build tools for no-build-isolation package builds.
uv pip install --python .venv/bin/python "setuptools>=77" wheel

# Local projects and runtime utilities.
SAM2_BUILD_CUDA=0 uv pip install --python .venv/bin/python --no-build-isolation \
  -e ./MedSAM2 \
  -e ./nnInteractive \
  "gradio[mcp]" pandas matplotlib huggingface_hub SimpleITK scipy opencv-python-headless
```

For CPU-only PyTorch, replace the PyTorch index URL with:

```text
https://download.pytorch.org/whl/cpu
```

## Checkpoints

The install script downloads these files locally:

```text
MedSAM2/checkpoints/medsam2_FLARE25_RECIST_baseline.pt
MedSAM2/checkpoints/eff_medsam2_small_FLARE25_RECIST_baseline.pt
checkpoints/nnInteractive/nnInteractive_v1.0/fold_0/checkpoint_final.pth
```

Checkpoints are intentionally not committed to GitHub.

## Validation

```bash
cd /path/to/RECISTto3D
.venv/bin/python -m pip check
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
.venv/bin/python -c "import sam2, efficient_track_anything, nnInteractive; print('imports ok')"
```

## Smoke Test

The GitHub repo includes one small CT image for testing:

```text
examples/eay_demo_lung_cancer_image.nii.gz
```

```bash
.venv/bin/python recist_infer.py \
  --model eff-medsam2 \
  --image examples/eay_demo_lung_cancer_image.nii.gz \
  --recist-line 45,179,235,205,245 \
  --spacing 3.0,0.763671875,0.763671875 \
  --intensity window --window=-175,275 \
  --output examples/eff_medsam2_smoke.npz
```

## Backend Defaults

- `eff-medsam2`: CPU by default, Efficient MedSAM2 small checkpoint, RECIST -> EAY bbox.
- `medsam2`: CUDA by default, MedSAM2 RECIST checkpoint, RECIST -> EAY bbox.
- `nninteractive`: CUDA by default, local `nnInteractive_v1.0`, RECIST -> 5 foreground points.

The shared entrypoint only shares I/O and RECIST geometry. Each backend keeps its
own preprocessing pipeline.
