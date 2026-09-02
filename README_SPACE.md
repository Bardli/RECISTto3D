---
title: RECISTto3D
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
python_version: "3.12"
pinned: false
license: apache-2.0
short_description: RECIST to 3D — CPU NIfTI load debug
---

# RECISTto3D (CPU / NIfTI-debug)

Gradio viewer for RECIST-prompted 3D CT segmentation.

## Behavior on this CPU Space

- **Download** the three checkpoints from HF Model repos at startup (~670MB):
  - `wanglab/MedSAM2` → MedSAM2 + EfficientMedSAM2 `.pt`
  - `nnInteractive/nnInteractive` → `nnInteractive_v1.0/.../checkpoint_final.pth`
- **Do not load** them into memory (`SKIP_WEIGHTS` auto-on for CPU Spaces)
- Upload / example NIfTI loading works (latency debugging)
- **RUN** stays disabled until `SKIP_WEIGHTS=0` + ZeroGPU / GPU

Overrides:

- `DOWNLOAD_WEIGHTS=0` — skip download
- `SKIP_WEIGHTS=0` — load weights into memory (needs full model deps)
