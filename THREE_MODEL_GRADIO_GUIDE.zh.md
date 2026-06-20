# RECISTto3D：三模型推理与 Gradio 交互显示方案

## 1. 用 Python 函数推理 3 个模型，并得到 combined result

核心文件：

```text
/mnt/pool/bard_data/RECISTto3D/run_three_models_parallel.py
```

推荐调用流程：

```python
from pathlib import Path

from run_three_models_parallel import (
    load_all_models,
    infer_with_loaded_models,
    make_bitmask_labels_image,
)

image_path = "image.nii.gz"
recist_path = "recist.nii.gz"
run_dir = Path("outputs/three_models/case_001")
run_dir.mkdir(parents=True, exist_ok=True)

# 1. 只加载一次模型
models = load_all_models(device="cuda:0")
# CPU:
# models = load_all_models(device="cpu")

# 2. 用已加载模型推理 3 个模型
results = infer_with_loaded_models(
    models=models,
    image=image_path,
    recist=recist_path,
    output_dir=run_dir,
    output_prefix="case_001",
    intensity="window",
    window="-175,275",
    run_concurrent=False,
)

# 3. 拿到三个模型的 NIfTI 输出
mask_by_model = {r.model: r.output_nifti for r in results}

eff_mask = mask_by_model["eff-medsam2"]
medsam_mask = mask_by_model["medsam2"]
nninteractive_mask = mask_by_model["nninteractive"]

# 4. 合成一个用于 Gradio/NiiVue 显示的 bitmask label NIfTI
combined_mask = make_bitmask_labels_image(
    eff_medsam2_mask=eff_mask,
    medsam2_mask=medsam_mask,
    nninteractive_mask=nninteractive_mask,
    output_nifti=run_dir / "combined_display_mask.nii.gz",
    reference_nifti=image_path,
)

print(combined_mask)
```

`combined_display_mask.nii.gz` 的 label 含义：

```text
0 = background
1 = eff-medsam2 only
2 = medsam2 only
4 = nninteractive only
3 = eff-medsam2 + medsam2 overlap
5 = eff-medsam2 + nninteractive overlap
6 = medsam2 + nninteractive overlap
7 = all three overlap
```

这样既能显示 3 个模型，也能保留 overlap 信息。

如果输入不是 RECIST mask，而是 RECIST line，也可以这样：

```python
results = infer_with_loaded_models(
    models=models,
    image=image_path,
    recist_lines=[
        "64,108,229,160,281,1",
    ],
    output_dir=run_dir,
    output_prefix="case_001",
    intensity="window",
    window="-175,275",
    run_concurrent=False,
)
```

## 2. 如何修改 `app.py`，用 combined result 做交互显示

当前 `app.py` 的显示逻辑是：

```text
run_inference(...)
  -> 生成 prediction_mask.nii.gz
  -> 返回 mask_url_state
  -> 前端调用 window.recistTo3DViewer.loadMask(maskUrl)
  -> NiiVue 用 nv.loadDrawingFromUrl(maskUrl, false) 显示 mask
```

要显示 3 个模型，建议不要让前端加载 3 个 NIfTI，而是：

```text
后端生成 3 个模型 NIfTI
后端合成 1 个 combined_display_mask.nii.gz
前端继续用现有 loadMask(maskUrl) 加载这个 combined mask
```

### 2.1 在 `app.py` 中 import

在 `app.py` 顶部加入：

```python
from run_three_models_parallel import (
    load_all_models,
    infer_with_loaded_models,
    make_bitmask_labels_image,
)
```

### 2.2 App 启动时加载模型

在 `with gr.Blocks(...)` 之前，或者更合适的位置初始化：

```python
LOADED_MODELS = load_all_models(device=DEFAULT_DEVICE)
```

如果显存不够，可以先用：

```python
LOADED_MODELS = load_all_models(device="cpu")
```

### 2.3 修改 `run_inference(...)`

原来的 `run_inference(...)` 只跑一个模型。可以改成同时跑 3 个模型：

```python
results = infer_with_loaded_models(
    models=LOADED_MODELS,
    image=image,
    recist_lines=recist_lines,
    output_dir=run_dir,
    output_prefix="prediction",
    intensity="window",
    window=_resolve_window(window_preset, window_width, window_level),
    run_concurrent=False,
)

mask_by_model = {r.model: r.output_nifti for r in results}

combined_mask = run_dir / "combined_display_mask.nii.gz"

make_bitmask_labels_image(
    eff_medsam2_mask=mask_by_model["eff-medsam2"],
    medsam2_mask=mask_by_model["medsam2"],
    nninteractive_mask=mask_by_model["nninteractive"],
    output_nifti=combined_mask,
    reference_nifti=image,
)

return (
    str(combined_mask),
    _file_url(combined_mask),
    "Inference complete: 3 model masks combined.",
    log,
)
```

这样 `mask_url_state` 返回的就是 combined NIfTI，现有 JS 的这段可以继续用：

```js
window.recistTo3DViewer?.loadMask(maskUrl)
```

### 2.4 增加用户选择 mask 的 CheckboxGroup

在 Gradio UI 里加：

```python
mask_selector = gr.CheckboxGroup(
    choices=[
        ("EfficientMedSAM2", 1),
        ("MedSAM2", 2),
        ("nnInteractive", 4),
    ],
    value=[1, 2, 4],
    label="Display masks",
)
```

### 2.5 在前端 JS 里增加交互函数

在 `_JS_TEMPLATE` 里面，加入一个函数，用来控制哪些 bit-label 可见：

```js
function setVisibleModelMaskBits(selectedBits) {
  const bits = new Set(selectedBits.map(Number));

  const cmap = makeLabelColormap();

  for (let i = 0; i < cmap.I.length; i++) {
    const label = Number(cmap.I[i]);

    if (label === 0) {
      cmap.A[i] = 0;
      continue;
    }

    const visible =
      (bits.has(1) && (label & 1)) ||
      (bits.has(2) && (label & 2)) ||
      (bits.has(4) && (label & 4));

    cmap.A[i] = visible ? 255 : 0;
  }

  nv.setDrawColormap(cmap);
  nv.setDrawOpacity(0.7);
  if (nv.drawScene) nv.drawScene();
}
```

然后暴露到 `window.recistTo3DViewer`：

```js
window.recistTo3DViewer = {
  loadImage,
  loadMask,
  setVisibleModelMaskBits,
  addManualRecistLine,
  clearViewerAnnotations,
  getRecistLine: () => ...
};
```

### 2.6 连接 Gradio checkbox 到 JS

在 `app.py` 里加：

```python
mask_selector.change(
    fn=None,
    inputs=mask_selector,
    outputs=None,
    js="""
    (selectedBits) => {
      window.recistTo3DViewer?.setVisibleModelMaskBits(selectedBits || []);
      return [];
    }
    """,
)
```

这样用户就可以点击：

```text
只看 EfficientMedSAM2
只看 MedSAM2
只看 nnInteractive
看任意两个
看三个
```

Overlap 也会保留，因为 combined mask 的 label 是 bitmask：

```text
3 = 1 + 2
5 = 1 + 4
6 = 2 + 4
7 = 1 + 2 + 4
```

## 3. `/mnt/pool/bard_data/RECISTto3D/run_three_models_parallel.py` 用法说明

这个文件现在主要提供 3 类功能。

### 3.1 `load_all_models(...)`

作用：提前加载 3 个模型到 CPU/GPU。

```python
models = load_all_models(device="cuda:0")
```

返回：

```python
models.eff_medsam2
models.medsam2
models.nninteractive
models.metadata
```

### 3.2 `infer_with_loaded_models(...)`

作用：使用已经加载好的模型推理，不重复加载 checkpoint。长时间运行的 Gradio 服务建议保持默认的顺序执行，避免重复创建线程带来的运行时缓存增长。

```python
results = infer_with_loaded_models(
    models=models,
    image="image.nii.gz",
    recist="recist.nii.gz",
    output_dir="outputs/three_models",
    output_prefix="case_001",
)
```

输出是 3 个 NIfTI：

```text
case_001_eff_medsam2.nii.gz
case_001_medsam2.nii.gz
case_001_nninteractive.nii.gz
```

每个 `result` 里面有：

```python
result.model
result.output_nifti
result.duration_s
result.metadata
```

### 3.3 `make_bitmask_labels_image(...)`

作用：把 3 个模型输出合成一个适合 Gradio/NiiVue 显示的 combined NIfTI。

```python
combined_mask = make_bitmask_labels_image(
    eff_medsam2_mask="case_001_eff_medsam2.nii.gz",
    medsam2_mask="case_001_medsam2.nii.gz",
    nninteractive_mask="case_001_nninteractive.nii.gz",
    output_nifti="combined_display_mask.nii.gz",
    reference_nifti="image.nii.gz",
)
```

这个函数会检查 mask 的 shape 和 NIfTI geometry，确保三个 mask 可以正确叠加显示。

## 4. 推荐最终流程

```text
Gradio 启动
  -> load_all_models(...)

用户上传 image，画 RECIST
  -> infer_with_loaded_models(...)
  -> 得到 3 个模型 NIfTI
  -> make_bitmask_labels_image(...)
  -> 得到 combined_display_mask.nii.gz
  -> app.py 返回 _file_url(combined_display_mask)
  -> NiiVue loadMask(maskUrl)
  -> Checkbox 控制 label 1/2/4 以及 overlap 显示
```
