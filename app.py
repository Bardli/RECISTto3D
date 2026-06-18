from __future__ import annotations

import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import quote

import gradio as gr


ROOT = Path(__file__).resolve().parent
EXAMPLE_IMAGE = ROOT / "examples" / "eay_demo_lung_cancer_image.nii.gz"
APP_DATA = ROOT / "tmp"/".gradio_recistto3d"
UPLOAD_DIR = APP_DATA / "uploads"
RUN_DIR = APP_DATA / "runs"
PYTHON = ROOT / ".venv" / "bin" / "python"
WINDOW_PRESETS = {
    "Soft tissues (W:400 L:50)": "-150,250",
    "Lungs (W:1500 L:-600)": "-1350,150",
}

CANVAS_HTML = """
<div id="niivue-wrap" style="display:flex;flex-direction:column;line-height:normal;position:relative;width:100%;max-width:960px;">
  <canvas id="niivue-gl" width="960" height="720" style="display:block;width:100%;height:auto;background:#000;"></canvas>
  <div id="nv-status-bar" style="background:#111120;border-top:1px solid #2a2a3a;padding:5px 8px;font:12px/1.35 'SF Mono',monospace;color:#ddd;width:100%;box-sizing:border-box;min-height:24px;">
    <span id="nv-status" style="display:block;max-width:100%;white-space:normal;color:#f5f5f5 !important;background:transparent !important;">Load a NIfTI or example</span>
  </div>
</div>
"""

_JS_TEMPLATE = r"""
(async () => {
  const { Niivue } = await import("https://unpkg.com/@niivue/niivue@0.68.2/dist/index.js");

  const canvas = element.querySelector('#niivue-gl');
  const wrap = element.querySelector('#niivue-wrap');
  const statusBar = element.querySelector('#nv-status-bar');
  const status = element.querySelector('#nv-status');
  status.style.setProperty('color', '#f5f5f5', 'important');
  status.style.setProperty('background', 'transparent', 'important');

  const overlay = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  overlay.style.cssText = 'position:absolute;left:0;top:0;pointer-events:none;z-index:2;';
  overlay.innerHTML = `
    <line id="recist-line-overlay" x1="0" y1="0" x2="0" y2="0"
      stroke="#00ff8a" stroke-width="3" stroke-linecap="round" visibility="hidden"></line>
    <circle id="recist-start-overlay" cx="0" cy="0" r="4" fill="#00ff8a" visibility="hidden"></circle>
    <circle id="recist-end-overlay" cx="0" cy="0" r="4" fill="#00ff8a" visibility="hidden"></circle>
  `;
  wrap.appendChild(overlay);

  if (!document.getElementById('nv-range-style')) {
    const s = document.createElement('style');
    s.id = 'nv-range-style';
    s.textContent = `
      .nv-r { position:absolute; width:100%; height:100%; top:0; left:0; margin:0;
               background:transparent; pointer-events:none;
               -webkit-appearance:none; appearance:none; }
      .nv-r::-webkit-slider-runnable-track { height:3px; background:transparent; }
      .nv-r::-webkit-slider-thumb { pointer-events:all; -webkit-appearance:none;
        width:11px; height:11px; border-radius:50%; background:#7b6cf0;
        cursor:pointer; margin-top:-4px; }
    `;
    document.head.appendChild(s);
  }

  const bar = document.createElement('div');
  bar.style.cssText = `
    display:flex; align-items:center; flex-wrap:wrap; gap:6px;
    background:#1a1a2e; padding:6px 8px;
    font:12px/1.2 'SF Mono',monospace; color:#aaa;
    width:100%; box-sizing:border-box;
  `;
  wrap.insertBefore(bar, statusBar);

  const wlDiv = document.createElement('div');
  wlDiv.style.cssText = 'display:flex; align-items:center; gap:5px; flex:0 1 auto;';
  wlDiv.innerHTML = `
    <span style="color:#666;font-size:12px;">WL</span>
    <div style="position:relative;width:110px;height:16px;">
      <div style="position:absolute;top:50%;left:0;right:0;height:3px;
                  margin-top:-1.5px;background:#333;border-radius:2px;"></div>
      <div id="nv-wl-fill" style="position:absolute;top:50%;height:3px;
                  margin-top:-1.5px;background:#7b6cf0;border-radius:2px;"></div>
      <input id="nv-wl-lo" class="nv-r" type="range" min="-1500" max="3000" value="-898">
      <input id="nv-wl-hi" class="nv-r" type="range" min="-1500" max="3000" value="401">
    </div>
    <span id="nv-wl-lbl" style="min-width:85px;color:#ddd;font-size:12px;">-898..401</span>
  `;
  bar.appendChild(wlDiv);

  const sliceDiv = document.createElement('div');
  sliceDiv.style.cssText = 'display:flex; align-items:center; gap:5px; flex:0 1 auto;';
  sliceDiv.innerHTML = `
    <span id="nv-slice-lbl" style="min-width:50px;color:#ddd;font-size:12px;">0/0</span>
    <input id="nv-slice" type="range" min="0" max="1" value="0"
      style="width:130px;accent-color:#7b6cf0;cursor:pointer;height:16px;">
  `;
  bar.appendChild(sliceDiv);

  const sp = document.createElement('div');
  sp.style.flex = '1 1 24px';
  bar.appendChild(sp);

  const BTN_DEFAULT = 'padding:3px 8px;border-radius:3px;cursor:pointer;font:12px/1.5 monospace;border:1px solid #444;background:transparent;color:#aaa;white-space:nowrap;';
  const BTN_ACTIVE  = 'padding:3px 8px;border-radius:3px;cursor:pointer;font:12px/1.5 monospace;border:1px solid #7b6cf0;background:#5a4db8;color:#fff;white-space:nowrap;';
  const BTN_DRAW_ACTIVE = 'padding:3px 8px;border-radius:3px;cursor:pointer;font:12px/1.5 monospace;border:1px solid #00ff8a;background:#08754a;color:#fff;white-space:nowrap;';

  const drawBtn = document.createElement('button');
  drawBtn.textContent = 'Draw RECIST';
  drawBtn.style.cssText = BTN_DEFAULT;
  bar.appendChild(drawBtn);

  const clearBtn = document.createElement('button');
  clearBtn.textContent = 'Clear';
  clearBtn.style.cssText = BTN_DEFAULT;
  bar.appendChild(clearBtn);

  const VIEWS = [[0,'Axial'],[1,'Coronal'],[2,'Sagittal'],[3,'Multi']];
  const btnMap = {};
  VIEWS.forEach(([t, name]) => {
    const btn = document.createElement('button');
    btn.textContent = name;
    btn.style.cssText = BTN_DEFAULT;
    btn.onclick = () => setView(t);
    btnMap[t] = btn;
    bar.appendChild(btn);
  });

  function hsvToRgb(h, s, v) {
    const i = Math.floor(h * 6);
    const f = h * 6 - i;
    const p = v * (1 - s);
    const q = v * (1 - f * s);
    const t = v * (1 - (1 - f) * s);
    const triplets = [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]];
    return triplets[i % 6].map(c => Math.round(c * 255));
  }

  function makeLabelColormap(maxLabel = 255) {
    const cmap = { R: [], G: [], B: [], A: [], I: [], labels: [] };
    for (let label = 0; label <= maxLabel; label++) {
      cmap.I.push(label);
      cmap.labels.push(label === 0 ? 'Background' : `Label ${label}`);
      if (label === 0) {
        cmap.R.push(0); cmap.G.push(0); cmap.B.push(0); cmap.A.push(0);
        continue;
      }
      const [r, g, b] = hsvToRgb((label * 0.61803398875) % 1, 0.72, 1.0);
      cmap.R.push(r); cmap.G.push(g); cmap.B.push(b); cmap.A.push(255);
    }
    return cmap;
  }

  let nSlices = 1;
  let recistLine = '';
  let drawMode = false;
  let isRadiological = true;
  let isDragging = false;
  let dragStart = null;

  const segLabelCmap = makeLabelColormap();
  const nv = new Niivue({
    backColor: [0.05, 0.05, 0.1, 1],
    isResizeCanvas: false,
    loadingText: '',
  });
  await nv.attachTo('niivue-gl');
  nv.setDrawColormap(segLabelCmap);
  nv.setDrawOpacity(0.7);
  nv.setDrawingEnabled(false);

  const sliceSlider = document.getElementById('nv-slice');
  const sliceLbl = document.getElementById('nv-slice-lbl');
  const wlLo = document.getElementById('nv-wl-lo');
  const wlHi = document.getElementById('nv-wl-hi');
  const wlLbl = document.getElementById('nv-wl-lbl');
  const wlFill = document.getElementById('nv-wl-fill');
  const WL_MIN = -1500, WL_MAX = 3000, WL_SPAN = WL_MAX - WL_MIN;

  function setGradioTextbox(elemId, value) {
    const host = document.getElementById(elemId);
    const input = host?.querySelector('textarea, input');
    if (!input) return;
    input.value = value;
    input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function updateOverlaySize() {
    const rect = canvas.getBoundingClientRect();
    overlay.setAttribute('width', rect.width);
    overlay.setAttribute('height', rect.height);
    overlay.setAttribute('viewBox', `0 0 ${rect.width} ${rect.height}`);
    overlay.style.width = rect.width + 'px';
    overlay.style.height = rect.height + 'px';
  }

  function setOverlayLine(start, end, visible = true) {
    updateOverlaySize();
    const ids = ['recist-line-overlay', 'recist-start-overlay', 'recist-end-overlay'];
    ids.forEach(id => overlay.querySelector('#' + id).setAttribute('visibility', visible ? 'visible' : 'hidden'));
    if (!visible || !start || !end) return;
    overlay.querySelector('#recist-line-overlay').setAttribute('x1', start.screen[0]);
    overlay.querySelector('#recist-line-overlay').setAttribute('y1', start.screen[1]);
    overlay.querySelector('#recist-line-overlay').setAttribute('x2', end.screen[0]);
    overlay.querySelector('#recist-line-overlay').setAttribute('y2', end.screen[1]);
    overlay.querySelector('#recist-start-overlay').setAttribute('cx', start.screen[0]);
    overlay.querySelector('#recist-start-overlay').setAttribute('cy', start.screen[1]);
    overlay.querySelector('#recist-end-overlay').setAttribute('cx', end.screen[0]);
    overlay.querySelector('#recist-end-overlay').setAttribute('cy', end.screen[1]);
  }

  function pct(v) { return (v - WL_MIN) / WL_SPAN * 100; }
  function syncWLFill() {
    const lo = Math.min(+wlLo.value, +wlHi.value);
    const hi = Math.max(+wlLo.value, +wlHi.value);
    wlFill.style.left = pct(lo) + '%';
    wlFill.style.width = (pct(hi) - pct(lo)) + '%';
    wlFill.style.right = 'auto';
  }

  function applyWL() {
    const vol = nv.volumes?.[0];
    if (!vol) return;
    const lo = Math.min(+wlLo.value, +wlHi.value);
    const hi = Math.max(+wlLo.value, +wlHi.value);
    vol.cal_min = lo; vol.cal_max = hi;
    if (nv.updateGLVolume) nv.updateGLVolume(); else nv.drawScene();
    wlLbl.textContent = lo + '..' + hi;
    syncWLFill();
  }
  wlLo.oninput = wlHi.oninput = applyWL;

  function curSlice() {
    return Math.round((nv.scene?.crosshairPos?.[2] ?? 0.5) * (nSlices - 1));
  }

  function syncSlice() {
    const c = curSlice();
    sliceLbl.textContent = c + '/' + (nSlices - 1);
    sliceSlider.value = c;
  }

  sliceSlider.oninput = () => {
    const idx = +sliceSlider.value;
    if (nv.scene?.crosshairPos && nSlices > 1) nv.scene.crosshairPos[2] = idx / (nSlices - 1);
    nv.drawScene();
    sliceLbl.textContent = idx + '/' + (nSlices - 1);
  };

  function setView(t) {
    if (t === 3) {
      canvas.width = 960; canvas.height = 960;
    } else {
      canvas.width = 960; canvas.height = 720;
    }
    nv.opts.multiplanarShowRender = (t === 3) ? 1 : 0;
    nv.resizeListener();
    nv.setSliceType(t);
    updateOverlaySize();
    Object.values(btnMap).forEach(b => b.style.cssText = BTN_DEFAULT);
    btnMap[t].style.cssText = BTN_ACTIVE;
    sliceDiv.style.display = (t === 0) ? 'flex' : 'none';
    if (t === 0) syncSlice();
  }

  function pointFromEvent(ev) {
    if (!nv.volumes?.[0] || typeof nv.canvasPos2frac !== 'function' || typeof nv.frac2vox !== 'function') {
      status.textContent = 'NiiVue coordinate API unavailable';
      return null;
    }
    const rect = canvas.getBoundingClientRect();
    const cssX = ev.clientX - rect.left;
    const cssY = ev.clientY - rect.top;
    const canvasX = cssX * (canvas.width / rect.width);
    const canvasY = cssY * (canvas.height / rect.height);
    const frac = nv.canvasPos2frac([canvasX, canvasY]);
    if (!frac || frac.some(v => !Number.isFinite(v) || v < -0.001 || v > 1.001)) return null;
    const vox = nv.frac2vox(frac).map(v => Math.round(v));
    const dims = nv.volumes[0].dims || [];
    const maxX = (dims[1] || 1) - 1;
    const maxY = (dims[2] || 1) - 1;
    const maxZ = (dims[3] || nSlices || 1) - 1;
    const x = Math.max(0, Math.min(maxX, vox[0]));
    const y = Math.max(0, Math.min(maxY, vox[1]));
    const z = Math.max(0, Math.min(maxZ, vox[2]));
    return {
      screen: [cssX, cssY],
      vox: [isRadiological ? maxX - x : x, y, z],
    };
  }

  function clearRecist() {
    recistLine = '';
    dragStart = null;
    setOverlayLine(null, null, false);
    setGradioTextbox('recist-line-box', '');
    status.textContent = 'RECIST line cleared';
  }

  function paintStatus(message) {
    status.textContent = message;
    return new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  }

  function applyOrientation() {
    nv.setRadiologicalConvention(isRadiological);
    if (nv.drawScene) nv.drawScene();
  }

  drawBtn.onclick = () => {
    drawMode = !drawMode;
    drawBtn.style.cssText = drawMode ? BTN_DRAW_ACTIVE : BTN_DEFAULT;
    if (drawMode) {
      setView(0);
      status.textContent = 'Drag one line on the axial slice';
    } else {
      status.textContent = 'RECIST draw mode off';
    }
  };
  clearBtn.onclick = clearRecist;

  canvas.addEventListener('mousedown', (ev) => {
    if (!drawMode) return;
    ev.preventDefault();
    ev.stopPropagation();
    setView(0);
    const pt = pointFromEvent(ev);
    if (!pt) return;
    isDragging = true;
    dragStart = pt;
    setOverlayLine(dragStart, dragStart, true);
  }, true);

  canvas.addEventListener('mousemove', (ev) => {
    if (!drawMode || !isDragging || !dragStart) return;
    ev.preventDefault();
    ev.stopPropagation();
    const pt = pointFromEvent(ev);
    if (pt) setOverlayLine(dragStart, pt, true);
  }, true);

  window.addEventListener('mouseup', (ev) => {
    if (!drawMode || !isDragging || !dragStart) return;
    ev.preventDefault();
    const pt = pointFromEvent(ev);
    isDragging = false;
    if (!pt) return;
    setOverlayLine(dragStart, pt, true);
    const z = Math.round((dragStart.vox[2] + pt.vox[2]) / 2);
    recistLine = `${z},${dragStart.vox[0]},${dragStart.vox[1]},${pt.vox[0]},${pt.vox[1]}`;
    setGradioTextbox('recist-line-box', recistLine);
    status.textContent = 'RECIST: ' + recistLine;
  }, true);

  async function loadImage(imageUrl, maskUrl = '') {
    if (!imageUrl) {
      status.textContent = 'No image selected';
      return;
    }
    clearRecist();
    await paintStatus('Loading image...');
    await nv.loadVolumes([{ url: imageUrl }]);
    applyOrientation();
    nv.setDrawColormap(segLabelCmap);
    nv.setDrawOpacity(0.7);
    nv.setDrawingEnabled(false);
    const vol = nv.volumes[0];
    nSlices = (vol.dims && vol.dims[3]) ? vol.dims[3] : 1;
    sliceSlider.max = Math.max(0, nSlices - 1);
    wlLo.value = Math.round(vol.cal_min ?? -898);
    wlHi.value = Math.round(vol.cal_max ?? 401);
    wlLbl.textContent = Math.round(vol.cal_min ?? +wlLo.value) + '..' + Math.round(vol.cal_max ?? +wlHi.value);
    syncWLFill();
    setView(0);
    if (maskUrl) await loadMask(maskUrl);
    status.textContent = 'Image loaded. Draw RECIST line.';
  }

  async function loadMask(maskUrl) {
    if (!maskUrl) return;
    await paintStatus('Loading mask...');
    if (nv.closeDrawing) nv.closeDrawing();
    nv.setDrawColormap(segLabelCmap);
    nv.setDrawOpacity(0.7);
    await nv.loadDrawingFromUrl(maskUrl, false);
    nv.setDrawingEnabled(false);
    nv.drawScene();
    status.textContent = 'Mask loaded';
  }

  window.recistTo3DViewer = {
    loadImage,
    loadMask,
    getRecistLine: () => recistLine || document.querySelector('#recist-line-box textarea, #recist-line-box input')?.value || '',
  };

  setView(0);
  applyOrientation();
  updateOverlaySize();
  setInterval(() => { if (sliceDiv.style.display !== 'none' && nv.volumes?.[0]) syncSlice(); }, 400);
})();
"""

JS_ON_LOAD = _JS_TEMPLATE


def _ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)


def _file_url(path: str | Path) -> str:
    return f"/gradio_api/file={quote(str(Path(path).resolve()))}"


def _nifti_suffix(path: str | Path) -> str:
    name = Path(path).name.lower()
    if name.endswith(".nii.gz"):
        return ".nii.gz"
    if name.endswith(".nii"):
        return ".nii"
    raise gr.Error("Please upload a .nii or .nii.gz file.")


def _copy_upload(file_path: str) -> Path:
    _ensure_dirs()
    src = Path(file_path)
    suffix = _nifti_suffix(src)
    dst = UPLOAD_DIR / f"{int(time.time())}_{uuid.uuid4().hex[:8]}{suffix}"
    shutil.copy2(src, dst)
    return dst


def prepare_uploaded_image(file_path: str | None):
    if not file_path:
        raise gr.Error("Please select a NIfTI file first.")
    image_path = _copy_upload(file_path)
    return (
        str(image_path),
        _file_url(image_path),
        "",
        "",
        f"Uploaded image loaded: {image_path.name}",
        "",
    )


def load_example_image():
    if not EXAMPLE_IMAGE.exists():
        raise gr.Error(f"Example file not found: {EXAMPLE_IMAGE}")
    return (
        str(EXAMPLE_IMAGE),
        _file_url(EXAMPLE_IMAGE),
        "",
        "",
        f"Example image loaded: {EXAMPLE_IMAGE.name}",
        "",
    )


def _validate_recist_line(value: str) -> str:
    parts = [int(round(float(x))) for x in value.replace(",", " ").split()]
    if len(parts) != 5:
        raise gr.Error("Please draw a RECIST line on the axial image first. Expected format: z,x1,y1,x2,y2.")
    return ",".join(str(v) for v in parts)


def run_inference(
    image_path: str,
    recist_line: str,
    model: str,
    device: str,
    window_preset: str,
):
    if not image_path:
        raise gr.Error("Please upload an image or click Load example first.")

    image = Path(image_path)
    if not image.exists():
        raise gr.Error(f"Image file does not exist: {image}")

    recist_line = _validate_recist_line(recist_line)
    _ensure_dirs()
    run_dir = RUN_DIR / f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output_npz = run_dir / "prediction.npz"
    output_nifti = run_dir / "prediction_mask.nii.gz"

    python_exe = PYTHON if PYTHON.exists() else Path(sys.executable)
    cmd = [
        str(python_exe),
        str(ROOT / "recist_infer.py"),
        "--model",
        model,
        "--image",
        str(image),
        "--recist-line",
        recist_line,
        "--output",
        str(output_npz),
        "--output-nifti",
        str(output_nifti),
    ]
    if device:
        cmd.extend(["--device", device])
    cmd.extend(["--intensity", "window", f"--window={WINDOW_PRESETS[window_preset]}"])

    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    log = "$ " + " ".join(cmd) + "\n\n" + proc.stdout
    if proc.stderr:
        log += "\n[stderr]\n" + proc.stderr
    if proc.returncode != 0:
        return (
            "",
            "",
            f"Inference failed with exit code {proc.returncode}. Please check the log.",
            log,
        )
    if not output_nifti.exists():
        return (
            "",
            "",
            "Inference finished but did not generate output-nifti.",
            log,
        )

    return (
        str(output_nifti),
        _file_url(output_nifti),
        f"Inference complete, RECIST line={recist_line}",
        log,
    )


with gr.Blocks(title="RECISTto3D Gradio App") as demo:
    gr.Markdown("## RECISTto3D NiiVue Demo")
    gr.Markdown(
        "Upload a `.nii/.nii.gz` file, or click **Load example**. In the **Axial** view, click **Draw RECIST**, "
        "drag to draw a line, then run inference. Spacing is read automatically from the NIfTI header."
    )

    image_path_state = gr.Textbox(visible=False)
    image_url_state = gr.Textbox(visible=False)
    mask_path_state = gr.Textbox(visible=False)
    mask_url_state = gr.Textbox(visible=False)

    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.File(label="Upload NIfTI (.nii / .nii.gz)", type="filepath")
            load_example = gr.Button("Load example", variant="secondary")
            recist_line = gr.Textbox(
                label="RECIST line (z,x1,y1,x2,y2)",
                placeholder="Automatically filled after drawing a line on the axial NiiVue image",
                elem_id="recist-line-box",
            )
            with gr.Row():
                model = gr.Dropdown(
                    choices=["eff-medsam2", "medsam2", "nninteractive"],
                    value="eff-medsam2",
                    label="Model",
                )
                device = gr.Dropdown(choices=["", "cpu", "cuda", "cuda:0"], value="", label="Device")
            window_preset = gr.Radio(
                choices=list(WINDOW_PRESETS.keys()),
                value="Soft tissues (W:400 L:50)",
                label="CT Window",
            )
            run = gr.Button("Run inference", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
            log = gr.Textbox(label="Run log", lines=10, interactive=False)

        with gr.Column(scale=2):
            gr.HTML(value=CANVAS_HTML, js_on_load=JS_ON_LOAD)

    upload.change(
        prepare_uploaded_image,
        inputs=upload,
        outputs=[image_path_state, image_url_state, mask_url_state, recist_line, status, log],
    ).then(
        fn=None,
        inputs=[image_url_state, mask_url_state],
        outputs=None,
        js="(imageUrl, maskUrl) => { window.recistTo3DViewer?.loadImage(imageUrl, maskUrl || ''); return []; }",
    )

    load_example.click(
        load_example_image,
        inputs=None,
        outputs=[image_path_state, image_url_state, mask_url_state, recist_line, status, log],
    ).then(
        fn=None,
        inputs=[image_url_state, mask_url_state],
        outputs=None,
        js="(imageUrl, maskUrl) => { window.recistTo3DViewer?.loadImage(imageUrl, maskUrl || ''); return []; }",
    )

    run.click(
        run_inference,
        inputs=[image_path_state, recist_line, model, device, window_preset],
        outputs=[mask_path_state, mask_url_state, status, log],
        js="""
        (imagePath, recistLine, model, device, windowPreset) => {
          const drawnLine = window.recistTo3DViewer?.getRecistLine?.() || recistLine || "";
          return [imagePath, drawnLine, model, device, windowPreset];
        }
        """,
    ).then(
        fn=None,
        inputs=mask_url_state,
        outputs=None,
        js="(maskUrl) => { window.recistTo3DViewer?.loadMask(maskUrl); return []; }",
    )


if __name__ == "__main__":
    demo.launch(
        server_port=7871,
        allowed_paths=[
            str(EXAMPLE_IMAGE.parent),
            str(APP_DATA),
        ],
    )
