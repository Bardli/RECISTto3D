from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import threading
import time
import warnings
import uuid
from pathlib import Path
from urllib.parse import quote

import gradio as gr

from run_three_models_parallel import MODELS, load_all_models, run_three_models


ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = ROOT / "examples"
EXAMPLE_IMAGES = {
    "Kidney cancer": EXAMPLES_DIR / "kidney_cancer.nii.gz",
    "Liver cancer": EXAMPLES_DIR / "liver_cancer.nii.gz",
    "Lung cancer": EXAMPLES_DIR / "lung_cancer.nii.gz",
    "Pancreas cancer": EXAMPLES_DIR / "pancreas_cancer.nii.gz",
}
APP_DATA = ROOT / "tmp"/".gradio_recistto3d"
UPLOAD_DIR = APP_DATA / "uploads"
RUN_DIR = APP_DATA / "runs"
WINDOW_PRESET_VALUES = {
    "Soft tissues (W:400 L:40)": (400, 40),
    "Lungs (W:1500 L:-600)": (1500, -600),
}
DEFAULT_WINDOW_PRESET = "Soft tissues (W:400 L:40)"
DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_LEVEL = WINDOW_PRESET_VALUES[DEFAULT_WINDOW_PRESET]
MODEL_LABELS = {
    "eff-medsam2": "EfficientMedSAM2",
    "medsam2": "MedSAM2",
    "nninteractive": "nnInteractive",
}
_LOADED_MODELS = None
_LOADED_MODELS_DEVICE: str | None = None
_MODEL_LOAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


def _detect_device_choices() -> list[str]:
    cuda_count = 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch = __import__("torch")
            cuda_count = torch.cuda.device_count()
    except Exception:
        cuda_count = 0
    if cuda_count <= 0:
        try:
            proc = subprocess.run(["nvidia-smi", "-L"], text=True, capture_output=True, check=False)
            if proc.returncode == 0:
                cuda_count = sum(1 for line in proc.stdout.splitlines() if line.strip().startswith("GPU "))
        except OSError:
            cuda_count = 0
    return ["cpu", *[f"cuda:{idx}" for idx in range(cuda_count)]]


DEVICE_CHOICES = _detect_device_choices()
DEFAULT_DEVICE = "cuda:0" if "cuda:0" in DEVICE_CHOICES else "cpu"
APP_CSS = """
#recist-line-box {
  display: none !important;
}
#example-buttons {
  width: 100%;
  max-width: 970px;
}
"""

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
  wrap.appendChild(overlay);

  const loadingOverlay = document.createElement('div');
  loadingOverlay.style.cssText = `
    position:absolute;left:0;top:0;display:none;align-items:center;justify-content:center;
    z-index:3;pointer-events:none;background:rgba(5,8,18,0.55);
    color:#ffffff;font:600 18px/1.4 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    text-shadow:0 1px 4px rgba(0,0,0,0.75);
  `;
  loadingOverlay.textContent = 'Loading...';
  wrap.appendChild(loadingOverlay);

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
      .recist-table { background:#050816 !important; color:#f8fbff !important; }
      .recist-table th {
        background:#243056 !important; color:#ffffff !important;
        border-bottom:1px solid #8fa4ff !important;
      }
      .recist-table td {
        background:#10162b !important; color:#f8fbff !important;
        border-bottom:1px solid #56627f !important;
      }
      .recist-table-row { background:#10162b !important; color:#f8fbff !important; }
      .recist-table-row:hover td { background:#31406c !important; color:#ffffff !important; }
      .recist-table-row.active td { background:#455c9a !important; color:#ffffff !important; }
      .recist-delete {
        padding:2px 7px; border-radius:3px; border:1px solid #ff9cc0 !important;
        background:#5f1438 !important; color:#ffffff !important; cursor:pointer; font:12px/1.4 monospace;
      }
      .recist-delete:hover { border-color:#ffd0df !important; color:#ffffff !important; background:#8f2453 !important; }
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
      <input id="nv-wl-lo" class="nv-r" type="range" min="-1500" max="3000" value="-160">
      <input id="nv-wl-hi" class="nv-r" type="range" min="-1500" max="3000" value="240">
    </div>
    <span id="nv-wl-lbl" style="min-width:95px;color:#ddd;font-size:12px;">W:400 L:40</span>
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

  const MODEL_MASKS = [
    { key: 'eff-medsam2', label: 'EfficientMedSAM2', colormap: 'recist_eff_medsam2', color: '#ff4d4d', rgb: [255, 77, 77], opacity: 0.50},
    { key: 'medsam2', label: 'MedSAM2', colormap: 'recist_medsam2', color: '#31d158', rgb: [49, 209, 88], opacity: 0.50 },
    { key: 'nninteractive', label: 'nnInteractive', colormap: 'recist_nninteractive', color: '#4cc9f0', rgb: [76, 201, 240], opacity: 0.50 },
  ];

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

  const maskControls = document.createElement('div');
  maskControls.style.cssText = 'display:flex;align-items:center;gap:8px;flex-wrap:wrap;color:#ddd;margin-left:4px;';
  maskControls.innerHTML = MODEL_MASKS.map(model => `
    <label title="Show ${model.label} overlay" style="display:flex;align-items:center;gap:3px;cursor:pointer;white-space:nowrap;color:#ddd;">
      <input class="model-mask-check" type="checkbox" data-model="${model.key}" checked
        style="accent-color:${model.color};cursor:pointer;">
      <span style="color:${model.color};">${model.label}</span>
    </label>
  `).join('');
  bar.appendChild(maskControls);
  maskControls.querySelectorAll('.model-mask-check').forEach(input => {
    input.addEventListener('change', () => applyModelMaskVisibility());
  });

  const tablePanel = document.createElement('div');
  tablePanel.style.cssText = `
    width:100%; box-sizing:border-box; background:#050816; color:#f8fbff;
    border-top:1px solid #8fa4ff; padding:8px 10px;
    font:12px/1.35 'SF Mono',monospace; overflow-x:auto;
  `;
  wrap.appendChild(tablePanel);

  let nSlices = 1;
  let recistLines = [];
  let activeView = 0;
  let drawMode = false;
  let isRadiological = true;
  let isDragging = false;
  let dragStart = null;
  let previewLine = null;
  let selectedLabel = null;
  let currentImageUrl = '';
  let maskVolumeIndexByKey = {};
  const RECIST_COLORS = ['#00ff8a', '#ffd166', '#4cc9f0', '#f72585', '#f77f00', '#b8f2e6', '#c77dff', '#90be6d'];

  const nv = new Niivue({
    backColor: [0.05, 0.05, 0.1, 1],
    isResizeCanvas: true,
    loadingText: '',
  });
  await nv.attachTo('niivue-gl');
  MODEL_MASKS.forEach(model => {
    if (typeof nv.addColormap === 'function') {
      nv.addColormap(model.colormap, {
        R: [0, model.rgb[0]],
        G: [0, model.rgb[1]],
        B: [0, model.rgb[2]],
        A: [0, 255],
        I: [0, 255],
      });
    }
  });
  nv.setDrawingEnabled(false);

  const sliceSlider = document.getElementById('nv-slice');
  const sliceLbl = document.getElementById('nv-slice-lbl');
  const wlLo = document.getElementById('nv-wl-lo');
  const wlHi = document.getElementById('nv-wl-hi');
  const wlLbl = document.getElementById('nv-wl-lbl');
  const wlFill = document.getElementById('nv-wl-fill');
  const WL_MIN = -1500, WL_MAX = 3000, WL_SPAN = WL_MAX - WL_MIN;
  const WL_DEFAULT_LOW = -160, WL_DEFAULT_HIGH = 240;

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
    loadingOverlay.style.width = rect.width + 'px';
    loadingOverlay.style.height = rect.height + 'px';
  }

  function addOverlayLine(start, end, color, strokeWidth = 3) {
    if (!start || !end) return;
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', start[0]);
    line.setAttribute('y1', start[1]);
    line.setAttribute('x2', end[0]);
    line.setAttribute('y2', end[1]);
    line.setAttribute('stroke', color);
    line.setAttribute('stroke-width', strokeWidth);
    line.setAttribute('stroke-linecap', 'round');
    overlay.appendChild(line);

    [start, end].forEach(pt => {
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', pt[0]);
      circle.setAttribute('cy', pt[1]);
      circle.setAttribute('r', 4);
      circle.setAttribute('fill', color);
      overlay.appendChild(circle);
    });
  }

  function renderRecistOverlays() {
    updateOverlaySize();
    overlay.replaceChildren();
    if (activeView !== 0) return;
    const z = curSlice();
    recistLines
      .filter(line => line.z === z)
      .forEach(line => {
        const [startScreen, endScreen] = lineScreenPointsFromVox(line);
        if (startScreen && endScreen) {
          line.startScreen = startScreen;
          line.endScreen = endScreen;
          addOverlayLine(startScreen, endScreen, line.color, line.label === selectedLabel ? 5 : 3);
        }
      });
    if (previewLine) addOverlayLine(previewLine.startScreen, previewLine.endScreen, previewLine.color, 4);
  }

  function pct(v) { return (v - WL_MIN) / WL_SPAN * 100; }
  function syncWLFill() {
    const lo = Math.min(+wlLo.value, +wlHi.value);
    const hi = Math.max(+wlLo.value, +wlHi.value);
    wlFill.style.left = pct(lo) + '%';
    wlFill.style.width = (pct(hi) - pct(lo)) + '%';
    wlFill.style.right = 'auto';
  }

  function wlLabel(lo, hi) {
    const width = Math.round(hi - lo);
    const level = Math.round((hi + lo) / 2);
    return `W:${width} L:${level}`;
  }

  function applyWL() {
    const vol = nv.volumes?.[0];
    if (!vol) return;
    const lo = Math.min(+wlLo.value, +wlHi.value);
    const hi = Math.max(+wlLo.value, +wlHi.value);
    vol.cal_min = lo; vol.cal_max = hi;
    if (nv.updateGLVolume) nv.updateGLVolume(); else nv.drawScene();
    wlLbl.textContent = wlLabel(lo, hi);
    syncWLFill();
  }
  wlLo.oninput = wlHi.oninput = applyWL;

  function modelMaskChecked(key) {
    const input = maskControls.querySelector(`.model-mask-check[data-model="${key}"]`);
    return !input || input.checked;
  }

  function setVolumeOpacity(volumeIndex, opacity) {
    if (!Number.isInteger(volumeIndex) || volumeIndex < 0 || !nv.volumes?.[volumeIndex]) return;
    if (typeof nv.setOpacity === 'function') {
      nv.setOpacity(volumeIndex, opacity);
      return;
    }
    nv.volumes[volumeIndex].opacity = opacity;
    if (nv.updateGLVolume) nv.updateGLVolume(); else if (nv.drawScene) nv.drawScene();
  }

  function applyModelMaskVisibility() {
    MODEL_MASKS.forEach(model => {
      const volumeIndex = maskVolumeIndexByKey[model.key];
      setVolumeOpacity(volumeIndex, modelMaskChecked(model.key) ? model.opacity : 0);
    });
    if (nv.drawScene) nv.drawScene();
  }

  function resetModelMaskChecks() {
    maskControls.querySelectorAll('.model-mask-check').forEach(input => {
      input.checked = true;
    });
  }

  function clearModelMasks() {
    const indices = Object.values(maskVolumeIndexByKey)
      .filter(index => Number.isInteger(index) && index > 0)
      .sort((a, b) => b - a);
    indices.forEach(index => {
      const vol = nv.volumes?.[index];
      if (vol && typeof nv.removeVolume === 'function') nv.removeVolume(vol);
    });
    maskVolumeIndexByKey = {};
    resetModelMaskChecks();
    if (nv.updateGLVolume) nv.updateGLVolume(); else if (nv.drawScene) nv.drawScene();
  }

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
    renderRecistOverlays();
  };

  function setView(t) {
    activeView = t;
    if (t !== 0 && drawMode) {
      drawMode = false;
      isDragging = false;
      dragStart = null;
      previewLine = null;
      drawBtn.style.cssText = BTN_DEFAULT;
      status.textContent = 'RECIST draw mode off outside axial view';
    }
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
    renderRecistOverlays();
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
      vox: [isRadiological ? maxX - x : x, maxY - y, z],
    };
  }

  function recistLineText(line) {
    return `${line.z},${line.x1},${line.y1},${line.x2},${line.y2},${line.label}`;
  }

  function syncRecistTextbox() {
    setGradioTextbox('recist-line-box', recistLines.map(recistLineText).join('\n'));
  }

  function nextRecistLabel() {
    return recistLines.reduce((maxLabel, line) => Math.max(maxLabel, line.label), 0) + 1;
  }

  function lineLength(line) {
    return Math.hypot(line.x2 - line.x1, line.y2 - line.y1).toFixed(1);
  }

  function lineScreenPointsFromVox(line) {
    if (!nv.volumes?.[0]) return [line.startScreen || null, line.endScreen || null];
    const dims = nv.volumes[0].dims || [];
    const maxX = (dims[1] || 1) - 1;
    const maxY = (dims[2] || 1) - 1;
    const rect = canvas.getBoundingClientRect();
    const x1 = line.x1;
    const x2 = line.x2;
    const y1 = line.y1;
    const y2 = line.y2;
    if (typeof nv.vox2frac === 'function' && typeof nv.frac2canvas === 'function') {
      const start = nv.frac2canvas(nv.vox2frac([x1, y1, line.z]));
      const end = nv.frac2canvas(nv.vox2frac([x2, y2, line.z]));
      if (start && end) {
        return [
          [start[0] * (rect.width / canvas.width), start[1] * (rect.height / canvas.height)],
          [end[0] * (rect.width / canvas.width), end[1] * (rect.height / canvas.height)],
        ];
      }
    }

    const scale = Math.min(rect.width / (maxX + 1), rect.height / (maxY + 1));
    const offsetX = (rect.width - (maxX + 1) * scale) / 2;
    const offsetY = (rect.height - (maxY + 1) * scale) / 2;
    return [
      [offsetX + (x1 + 0.5) * scale, offsetY + (y1 + 0.5) * scale],
      [offsetX + (x2 + 0.5) * scale, offsetY + (y2 + 0.5) * scale],
    ];
  }

  function parseRecistLineValue(value) {
    const parts = String(value || '').trim().replaceAll(',', ' ').split(/\s+/).filter(Boolean).map(Number);
    if (!parts.length) throw new Error('Enter RECIST as z,x1,y1,x2,y2 or z,x1,y1,x2,y2,label.');
    if (parts.some(v => !Number.isFinite(v))) throw new Error('RECIST line contains a non-numeric value.');
    if (![5, 6].includes(parts.length)) throw new Error('RECIST line must be z,x1,y1,x2,y2 or z,x1,y1,x2,y2,label.');
    const ints = parts.map(v => Math.round(v));
    const label = ints.length === 6 ? ints[5] : nextRecistLabel();
    if (label <= 0) throw new Error('RECIST label must be a positive nonzero integer.');
    if (recistLines.some(line => line.label === label)) throw new Error(`RECIST label ${label} already exists.`);
    return {
      label,
      z: ints[0],
      x1: ints[1],
      y1: ints[2],
      x2: ints[3],
      y2: ints[4],
      color: RECIST_COLORS[(label - 1) % RECIST_COLORS.length],
    };
  }

  function addRecistLine(line, jumpToLine = false) {
    const [startScreen, endScreen] = lineScreenPointsFromVox(line);
    const storedLine = { ...line, startScreen, endScreen };
    recistLines.push(storedLine);
    selectedLabel = line.label;
    if (jumpToLine && nv.volumes?.[0]) {
      jumpToRecistLine(storedLine);
    } else {
      syncRecistUi();
      status.textContent = nv.volumes?.[0]
        ? 'RECIST: ' + recistLineText(line)
        : 'RECIST added. Load an image to display the overlay.';
    }
  }

  function addManualRecistLine(value) {
    try {
      addRecistLine(parseRecistLineValue(value), true);
    } catch (err) {
      status.textContent = err.message || String(err);
    }
  }

  function renderRecistTable() {
    if (!recistLines.length) {
      tablePanel.innerHTML = '<span style="color:#d5dcff;">No RECIST lines yet. Click Draw RECIST and drag on an axial slice.</span>';
      return;
    }
    tablePanel.innerHTML = `
      <table class="recist-table" style="width:100%;border-collapse:collapse;min-width:620px;">
        <thead>
          <tr style="color:#ffffff;text-align:left;">
            <th style="padding:5px 7px;">Label</th>
            <th style="padding:5px 7px;">Z</th>
            <th style="padding:5px 7px;">X1</th>
            <th style="padding:5px 7px;">Y1</th>
            <th style="padding:5px 7px;">X2</th>
            <th style="padding:5px 7px;">Y2</th>
            <th style="padding:5px 7px;">Length</th>
            <th style="padding:5px 7px;">Color</th>
            <th style="padding:5px 7px;">Action</th>
          </tr>
        </thead>
        <tbody>
          ${recistLines.map(line => `
            <tr class="recist-table-row${line.label === selectedLabel ? ' active' : ''}" data-label="${line.label}" style="cursor:pointer;">
              <td style="padding:5px 7px;font-weight:700;">${line.label}</td>
              <td style="padding:5px 7px;">${line.z}</td>
              <td style="padding:5px 7px;">${line.x1}</td>
              <td style="padding:5px 7px;">${line.y1}</td>
              <td style="padding:5px 7px;">${line.x2}</td>
              <td style="padding:5px 7px;">${line.y2}</td>
              <td style="padding:5px 7px;">${lineLength(line)}</td>
              <td style="padding:5px 7px;"><span style="display:inline-block;width:48px;height:12px;border-radius:8px;border:1px solid #fff;background:${line.color};"></span></td>
              <td style="padding:5px 7px;"><button class="recist-delete" data-label="${line.label}">Delete</button></td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
    tablePanel.querySelectorAll('.recist-table-row').forEach(row => {
      row.addEventListener('click', () => {
        const line = recistLines.find(item => item.label === Number(row.dataset.label));
        if (line) jumpToRecistLine(line);
      });
    });
    tablePanel.querySelectorAll('.recist-delete').forEach(btn => {
      btn.addEventListener('click', ev => {
        ev.stopPropagation();
        deleteRecistLine(Number(btn.dataset.label));
      });
    });
  }

  function syncRecistUi() {
    syncRecistTextbox();
    renderRecistTable();
    renderRecistOverlays();
  }

  function clearAllRecist() {
    recistLines = [];
    selectedLabel = null;
    previewLine = null;
    dragStart = null;
    isDragging = false;
    syncRecistUi();
  }

  function clearMaskDrawing() {
    if (nv.closeDrawing) nv.closeDrawing();
    nv.setDrawingEnabled(false);
    if (nv.drawScene) nv.drawScene();
  }

  function clearViewerAnnotations() {
    clearAllRecist();
    clearModelMasks();
    clearMaskDrawing();
    status.textContent = 'Cleared RECIST lines and model overlays';
  }

  function deleteRecistLine(label) {
    recistLines = recistLines.filter(line => line.label !== label);
    if (selectedLabel === label) selectedLabel = null;
    syncRecistUi();
    status.textContent = `RECIST label ${label} deleted`;
  }

  function jumpToRecistLine(line) {
    selectedLabel = line.label;
    setView(0);
    if (nv.scene?.crosshairPos && nSlices > 1) nv.scene.crosshairPos[2] = line.z / (nSlices - 1);
    syncSlice();
    if (nv.drawScene) nv.drawScene();
    renderRecistTable();
    renderRecistOverlays();
    status.textContent = `RECIST label ${line.label}: z=${line.z}`;
  }

  function paintStatus(message) {
    status.textContent = message;
    return new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
  }

  function showLoading(message = 'Loading...') {
    loadingOverlay.textContent = message;
    updateOverlaySize();
    loadingOverlay.style.display = 'flex';
  }

  function hideLoading() {
    loadingOverlay.style.display = 'none';
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
      status.textContent = 'Drag a RECIST line on the axial slice';
    } else {
      status.textContent = 'RECIST draw mode off';
    }
  };

  clearBtn.onclick = clearViewerAnnotations;

  canvas.addEventListener('mousedown', (ev) => {
    if (!drawMode || activeView !== 0) return;
    ev.preventDefault();
    ev.stopPropagation();
    const pt = pointFromEvent(ev);
    if (!pt) return;
    isDragging = true;
    dragStart = pt;
    previewLine = {
      z: curSlice(),
      color: RECIST_COLORS[recistLines.length % RECIST_COLORS.length],
      startScreen: dragStart.screen,
      endScreen: dragStart.screen,
    };
    renderRecistOverlays();
  }, true);

  canvas.addEventListener('mousemove', (ev) => {
    if (!drawMode || activeView !== 0 || !isDragging || !dragStart) return;
    ev.preventDefault();
    ev.stopPropagation();
    const pt = pointFromEvent(ev);
    if (pt) {
      previewLine = {
        z: curSlice(),
        color: RECIST_COLORS[recistLines.length % RECIST_COLORS.length],
        startScreen: dragStart.screen,
        endScreen: pt.screen,
      };
      renderRecistOverlays();
    }
  }, true);

  window.addEventListener('mouseup', (ev) => {
    if (!drawMode || activeView !== 0 || !isDragging || !dragStart) return;
    ev.preventDefault();
    const pt = pointFromEvent(ev);
    isDragging = false;
    if (!pt) {
      previewLine = null;
      renderRecistOverlays();
      return;
    }
    const z = curSlice();
    const label = nextRecistLabel();
    const line = {
      label,
      z,
      x1: dragStart.vox[0],
      y1: dragStart.vox[1],
      x2: pt.vox[0],
      y2: pt.vox[1],
      color: RECIST_COLORS[(label - 1) % RECIST_COLORS.length],
      startScreen: dragStart.screen,
      endScreen: pt.screen,
    };
    recistLines.push(line);
    selectedLabel = label;
    previewLine = null;
    syncRecistUi();
    status.textContent = 'RECIST: ' + recistLineText(line);
  }, true);

  async function loadImage(imageUrl) {
    if (!imageUrl) {
      status.textContent = 'No image selected';
      return;
    }
    currentImageUrl = imageUrl;
    maskVolumeIndexByKey = {};
    clearAllRecist();
    clearMaskDrawing();
    showLoading('Loading image...');
    try {
      await paintStatus('Loading image...');
      await nv.loadVolumes([{ url: imageUrl, name: 'image.nii.gz' }]);
      applyOrientation();
      nv.setDrawingEnabled(false);
      const vol = nv.volumes[0];
      nSlices = (vol.dims && vol.dims[3]) ? vol.dims[3] : 1;
      sliceSlider.max = Math.max(0, nSlices - 1);
      wlLo.value = WL_DEFAULT_LOW;
      wlHi.value = WL_DEFAULT_HIGH;
      applyWL();
      setView(0);
      status.textContent = 'Image loaded. Draw RECIST lines.';
    } finally {
      hideLoading();
    }
  }

  function modelMaskVolume(model, url) {
    return {
      url,
      name: `${model.key}.nii.gz`,
      colormap: model.colormap,
      colorMap: model.colormap,
      opacity: modelMaskChecked(model.key) ? model.opacity : 0,
      cal_min: 0,
      cal_max: 1,
    };
  }

  async function loadModelMasks(effMaskUrl = '', medsam2MaskUrl = '', nninteractiveMaskUrl = '') {
    const urlsByKey = {
      'eff-medsam2': effMaskUrl,
      medsam2: medsam2MaskUrl,
      nninteractive: nninteractiveMaskUrl,
    };
    const available = MODEL_MASKS.filter(model => urlsByKey[model.key]);
    if (!available.length) {
      status.textContent = 'No model masks to load';
      return;
    }
    if (!currentImageUrl) {
      status.textContent = 'Load an image before loading model masks';
      return;
    }
    showLoading('Loading model masks...');
    try {
      await paintStatus('Loading model masks...');
      if (nv.closeDrawing) nv.closeDrawing();
      const previousSlice = curSlice();
      maskVolumeIndexByKey = {};
      const volumes = [{ url: currentImageUrl, name: 'image.nii.gz' }];
      available.forEach(model => {
        maskVolumeIndexByKey[model.key] = volumes.length;
        volumes.push(modelMaskVolume(model, urlsByKey[model.key]));
      });
      await nv.loadVolumes(volumes);
      applyOrientation();
      const vol = nv.volumes[0];
      nSlices = (vol.dims && vol.dims[3]) ? vol.dims[3] : 1;
      sliceSlider.max = Math.max(0, nSlices - 1);
      if (nv.scene?.crosshairPos && nSlices > 1) {
        const restoredSlice = Math.max(0, Math.min(nSlices - 1, previousSlice));
        nv.scene.crosshairPos[2] = restoredSlice / (nSlices - 1);
        sliceSlider.value = restoredSlice;
      }
      nv.setDrawingEnabled(false);
      applyWL();
      applyModelMaskVisibility();
      setView(activeView);
      status.textContent = 'Model masks loaded as overlays';
    } catch (err) {
      status.textContent = 'Failed to load model overlays: ' + (err?.message || String(err));
      throw err;
    } finally {
      hideLoading();
    }
  }

  window.recistTo3DViewer = {
    loadImage,
    loadModelMasks,
    addManualRecistLine,
    clearViewerAnnotations,
    getRecistLine: () => recistLines.map(recistLineText).join('\n') || document.querySelector('#recist-line-box textarea, #recist-line-box input')?.value || '',
    debugState: () => ({
      status: status.textContent,
      nVolumes: nv.volumes?.length || 0,
      volumes: (nv.volumes || []).map((vol, index) => ({
        index,
        name: vol.name,
        colormap: vol.colormap,
        opacity: vol.opacity,
        cal_min: vol.cal_min,
        cal_max: vol.cal_max,
        dims: vol.dims,
      })),
      maskVolumeIndexByKey,
      nSlices,
      currentSlice: curSlice(),
    }),
  };

  setView(0);
  applyOrientation();
  updateOverlaySize();
  renderRecistTable();
  setInterval(() => {
    if (sliceDiv.style.display !== 'none' && nv.volumes?.[0]) {
      syncSlice();
      renderRecistOverlays();
    }
  }, 400);
})();
"""

JS_ON_LOAD = _JS_TEMPLATE


def _ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)


def _file_url(path: str | Path) -> str:
    return f"/gradio_api/file={quote(str(Path(path).resolve()))}"


def _redact_log_paths(text: str, extra_paths: list[Path] | None = None) -> str:
    replacements: list[tuple[str, str]] = [
        (str(RUN_DIR.resolve()), "<RUN_DIR>"),
        (str(UPLOAD_DIR.resolve()), "<UPLOAD_DIR>"),
        (str(APP_DATA.resolve()), "<APP_DATA>"),
        (str(EXAMPLES_DIR.resolve()), "<EXAMPLES>"),
        (str(ROOT.resolve()), "<ROOT>"),
        (str(Path.home().resolve()), "<HOME>"),
    ]
    for path in extra_paths or []:
        try:
            replacements.append((str(path.resolve()), f"<{path.name}>"))
        except OSError:
            replacements.append((str(path), f"<{path.name}>"))
    redacted = text
    for raw, replacement in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        redacted = redacted.replace(raw, replacement)
    return redacted


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
        return (
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "Please select a NIfTI file first.",
            "",
        )
    image_path = _copy_upload(file_path)
    return (
        str(image_path),
        _file_url(image_path),
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        f"Uploaded image loaded: {image_path.name}",
        "",
    )


def load_example_image(example_path: str):
    image = Path(example_path)
    if not image.exists():
        raise gr.Error(f"Example file not found: {image}")
    return (
        str(image),
        _file_url(image),
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        f"Example image loaded: {image.name}",
        "",
    )


def _validate_recist_lines(value: str) -> list[str]:
    lines: list[str] = []
    labels: set[int] = set()
    for line_number, raw_line in enumerate((value or "").splitlines(), start=1):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            parts = [int(round(float(x))) for x in raw_line.replace(",", " ").split()]
        except ValueError as exc:
            raise gr.Error(f"RECIST line {line_number} contains a non-numeric value.") from exc
        if len(parts) == 5:
            parts.append(len(lines) + 1)
        elif len(parts) != 6:
            raise gr.Error(
                "Please draw at least one RECIST line on the axial image first. "
                "Expected one line per row: z,x1,y1,x2,y2,label."
            )
        label = parts[-1]
        if label <= 0:
            raise gr.Error(f"RECIST line {line_number} label must be a positive nonzero integer.")
        if label in labels:
            raise gr.Error(f"RECIST label {label} is duplicated. Each line needs a unique label.")
        labels.add(label)
        lines.append(",".join(str(v) for v in parts))
    if not lines:
        raise gr.Error("Please draw at least one RECIST line on the axial image first.")
    return lines


def _window_bounds_from_wl(width: float, level: float) -> str:
    width_int = int(round(float(width)))
    level_int = int(round(float(level)))
    if width_int <= 0:
        raise gr.Error("Custom CT Window W must be a positive integer.")
    low = level_int - width_int / 2
    high = level_int + width_int / 2
    return f"{low:g},{high:g}"


def _window_values_for_preset(window_preset: str | None):
    if not window_preset:
        return gr.update(), gr.update()
    return WINDOW_PRESET_VALUES[window_preset]


def _preset_for_window_values(window_width: float | None, window_level: float | None):
    if window_width is None or window_level is None:
        return None
    width = int(round(float(window_width)))
    level = int(round(float(window_level)))
    for preset, (preset_width, preset_level) in WINDOW_PRESET_VALUES.items():
        if width == preset_width and level == preset_level:
            return preset
    return None


def _resolve_window(window_preset: str, window_width: float | None, window_level: float | None) -> str:
    has_width = window_width is not None
    has_level = window_level is not None
    if has_width and has_level:
        return _window_bounds_from_wl(window_width, window_level)
    if has_width or has_level:
        raise gr.Error("Please enter both custom CT Window W and Level L, or leave both empty.")
    if not window_preset:
        raise gr.Error("Please select a CT window preset or enter both custom W and L values.")
    width, level = WINDOW_PRESET_VALUES[window_preset]
    return _window_bounds_from_wl(width, level)


def _get_loaded_models(device: str | None):
    global _LOADED_MODELS, _LOADED_MODELS_DEVICE

    requested_device = device or DEFAULT_DEVICE
    with _MODEL_LOAD_LOCK:
        if _LOADED_MODELS is None:
            _LOADED_MODELS = load_all_models(device=requested_device)
            _LOADED_MODELS_DEVICE = requested_device
        elif _LOADED_MODELS_DEVICE != requested_device:
            raise gr.Error(
                "Static models are already loaded on "
                f"{_LOADED_MODELS_DEVICE}. Restart this app to switch to {requested_device}."
            )
        return _LOADED_MODELS


def _preload_models_on_startup() -> None:
    started = time.time()
    print(f"Loading static three-model weights on {DEFAULT_DEVICE}...", flush=True)
    _get_loaded_models(DEFAULT_DEVICE)
    print(f"Static three-model weights loaded in {time.time() - started:.1f}s.", flush=True)


def run_inference(
    image_path: str,
    recist_line: str,
    device: str,
    window_preset: str,
    window_width: float | None,
    window_level: float | None,
):
    if not image_path:
        raise gr.Error("Please upload an image or click Load example first.")

    image = Path(image_path)
    if not image.exists():
        raise gr.Error(f"Image file does not exist: {image}")

    recist_lines = _validate_recist_lines(recist_line)
    _ensure_dirs()
    run_dir = RUN_DIR / f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    window = _resolve_window(window_preset, window_width, window_level)
    stdout = io.StringIO()
    stderr = io.StringIO()
    log_paths = [image, run_dir]
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            models = _get_loaded_models(device or None)
            with _INFERENCE_LOCK:
                results = run_three_models(
                    image=image,
                    loaded_models=models,
                    recist_lines=recist_lines,
                    output_dir=run_dir,
                    output_prefix="prediction",
                    intensity="window",
                    window=window,
                )
    except Exception as exc:
        log = "\n".join(
            part
            for part in [
                f"run_three_models failed: {exc}",
                stdout.getvalue(),
                "[stderr]\n" + stderr.getvalue() if stderr.getvalue() else "",
            ]
            if part
        )
        return (
            "",
            "",
            "",
            "",
            "",
            "",
            "Three-model inference failed. Please check the log.",
            _redact_log_paths(log, log_paths),
        )

    results_by_model = {result.model: result for result in results}
    missing_models = [model for model in MODELS if model not in results_by_model]
    missing_files = [
        result.output_nifti
        for result in results_by_model.values()
        if not Path(result.output_nifti).exists()
    ]
    log_paths.extend(Path(result.output_nifti) for result in results_by_model.values())
    log = "\n".join(
        part
        for part in [
            f"Static models device: {_LOADED_MODELS_DEVICE}",
            "run_three_models(..., loaded_models=<cached>)",
            stdout.getvalue(),
            "[stderr]\n" + stderr.getvalue() if stderr.getvalue() else "",
            json.dumps(
                [
                    {
                        "model": result.model,
                        "output_nifti": result.output_nifti,
                        "duration_s": round(result.duration_s, 3),
                    }
                    for result in results
                ],
                indent=2,
                sort_keys=True,
            ),
        ]
        if part
    )

    if missing_models or missing_files:
        return (
            "",
            "",
            "",
            "",
            "",
            "",
            f"Inference finished with missing outputs. Missing models={missing_models}, missing files={missing_files}",
            _redact_log_paths(log, log_paths),
        )

    ordered_paths = [Path(results_by_model[model].output_nifti) for model in MODELS]
    ordered_urls = [_file_url(path) for path in ordered_paths]
    durations = ", ".join(
        f"{MODEL_LABELS[model]}={results_by_model[model].duration_s:.1f}s"
        for model in MODELS
    )
    return (
        *(str(path) for path in ordered_paths),
        *ordered_urls,
        f"Three-model inference complete, RECIST lines={len(recist_lines)}; {durations}",
        _redact_log_paths(log, log_paths),
    )


with gr.Blocks(title="RECISTto3D Three-Model Gradio App") as demo:
    gr.Markdown("## RECISTto3D Three-Model NiiVue Demo")
    gr.Markdown(
        "Upload a `.nii/.nii.gz` file, or click **Load example**. In the **Axial** view, click **Draw RECIST**, "
        "drag one or more lines, then run all three models. Spacing is read automatically from the NIfTI header."
    )

    image_path_state = gr.Textbox(visible=False)
    image_url_state = gr.Textbox(visible=False)
    eff_mask_path_state = gr.Textbox(visible=False)
    medsam2_mask_path_state = gr.Textbox(visible=False)
    nninteractive_mask_path_state = gr.Textbox(visible=False)
    eff_mask_url_state = gr.Textbox(visible=False)
    medsam2_mask_url_state = gr.Textbox(visible=False)
    nninteractive_mask_url_state = gr.Textbox(visible=False)

    with gr.Row():
        with gr.Column(scale=1):
            upload = gr.File(label="Upload NIfTI (.nii / .nii.gz)", type="filepath")
            recist_line = gr.Textbox(
                label="RECIST lines (one per row: z,x1,y1,x2,y2,label)",
                placeholder="Automatically filled after drawing RECIST lines on the axial NiiVue image",
                lines=4,
                elem_id="recist-line-box",
            )
            with gr.Row():
                manual_recist_line = gr.Textbox(
                    label="Debug add RECIST line",
                    placeholder="z,x1,y1,x2,y2 or z,x1,y1,x2,y2,label",
                    scale=3,
                )
                manual_add = gr.Button("Add line", variant="secondary", scale=1)
            device = gr.Dropdown(
                choices=DEVICE_CHOICES,
                value=DEFAULT_DEVICE,
                label="Static model device (loaded at startup)",
                interactive=False,
            )
            window_preset = gr.Radio(
                choices=list(WINDOW_PRESET_VALUES.keys()),
                value=DEFAULT_WINDOW_PRESET,
                label="CT Window",
            )
            with gr.Row():
                window_width = gr.Number(label="Custom Window W (integer)", value=DEFAULT_WINDOW_WIDTH)
                window_level = gr.Number(label="Custom Level L (integer)", value=DEFAULT_WINDOW_LEVEL)
            run = gr.Button("Run three models", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
            log = gr.Textbox(label="Run log", lines=10, interactive=False)

        with gr.Column(scale=2):
            gr.HTML(value=CANVAS_HTML, js_on_load=JS_ON_LOAD)
            with gr.Group(elem_id="example-buttons"):
                with gr.Row():
                    load_kidney = gr.Button("Kidney cancer", variant="secondary")
                    load_liver = gr.Button("Liver cancer", variant="secondary")
                    load_lung = gr.Button("Lung cancer", variant="secondary")
                    load_pancreas = gr.Button("Pancreas cancer", variant="secondary")

    upload.change(
        prepare_uploaded_image,
        inputs=upload,
        outputs=[
            image_path_state,
            image_url_state,
            eff_mask_path_state,
            medsam2_mask_path_state,
            nninteractive_mask_path_state,
            eff_mask_url_state,
            medsam2_mask_url_state,
            nninteractive_mask_url_state,
            recist_line,
            status,
            log,
        ],
    ).then(
        fn=None,
        inputs=image_url_state,
        outputs=None,
        js="(imageUrl) => { window.recistTo3DViewer?.loadImage(imageUrl); return []; }",
    )

    for button, example_path in [
        (load_kidney, EXAMPLE_IMAGES["Kidney cancer"]),
        (load_liver, EXAMPLE_IMAGES["Liver cancer"]),
        (load_lung, EXAMPLE_IMAGES["Lung cancer"]),
        (load_pancreas, EXAMPLE_IMAGES["Pancreas cancer"]),
    ]:
        button.click(
            load_example_image,
            inputs=gr.State(str(example_path)),
            outputs=[
                image_path_state,
                image_url_state,
                eff_mask_path_state,
                medsam2_mask_path_state,
                nninteractive_mask_path_state,
                eff_mask_url_state,
                medsam2_mask_url_state,
                nninteractive_mask_url_state,
                recist_line,
                status,
                log,
            ],
        ).then(
            fn=None,
            inputs=image_url_state,
            outputs=None,
            js="(imageUrl) => { window.recistTo3DViewer?.loadImage(imageUrl); return []; }",
        )

    manual_add.click(
        fn=None,
        inputs=manual_recist_line,
        outputs=manual_recist_line,
        js="""
        (line) => {
          window.recistTo3DViewer?.addManualRecistLine?.(line || "");
          return [""];
        }
        """,
    )

    window_preset.change(
        _window_values_for_preset,
        inputs=window_preset,
        outputs=[window_width, window_level],
    )

    window_width.change(
        _preset_for_window_values,
        inputs=[window_width, window_level],
        outputs=window_preset,
    )

    window_level.change(
        _preset_for_window_values,
        inputs=[window_width, window_level],
        outputs=window_preset,
    )

    run.click(
        run_inference,
        inputs=[image_path_state, recist_line, device, window_preset, window_width, window_level],
        outputs=[
            eff_mask_path_state,
            medsam2_mask_path_state,
            nninteractive_mask_path_state,
            eff_mask_url_state,
            medsam2_mask_url_state,
            nninteractive_mask_url_state,
            status,
            log,
        ],
        js="""
        (imagePath, recistLine, device, windowPreset, windowWidth, windowLevel) => {
          const drawnLine = window.recistTo3DViewer?.getRecistLine?.() || recistLine || "";
          return [imagePath, drawnLine, device, windowPreset, windowWidth, windowLevel];
        }
        """,
    ).then(
        fn=None,
        inputs=[eff_mask_url_state, medsam2_mask_url_state, nninteractive_mask_url_state],
        outputs=None,
        js="(effMaskUrl, medsam2MaskUrl, nninteractiveMaskUrl) => { window.recistTo3DViewer?.loadModelMasks(effMaskUrl, medsam2MaskUrl, nninteractiveMaskUrl); return []; }",
    )


if __name__ == "__main__":
    _preload_models_on_startup()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7872,
        css=APP_CSS,
        allowed_paths=[
            str(EXAMPLES_DIR),
            str(APP_DATA),
        ],
        share=False,
        mcp_server=True
    )
