from __future__ import annotations

import shutil
import subprocess
import sys
import time
import warnings
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
WINDOW_PRESET_VALUES = {
    "Soft tissues (W:400 L:50)": (400, 50),
    "Lungs (W:1500 L:-600)": (1500, -600),
}
DEFAULT_WINDOW_PRESET = "Soft tissues (W:400 L:50)"
DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_LEVEL = WINDOW_PRESET_VALUES[DEFAULT_WINDOW_PRESET]
MODEL_CHOICES = [
    ("EfficientMedSAM2", "eff-medsam2"),
    ("MedSAM2", "medsam2"),
    ("nnInteractive", "nninteractive"),
]
MODEL_VALUE_BY_DISPLAY = {display: value for display, value in MODEL_CHOICES}


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

  const tablePanel = document.createElement('div');
  tablePanel.style.cssText = `
    width:100%; box-sizing:border-box; background:#050816; color:#f8fbff;
    border-top:1px solid #8fa4ff; padding:8px 10px;
    font:12px/1.35 'SF Mono',monospace; overflow-x:auto;
  `;
  wrap.appendChild(tablePanel);

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
  let recistLines = [];
  let activeView = 0;
  let drawMode = false;
  let isRadiological = true;
  let isDragging = false;
  let dragStart = null;
  let previewLine = null;
  let selectedLabel = null;
  let visibleMaskLabels = null;
  const RECIST_COLORS = ['#00ff8a', '#ffd166', '#4cc9f0', '#f72585', '#f77f00', '#b8f2e6', '#c77dff', '#90be6d'];

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
      .filter(line => line.z === z && isMaskLabelVisible(line.label))
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

  function isMaskLabelVisible(label) {
    return visibleMaskLabels === null || visibleMaskLabels.has(Number(label));
  }

  function applyMaskLabelVisibility() {
    const cmap = makeLabelColormap();
    for (let i = 0; i < cmap.I.length; i++) {
      const label = cmap.I[i];
      cmap.A[i] = label !== 0 && isMaskLabelVisible(label) ? 255 : 0;
    }
    nv.setDrawColormap(cmap);
    nv.setDrawOpacity(0.7);
    if (nv.drawScene) nv.drawScene();
  }

  function ensureMaskLabelVisible(label) {
    if (visibleMaskLabels !== null) visibleMaskLabels.add(Number(label));
  }

  function forgetMaskLabel(label) {
    if (visibleMaskLabels !== null) visibleMaskLabels.delete(Number(label));
  }

  function resetMaskLabelVisibility() {
    visibleMaskLabels = null;
  }

  function syncMaskLabelVisibilityFromTable() {
    visibleMaskLabels = new Set(
      [...tablePanel.querySelectorAll('.mask-visible-check:checked')]
        .map(item => Number(item.dataset.label))
    );
    applyMaskLabelVisibility();
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
    ensureMaskLabelVisible(line.label);
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
            <th style="padding:5px 7px;">Show</th>
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
              <td style="padding:5px 7px;">
                <input class="mask-visible-check" type="checkbox" data-label="${line.label}" ${isMaskLabelVisible(line.label) ? 'checked' : ''}
                  title="Show mask and RECIST line ${line.label}" style="accent-color:#7b6cf0;cursor:pointer;">
              </td>
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
    tablePanel.querySelectorAll('.mask-visible-check').forEach(input => {
      input.addEventListener('click', ev => ev.stopPropagation());
      input.addEventListener('change', ev => {
        ev.stopPropagation();
        syncMaskLabelVisibilityFromTable();
        renderRecistOverlays();
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
    resetMaskLabelVisibility();
    syncRecistUi();
  }

  function clearMaskDrawing() {
    if (nv.closeDrawing) nv.closeDrawing();
    nv.setDrawingEnabled(false);
    if (nv.drawScene) nv.drawScene();
  }

  function clearViewerAnnotations() {
    clearAllRecist();
    drawMode = false;
    drawBtn.style.cssText = BTN_DEFAULT;
    clearMaskDrawing();
    status.textContent = 'Cleared RECIST lines and mask';
  }

  function deleteRecistLine(label) {
    recistLines = recistLines.filter(line => line.label !== label);
    if (selectedLabel === label) selectedLabel = null;
    forgetMaskLabel(label);
    applyMaskLabelVisibility();
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
    if (!drawMode) return;
    ev.preventDefault();
    ev.stopPropagation();
    setView(0);
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
    if (!drawMode || !isDragging || !dragStart) return;
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
    if (!drawMode || !isDragging || !dragStart) return;
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
    ensureMaskLabelVisible(label);
    previewLine = null;
    syncRecistUi();
    status.textContent = 'RECIST: ' + recistLineText(line);
  }, true);

  async function loadImage(imageUrl, maskUrl = '') {
    if (!imageUrl) {
      status.textContent = 'No image selected';
      return;
    }
    clearAllRecist();
    clearMaskDrawing();
    showLoading('Loading image...');
    try {
      await paintStatus('Loading image...');
      await nv.loadVolumes([{ url: imageUrl }]);
      applyOrientation();
      applyMaskLabelVisibility();
      nv.setDrawingEnabled(false);
      const vol = nv.volumes[0];
      nSlices = (vol.dims && vol.dims[3]) ? vol.dims[3] : 1;
      sliceSlider.max = Math.max(0, nSlices - 1);
      wlLo.value = Math.round(vol.cal_min ?? -898);
      wlHi.value = Math.round(vol.cal_max ?? 401);
      wlLbl.textContent = Math.round(vol.cal_min ?? +wlLo.value) + '..' + Math.round(vol.cal_max ?? +wlHi.value);
      syncWLFill();
      setView(0);
      if (maskUrl) {
        showLoading('Loading mask...');
        await loadMask(maskUrl, false);
      }
      status.textContent = 'Image loaded. Draw RECIST lines.';
    } finally {
      hideLoading();
    }
  }

  async function loadMask(maskUrl, manageLoading = true) {
    if (!maskUrl) return;
    if (manageLoading) showLoading('Loading mask...');
    try {
      await paintStatus('Loading mask...');
      if (nv.closeDrawing) nv.closeDrawing();
      applyMaskLabelVisibility();
      await nv.loadDrawingFromUrl(maskUrl, false);
      nv.setDrawingEnabled(false);
      applyMaskLabelVisibility();
      status.textContent = 'Mask loaded';
    } finally {
      if (manageLoading) hideLoading();
    }
  }

  window.recistTo3DViewer = {
    loadImage,
    loadMask,
    addManualRecistLine,
    clearViewerAnnotations,
    getRecistLine: () => recistLines.map(recistLineText).join('\n') || document.querySelector('#recist-line-box textarea, #recist-line-box input')?.value || '',
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
        (str(EXAMPLE_IMAGE.parent.resolve()), "<EXAMPLES>"),
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
            "Please select a NIfTI file first.",
            "",
        )
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


def _normalize_model(model: str) -> str:
    return MODEL_VALUE_BY_DISPLAY.get(model, model)


def _default_device_for_model(model: str) -> str:
    return DEFAULT_DEVICE


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


def run_inference(
    image_path: str,
    recist_line: str,
    model: str,
    device: str,
    window_preset: str,
    window_width: float | None,
    window_level: float | None,
):
    if not image_path:
        raise gr.Error("Please upload an image or click Load example first.")

    model = _normalize_model(model)
    image = Path(image_path)
    if not image.exists():
        raise gr.Error(f"Image file does not exist: {image}")

    recist_lines = _validate_recist_lines(recist_line)
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
        "--output",
        str(output_npz),
        "--output-nifti",
        str(output_nifti),
    ]
    for line in recist_lines:
        cmd.extend(["--recist-line", line])
    if device:
        cmd.extend(["--device", device])
    cmd.extend(["--intensity", "window", f"--window={_resolve_window(window_preset, window_width, window_level)}"])

    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    log_paths = [image, run_dir, output_npz, output_nifti]
    log = _redact_log_paths("$ " + " ".join(cmd), log_paths) + "\n\n" + _redact_log_paths(proc.stdout, log_paths)
    if proc.stderr:
        log += "\n[stderr]\n" + _redact_log_paths(proc.stderr, log_paths)
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
        f"Inference complete, RECIST lines={len(recist_lines)}",
        log,
    )


with gr.Blocks(title="RECISTto3D Gradio App") as demo:
    gr.Markdown("## RECISTto3D NiiVue Demo")
    gr.Markdown(
        "Upload a `.nii/.nii.gz` file, or click **Load example**. In the **Axial** view, click **Draw RECIST**, "
        "drag one or more lines, then run inference. Spacing is read automatically from the NIfTI header."
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
            with gr.Row():
                model = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value="eff-medsam2",
                    label="Model",
                )
                device = gr.Dropdown(choices=DEVICE_CHOICES, value=DEFAULT_DEVICE, label="Device")
            window_preset = gr.Radio(
                choices=list(WINDOW_PRESET_VALUES.keys()),
                value=DEFAULT_WINDOW_PRESET,
                label="CT Window",
            )
            with gr.Row():
                window_width = gr.Number(label="Custom Window W (integer)", value=DEFAULT_WINDOW_WIDTH)
                window_level = gr.Number(label="Custom Level L (integer)", value=DEFAULT_WINDOW_LEVEL)
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

    model.change(
        _default_device_for_model,
        inputs=model,
        outputs=device,
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
        inputs=[image_path_state, recist_line, model, device, window_preset, window_width, window_level],
        outputs=[mask_path_state, mask_url_state, status, log],
        js="""
        (imagePath, recistLine, model, device, windowPreset, windowWidth, windowLevel) => {
          const drawnLine = window.recistTo3DViewer?.getRecistLine?.() || recistLine || "";
          return [imagePath, drawnLine, model, device, windowPreset, windowWidth, windowLevel];
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
        server_name="0.0.0.0",
        server_port=7871,
        css=APP_CSS,
        allowed_paths=[
            str(EXAMPLE_IMAGE.parent),
            str(APP_DATA),
        ],
        share=False,
        mcp_server=True
    )
