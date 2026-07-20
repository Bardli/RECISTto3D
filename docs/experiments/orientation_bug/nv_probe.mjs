// Drive a REAL NiiVue 0.68.2 headless to get ground-truth coordinate numbers
// for pancreas_cancer.nii.gz (direction diag(-1,-1,1)).
import puppeteer from 'puppeteer-core';

const FILE = 'http://127.0.0.1:7872/gradio_api/file=' +
  encodeURIComponent('/mnt/pool/bard_data/RECISTto3D/examples/pancreas_cancer.nii.gz');

const html = `<!doctype html><html><body>
<canvas id="gl" width="960" height="720"></canvas>
<script type="module">
  const { Niivue } = await import("https://unpkg.com/@niivue/niivue@0.68.2/dist/index.js");
  const nv = new Niivue({ isRadiologicalConvention: true });
  await nv.attachTo('gl');
  await nv.loadVolumes([{ url: ${JSON.stringify(FILE)}, name: 'p.nii.gz' }]);
  nv.setRadiologicalConvention(true);
  nv.drawScene();
  const vol = nv.volumes[0];
  window.__probe = () => {
    // pick a fractional point, get its RAS voxel, then round-trip back to canvas
    const frac = [0.3, 0.7, 0.5];
    const rasVox = Array.from(nv.frac2vox(frac)).map(v => Math.round(v));
    const backFrac = Array.from(nv.vox2frac(rasVox));
    const canvasFromFrac = (typeof nv.frac2canvas === 'function') ? Array.from(nv.frac2canvas(frac)) : null;
    const canvasFromRasVox = (typeof nv.frac2canvas === 'function') ? Array.from(nv.frac2canvas(nv.vox2frac(rasVox))) : null;
    // Now feed the FLIPPED value (what the app stores as x1,y1) into vox2frac->canvas
    const dims = vol.dims;
    const maxX = dims[1]-1, maxY = dims[2]-1;
    const flipped = [maxX - rasVox[0], maxY - rasVox[1], rasVox[2]];
    const canvasFromFlipped = (typeof nv.frac2canvas === 'function') ? Array.from(nv.frac2canvas(nv.vox2frac(flipped))) : null;
    return {
      dims: Array.from(vol.dims),
      permRAS: vol.permRAS ? Array.from(vol.permRAS) : null,
      dimsRAS: vol.dimsRAS ? Array.from(vol.dimsRAS) : null,
      frac, rasVox, backFrac,
      canvasFromFrac, canvasFromRasVox,
      flipped, canvasFromFlipped,
    };
  };
  window.__ready = true;
</script></body></html>`;

const browser = await puppeteer.launch({
  executablePath: '/usr/bin/google-chrome-stable',
  headless: 'new',
  args: ['--no-sandbox','--disable-gpu','--use-gl=swiftshader','--enable-unsafe-swiftshader'],
});
const page = await browser.newPage();
const logs = [];
page.on('console', m => logs.push('PAGE: ' + m.text()));
page.on('pageerror', e => logs.push('PAGEERR: ' + e.message));
await page.setContent(html, { waitUntil: 'networkidle0', timeout: 60000 });
try {
  await page.waitForFunction('window.__ready === true', { timeout: 60000 });
  const probe = await page.evaluate('window.__probe()');
  console.log(JSON.stringify(probe, null, 2));
} catch (e) {
  console.log('ERROR:', e.message);
  console.log(logs.join('\n'));
} finally {
  await browser.close();
}
