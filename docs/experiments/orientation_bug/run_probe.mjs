import puppeteer from 'puppeteer-core';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const DIR = '/tmp/claude-1003/-mnt-pool-bard-data/bce90159-c4ee-4100-80a0-ab44d6052776/scratchpad';
const server = http.createServer((req, res) => {
  const p = path.join(DIR, decodeURIComponent(req.url.split('?')[0]));
  fs.readFile(p, (err, buf) => {
    if (err) { res.writeHead(404); res.end('nf'); return; }
    const ext = path.extname(p);
    const ct = ext === '.html' ? 'text/html' : ext === '.gz' ? 'application/gzip' : 'application/octet-stream';
    res.writeHead(200, { 'Content-Type': ct, 'Access-Control-Allow-Origin': '*' });
    res.end(buf);
  });
});
await new Promise(r => server.listen(8899, '127.0.0.1', r));

const browser = await puppeteer.launch({
  executablePath: '/usr/bin/google-chrome-stable',
  headless: 'new',
  args: ['--no-sandbox','--disable-gpu','--use-gl=swiftshader','--enable-unsafe-swiftshader'],
});
const page = await browser.newPage();
const logs = [];
page.on('console', m => logs.push('PAGE: ' + m.text()));
page.on('pageerror', e => logs.push('PAGEERR: ' + e.message));
try {
  await page.goto('http://127.0.0.1:8899/probe.html', { waitUntil: 'networkidle0', timeout: 60000 });
  await page.waitForFunction('window.__ready === true', { timeout: 60000 });
  for (const f of ['kidney.nii.gz','pancreas.nii.gz']) {
    const r = await page.evaluate('window.__run(' + JSON.stringify(f) + ')');
    console.log('\n===== ' + f + ' =====');
    console.log(JSON.stringify(r, null, 2));
  }
} catch (e) {
  console.log('ERROR:', e.message);
  console.log(logs.slice(-15).join('\n'));
} finally {
  await browser.close();
  server.close();
}
