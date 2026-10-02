import { useScene } from '@/store/scene';
import { download, safeStem } from '@/lib/download';
import { viewportApi } from '@/store/camera';

const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);
const f2 = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : '—');

async function blobToDataUrl(b: Blob | null) {
  if (!b) return null;
  return new Promise<string>((res) => {
    const r = new FileReader();
    r.onload = () => res(r.result as string);
    r.readAsDataURL(b);
  });
}

/** Self-contained HTML validation report (prints cleanly to PDF). */
export async function exportValidationReport() {
  const { scene, reference, validation } = useScene.getState();
  if (!scene || !reference || !validation) return;
  const shot = await blobToDataUrl(await viewportApi.screenshot());
  const a = validation.all;
  const rows = [
    ['RMSE', `${f2(a.rmse)} m`],
    ['MAE', `${f2(a.mae)} m`],
    ['Bias (prediction − reference)', `${f2(a.bias)} m`],
    ['Pearson r', f2(a.r)],
    ['Within ±1 m', `${(a.within1m * 100).toFixed(1)} %`],
    ['Within ±2 m', `${(a.within2m * 100).toFixed(1)} %`],
    ['Balanced RMSE (mean over strata)', `${f2(validation.balancedRmse)} m`],
    ['Pixels compared', a.n.toLocaleString()],
  ];
  const u = validation.uncertainty;
  if (u) {
    rows.push([`RMSE on confident pixels (σ ≤ ${u.confidentM.toFixed(2)} m)`, `${f2(u.confident.rmse)} m`], ['Confident share of compared pixels', `${(u.coverage * 100).toFixed(1)} %`]);
    if (u.sparsification) rows.push(['σ AUSE / AURG (sparsification, Poggi et al. 2020)', `${f2(u.sparsification.ause)} / ${f2(u.sparsification.aurg)} m`]);
  }
  const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>DepthWizard validation — ${esc(scene.name)}</title>
<style>
body{font:14px/1.5 Inter,Segoe UI,system-ui,sans-serif;color:#151a20;margin:32px auto;max-width:900px;padding:0 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:28px 0 8px;border-bottom:1px solid #dde2e8;padding-bottom:4px}
.dim{color:#5b6572}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #dde2e8;padding:5px 8px;text-align:left}
td.n{text-align:right;font-family:JetBrains Mono,Consolas,monospace}img{max-width:100%;border:1px solid #dde2e8}
@media print{body{margin:0}}
</style></head><body>
<h1>DepthWizard — validation report</h1>
<div class="dim">${esc(scene.name)} · generated ${new Date().toLocaleString()}</div>
<h2>Scene</h2>
<table><tbody>
<tr><td>Product</td><td class="n">${scene.product}</td></tr>
<tr><td>Grid</td><td class="n">${scene.heights.width} × ${scene.heights.height} px @ ${scene.gsd.toFixed(3)} m/px (${scene.gsdSource})</td></tr>
<tr><td>CRS</td><td class="n">${scene.georef?.epsg ? `EPSG:${scene.georef.epsg}` : 'not georeferenced'}</td></tr>
<tr><td>Source</td><td class="n">${esc(scene.provenance.provider)}${scene.provenance.params ? ` · TTA ${scene.provenance.params.tta ? 'on' : 'off'}` : ''}</td></tr>
<tr><td>Reference</td><td class="n">${esc(reference.name)} (${reference.kind}, ${reference.alignment})</td></tr>
${validation.biasRemoved ? `<tr><td>Ground offset removed</td><td class="n">${f2(validation.offset)} m</td></tr>` : ''}
</tbody></table>
<h2>Accuracy</h2>
<table><tbody>${rows.map(([k, v]) => `<tr><td>${k}</td><td class="n">${v}</td></tr>`).join('')}</tbody></table>
<h2>By reference height stratum</h2>
<table><thead><tr><th>Stratum</th><th>N</th><th>RMSE (m)</th><th>MAE (m)</th><th>Bias (m)</th></tr></thead><tbody>
${validation.strata.map((s) => `<tr><td>${s.label}</td><td class="n">${s.n.toLocaleString()}</td><td class="n">${f2(s.rmse)}</td><td class="n">${f2(s.mae)}</td><td class="n">${f2(s.bias)}</td></tr>`).join('')}
</tbody></table>
<h2>Alignment notes</h2><ul>${reference.notes.map((n) => `<li>${esc(n)}</li>`).join('')}</ul>
${shot ? `<h2>View</h2><img src="${shot}" alt="Viewport at export time">` : ''}
<h2>What this does not establish</h2>
<p class="dim">Metrics describe agreement with this one reference over this scene only. A reference with a different acquisition date, datum or resolution adds its own error. Relative (rDSM) results depend on the declared ground resolution.</p>
</body></html>`;
  download(new Blob([html], { type: 'text/html' }), `${safeStem(scene.name)}_validation.html`);
}
