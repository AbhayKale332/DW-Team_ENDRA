import { _electron as electron, expect } from '@playwright/test';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import assert from 'node:assert/strict';

const userData = await mkdtemp(join(tmpdir(), 'dw-electron-smoke-'));
let desktop;
try {
  const args = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--disable-gpu-sandbox'];
  if (!process.env.DW_DESKTOP_EXECUTABLE) args.push(resolve('.'));
  desktop = await electron.launch({
    ...(process.env.DW_DESKTOP_EXECUTABLE ? { executablePath: process.env.DW_DESKTOP_EXECUTABLE } : {}),
    args, timeout: 60_000, env: { ...process.env, DW_USER_DATA: userData },
  });
  const page = await desktop.firstWindow();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.waitForLoadState('networkidle');
  assert.match(page.url(), /127\.0\.0\.1:\d+\/\?desktop/);
  await page.getByText('DepthWizard', { exact: true }).first().waitFor();
  const state = await desktop.evaluate(({ BrowserWindow }) => {
    const prefs = BrowserWindow.getAllWindows()[0].webContents.getLastWebPreferences();
    return { sandbox: prefs.sandbox, nodeIntegration: prefs.nodeIntegration, contextIsolation: prefs.contextIsolation };
  });
  assert.deepEqual(state, { sandbox: true, nodeIntegration: false, contextIsolation: true });
  assert.equal(await page.evaluate(() => typeof window.require), 'undefined');
  const samples = await page.evaluate(async () => (await fetch('./samples/index.json')).json());
  assert.ok(Array.isArray(samples) ? samples.length : samples.samples?.length, 'Offline sample index should contain scenes');
  const health = await page.evaluate(async () => { const res = await fetch('./local-api/api/health'); return { status: res.status, body: await res.json() }; });
  assert.ok(health.status === 200 || health.status === 503);
  if (process.env.DW_ONNX_MODEL) {
    await expect.poll(async () => page.evaluate(async () => (await fetch('./local-api/api/health')).status), { timeout: 120_000 }).toBe(200);
    // Electron's native picker is outside Playwright's filechooser interception.
    // Keep the real input/change flow, but supply the file through the DOM.
    await page.evaluate(() => {
      const click = HTMLInputElement.prototype.click;
      HTMLInputElement.prototype.click = function () { if (this.type !== 'file') click.call(this); };
    });
    await page.getByRole('button', { name: 'Open image…' }).click({ force: true });
    const image = await desktop.evaluate(({ nativeImage }, filename) => nativeImage.createFromPath(filename).resize({ width: 128, height: 128 }).toPNG().toString('base64'), resolve('e2e/fixtures/rgb.png'));
    await page.locator('input[type=file]').last().setInputFiles({ name: 'desktop-smoke.png', mimeType: 'image/png', buffer: Buffer.from(image, 'base64') });
    const run = page.getByRole('button', { name: 'Estimate heights' });
    await expect(run).toBeEnabled();
    await run.click({ force: true });
    await expect(page.getByRole('status', { name: 'Model run progress' })).toBeVisible();
    await expect(page.getByRole('status', { name: 'Model run progress' })).toHaveCount(0, { timeout: 90_000 });
    await expect(page.getByRole('contentinfo', { name: 'Status bar' })).toContainText(/\d+k? verts/);
    console.log('Desktop local prediction passed: image upload → ONNX → height grid → rendered scene.');
  }
  assert.deepEqual(errors, []);
  console.log(`Desktop smoke passed: sandboxed renderer, offline samples, ${process.env.DW_ONNX_MODEL ? 'real local model and rendered mesh' : `initial model status ${health.status}`}.`);
} catch (error) {
  if (desktop) {
    const page = await desktop.firstWindow();
    console.error((await page.locator('body').innerText()).slice(-3000));
    await page.screenshot({ path: join(tmpdir(), 'depthwizard-desktop-smoke.png') }).catch(() => {});
  }
  throw error;
} finally {
  await desktop?.close();
  await rm(userData, { recursive: true, force: true, maxRetries: 5, retryDelay: 500 });
}
