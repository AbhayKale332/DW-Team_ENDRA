import { expect, test, type Page } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

const SHOTS = process.env.DW_SHOTS;

// Software WebGL in CI is slow: start with the lightest mesh and no post-processing.
test.beforeEach(async ({ page }) => {
  test.setTimeout(240_000);
  await page.addInitScript(() => {
    if (!localStorage.getItem('dw.settings'))
      localStorage.setItem('dw.settings', JSON.stringify({ state: { quality: 'fast', postFx: false, provider: 'mock' }, version: 1 }));
  });
});

function collectErrors(page: Page) {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error' && !/Failed to load resource|favicon/i.test(m.text())) errors.push(`console: ${m.text()}`);
  });
  return errors;
}

async function waitForMesh(page: Page) {
  await expect(page.getByText('Building 3D mesh…')).toHaveCount(0, { timeout: 60_000 });
  await expect(page.getByRole('button', { name: /Compass/ })).toBeVisible();
  await page.waitForTimeout(1500);
}

test('empty state, header menus and accessibility', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/?mock');
  await expect(page.getByText('Estimate heights from a single image')).toBeVisible();
  await expect(page.getByRole('button', { name: /3D DSM View/ })).toBeVisible();
  for (const m of ['File', 'View', 'Tools', 'Help']) await expect(page.getByRole('menuitem', { name: m, exact: true })).toBeVisible();
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/01-empty.png` });
  const axe = await new AxeBuilder({ page }).exclude('canvas').analyze();
  const serious = axe.violations.filter((v) => v.impact === 'serious' || v.impact === 'critical');
  expect(serious.map((v) => `${v.id}: ${v.nodes.length}`)).toEqual([]);
  expect(errors).toEqual([]);
});

test('sample scene: 3D, height map, image views, probe and validation', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/?mock');
  await page.getByRole('button', { name: /Try a sample scene/ }).click();
  await page.getByRole('menuitem', { name: /River valley/ }).click();
  await waitForMesh(page);
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/02-valley-3d.png` });

  await page.keyboard.press('2');
  await page.waitForTimeout(1200);
  await expect(page.getByRole('button', { name: /Primary view: Height Map/ })).toBeVisible();
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/03-heightmap.png` });

  // Probe the centre of the map.
  await page.keyboard.press('p');
  const box = (await page.locator('#dw-canvas').boundingBox())!;
  await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
  await expect(page.getByRole('tab', { name: 'Analysis' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByText('Height above ground').first()).toBeVisible();

  await page.keyboard.press('3');
  await page.waitForTimeout(800);
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/04-image.png` });

  // Synthetic city ships with an exact reference → validation metrics.
  await page.getByRole('menuitem', { name: 'File', exact: true }).click();
  await page.getByRole('menuitem', { name: 'Sample scenes' }).hover();
  await page.getByRole('menuitem', { name: /Synthetic city/ }).click();
  await waitForMesh(page);
  await page.getByRole('tab', { name: 'Validation' }).click();
  await expect(page.getByRole('table', { name: 'Validation metrics' })).toBeVisible();
  await expect(page.getByText('RMSE').first()).toBeVisible();
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/05-validation.png` });
  expect(errors).toEqual([]);
});

test('mock inference run shows staged progress and produces a scene', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/?mock');
  const chooser = page.waitForEvent('filechooser');
  await page.getByRole('button', { name: 'Open image…' }).click();
  await (await chooser).setFiles('public/samples/buildings_large_campus/rgb.png');
  const run = page.getByRole('button', { name: 'Estimate heights' });
  await expect(run).toBeEnabled();
  await run.click();
  await expect(page.getByRole('status', { name: 'Model run progress' })).toBeVisible();
  await expect(run).toBeDisabled();
  await waitForMesh(page);
  await expect(page.getByRole('status', { name: 'Model run progress' })).toHaveCount(0);
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/06-after-run.png` });
  expect(errors).toEqual([]);
});

test('dark theme and flight simulator HUD', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/?mock');
  await page.getByRole('button', { name: /Try a sample scene/ }).click();
  await page.getByRole('menuitem', { name: /River valley/ }).click();
  await waitForMesh(page);
  await page.getByRole('button', { name: /Switch to dark theme/ }).click();
  await page.waitForTimeout(800);
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/07-dark.png` });
  await page.keyboard.press('g');
  await expect(page.getByRole('toolbar', { name: /Flight simulator controls/ })).toBeVisible();
  await page.waitForTimeout(1500);
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/08-flight.png` });
  await page.keyboard.press('Escape');
  expect(errors).toEqual([]);
});
