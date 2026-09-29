import { expect, test, type Page } from '@playwright/test';

const SHOTS = process.env.DW_SHOTS;

// Three mesh builds under software WebGL (objects → raw → objects) take minutes each on a slow machine.
test.beforeEach(async ({ page }) => {
  test.setTimeout(900_000);
  await page.addInitScript(() => {
    if (!localStorage.getItem('dw.settings'))
      localStorage.setItem('dw.settings', JSON.stringify({ state: { quality: 'fast', postFx: false, provider: 'mock' }, version: 1 }));
  });
});

// Scene first (the compass only renders with one), then the mesh: in the other order the check can
// pass before the build has even started.
async function waitForMesh(page: Page) {
  await expect(page.getByRole('button', { name: /Compass/ })).toBeVisible();
  await page.waitForTimeout(300);
  await expect(page.getByText('Building 3D mesh…')).toHaveCount(0, { timeout: 120_000 });
  await page.waitForTimeout(1500);
}

test('sample scene objects: per-class 3D toggles, class layer and summary', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => {
    if (m.type() === 'error' && !/Failed to load resource|favicon/i.test(m.text())) errors.push(`console: ${m.text()}`);
  });

  await page.goto('/?mock');
  await page.getByRole('button', { name: /Try a sample scene/ }).click();
  await page.getByRole('menuitem', { name: /Synthetic city/ }).click();
  await waitForMesh(page);
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/10-objects-3d.png` });

  await page.getByRole('button', { name: 'Toggle inspector' }).click();
  await page.getByRole('tab', { name: 'Layers' }).click();
  // per-class switches; only buildings are on by default
  const buildings = page.getByRole('switch', { name: 'Buildings (8)' });
  const trees = page.getByRole('switch', { name: 'Trees (19)' });
  await expect(buildings).toBeChecked();
  await expect(trees).not.toBeChecked();
  await expect(page.getByRole('switch', { name: 'Water (0)' })).toBeDisabled();
  await trees.click();
  await waitForMesh(page);
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/11-buildings-and-trees.png` });
  // the tool palette button mirrors the switch
  await expect(page.getByRole('button', { name: '3D trees' })).toHaveAttribute('aria-pressed', 'true');
  await page.keyboard.press('o'); // everything off: the raw mesh
  await waitForMesh(page);
  await expect(buildings).not.toBeChecked();
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/12-raw.png` });
  await page.keyboard.press('o'); // back to the previous selection
  await waitForMesh(page);
  await expect(trees).toBeChecked();

  await page.getByRole('textbox', { name: 'Drape layer' }).click();
  await page.getByRole('option', { name: 'Object classes' }).click();
  await expect(page.getByRole('list', { name: 'Object classes legend' })).toBeVisible();
  if (SHOTS) await page.screenshot({ path: `${SHOTS}/13-classes.png` });

  await page.getByRole('tab', { name: 'Info' }).click();
  await expect(page.getByText('Detected objects')).toBeVisible();

  expect(errors).toEqual([]);
});
