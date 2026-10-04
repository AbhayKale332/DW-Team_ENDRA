import { beforeEach, expect, it, vi } from 'vitest';
vi.mock('@/api/registry', () => ({ IS_DESKTOP: true, DEFAULT_SPACE_ID: 'test/space' }));
import { configureDesktopInference, useSettings } from './settings';

beforeEach(() => {
  useSettings.setState({ provider: 'depthwizard-serve', desktopModelKey: undefined });
});
it('selects hosted or local automatically, preserving a user choice until the model changes', async () => {
  let modelKey: string | null = null;
  vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => ({ modelKey }) })));
  await configureDesktopInference();
  expect(useSettings.getState().provider).toBe('gradio-space');
  modelKey = '/bundled/depthwizard.onnx';
  await configureDesktopInference();
  expect(useSettings.getState().provider).toBe('depthwizard-serve');
  useSettings.getState().set({ provider: 'gradio-space' });
  await configureDesktopInference();
  expect(useSettings.getState().provider).toBe('gradio-space');
  modelKey = '/imported/new-model.onnx';
  await configureDesktopInference();
  expect(useSettings.getState().provider).toBe('depthwizard-serve');
  vi.unstubAllGlobals();
});
