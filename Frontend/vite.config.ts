/// <reference types="vitest/config" />
import { defineConfig, loadEnv, type Connect, type Plugin, type ProxyOptions } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';
import { readFileSync } from 'node:fs';
import { spaceUrlFromId } from './server/space.mjs';
import { relayOverpass } from './server/overpass.mjs';

const pkg = JSON.parse(readFileSync(new URL('./package.json', import.meta.url), 'utf-8')) as { version: string };

/** /overpass/<mirror> → OpenStreetMap Overpass mirror, as server/serve.mjs and api/overpass.js do in production. */
function overpassRelay(): Plugin {
  const handle: Connect.NextHandleFunction = (req, res, next) => {
    const m = /^\/overpass\/([^/?]+)/.exec(req.url ?? '');
    if (!m || req.method !== 'POST') return next();
    const chunks: Buffer[] = [];
    req.on('data', (c: Buffer) => chunks.push(c));
    req.on('end', async () => {
      const r = await relayOverpass(decodeURIComponent(m[1]), Buffer.concat(chunks).toString('utf-8'), req.headers.host ? `http://${req.headers.host}` : undefined);
      res.writeHead(r.status, { 'content-type': r.contentType, 'cache-control': 'no-store' }).end(r.body);
    });
  };
  return {
    name: 'overpass-relay',
    configureServer: (server) => void server.middlewares.use(handle),
    configurePreviewServer: (server) => void server.middlewares.use(handle),
  };
}

export default defineConfig(({ mode }) => {
  // Load ALL variables (no prefix filter) — HF_TOKEN stays in Node and is never exposed to client code.
  const env = loadEnv(mode, process.cwd(), '');
  const spaceUrl = env.HF_SPACE_URL || spaceUrlFromId(env.VITE_SPACE_ID || 'akashch1512/SingleViewHeigthEstimation');
  const token = env.HF_TOKEN;
  if (!token && mode !== 'test') console.warn('\n[depthwizard] HF_TOKEN is not set in .env — requests to the private Space will be rejected.\n');

  // /hf-space/* → private Space, with the access token injected server-side.
  const hfProxy: Record<string, ProxyOptions> = {
    '/hf-space': {
      target: spaceUrl,
      changeOrigin: true,
      secure: true,
      rewrite: (p) => p.replace(/^\/hf-space/, '') || '/',
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    },
  };

  return {
    // Relative base keeps the build portable: static hosts, file:// and a future Tauri shell.
    base: './',
    plugins: [react(), overpassRelay()],
    define: { __APP_VERSION__: JSON.stringify(pkg.version) },
    resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
    server: { proxy: hfProxy },
    preview: { proxy: hfProxy },
    worker: { format: 'es' },
    build: {
      target: 'es2022',
      sourcemap: true,
      chunkSizeWarningLimit: 1500,
      rollupOptions: {
        output: {
          manualChunks(id: string) {
            if (!id.includes('node_modules')) return undefined;
            if (/[\\/](three|@react-three|three-custom-shader-material|postprocessing|three-stdlib|camera-controls)[\\/]/.test(id)) return 'three';
            if (/[\\/](echarts|zrender)[\\/]/.test(id)) return 'charts';
            if (/[\\/](geotiff|proj4)[\\/]/.test(id)) return 'geo';
            if (/[\\/]@mantine[\\/]/.test(id)) return 'mantine';
            return undefined;
          },
        },
      },
    },
    test: {
      environment: 'jsdom',
      include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
    },
  };
});
