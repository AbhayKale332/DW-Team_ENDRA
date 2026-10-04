import { app, BrowserWindow, dialog, Menu, shell } from 'electron';
import { spawn } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { existsSync, createWriteStream } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { desktopHandler } from './local-api.mjs';
import { findModel, installModel, modelFiles } from './model.mjs';

const { createAppServer } = await import(app.isPackaged ? '../server/serve.mjs' : '../../Frontend/server/serve.mjs');
const here = dirname(fileURLToPath(import.meta.url));
let window, server, child, log, graph, origin;
let backend = { message: 'No model installed. Use Model → Install ONNX model…' };
let installing = false;
if (process.env.DW_USER_DATA) app.setPath('userData', resolve(process.env.DW_USER_DATA));
const singleInstance = app.requestSingleInstanceLock();
if (!singleInstance) app.quit();

function stopBackend() {
  const previous = child;
  child = null;
  backend = { message: 'Local model stopped.' };
  previous?.kill('SIGKILL');
}

async function startBackend() {
  stopBackend();
  if (!graph) { backend.message = 'No model installed. Use Model → Install ONNX model…'; return; }
  try {
    await modelFiles(graph);
    const executable = join(app.isPackaged ? process.resourcesPath : join(here, 'resources'), 'backend', 'depthwizard-inference', process.platform === 'win32' ? 'depthwizard-inference.exe' : 'depthwizard-inference');
    const python = process.env.DW_PYTHON;
    if (!existsSync(executable) && !python) throw new Error('Inference runtime missing. Build it with npm run desktop:backend, or set DW_PYTHON for development.');
    const token = randomBytes(32).toString('hex');
    backend = { token, message: 'Loading local ONNX model…' };
    const args = ['--onnx', graph];
    const proc = spawn(python && !app.isPackaged ? python : executable, python && !app.isPackaged ? [join(here, 'inference.py'), ...args] : args, {
      windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, DW_DESKTOP_TOKEN: token, DW_JOBS_DIR: join(app.getPath('userData'), 'jobs'), HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1', PROJ_NETWORK: 'OFF', PYTHONUNBUFFERED: '1', DW_CKPT: '', DW_ONNX: '' },
    });
    child = proc;
    let pending = '';
    proc.stdout.on('data', (data) => {
      log.write(data);
      pending += data.toString();
      const lines = pending.split(/\r?\n/);
      pending = lines.pop();
      for (const line of lines) {
        if (child === proc && /^DW_READY \d+$/.test(line)) backend = { token, port: Number(line.slice(9)), message: 'Local model ready' };
      }
    });
    proc.stderr.on('data', (data) => log.write(data));
    const failed = (message) => { if (child === proc) { backend = { message: `${message} See Model → Open application data → inference.log.` }; child = null; } };
    proc.on('error', (error) => failed(error.message));
    proc.on('exit', (code) => failed(`Inference service exited (${code}).`));
  } catch (error) {
    backend = { message: error.message };
  }
}

async function importModel() {
  if (installing) return;
  installing = true;
  try {
    const selection = await dialog.showOpenDialog(window, { title: 'Select ONNX graph (keep its .json and weight files beside it)', properties: ['openFile'], filters: [{ name: 'ONNX model', extensions: ['onnx'] }] });
    if (selection.canceled) return;
    graph = await installModel(selection.filePaths[0], app.getPath('userData'));
    await startBackend();
    await dialog.showMessageBox(window, { type: 'info', message: 'Model installed', detail: 'The local model is loading. Select Local ONNX model in File → Settings, then Test connection. Replacing a model stops any active prediction.' });
  } catch (error) { dialog.showErrorBox('Could not install model', error.message); }
  finally { installing = false; }
}

function createWindow() {
  window = new BrowserWindow({ width: 1440, height: 900, minWidth: 900, minHeight: 600, title: 'DepthWizard', icon: join(here, 'icons', 'icon.png'), backgroundColor: '#f4f5f7', show: true, webPreferences: { nodeIntegration: false, contextIsolation: true, sandbox: true, webSecurity: true, backgroundThrottling: false } });
  window.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//.test(url)) void shell.openExternal(url);
    return { action: 'deny' };
  });
  window.webContents.on('will-navigate', (event, url) => { if (new URL(url).origin !== origin) event.preventDefault(); });
  // Save downloads with a native dialog rather than navigating the app to the export.
  window.webContents.session.removeAllListeners('will-download');
  window.webContents.session.on('will-download', (_event, item) => item.setSaveDialogOptions({ title: 'Save DepthWizard export', defaultPath: join(app.getPath('downloads'), item.getFilename()) }));
  void window.loadURL(`${origin}/?desktop`);
  window.on('closed', () => { window = null; });
}

app.on('second-instance', () => { if (window?.isMinimized()) window.restore(); window?.focus(); });
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
app.on('activate', () => { if (!window && origin) createWindow(); });
app.on('before-quit', () => { stopBackend(); server?.close(); log?.end(); });

if (singleInstance) app.whenReady().then(async () => {
  const data = app.getPath('userData');
  await mkdir(data, { recursive: true });
  log = createWriteStream(join(data, 'inference.log'), { flags: 'a' });
  const config = await readFile(join(data, 'model.json'), 'utf8').then(JSON.parse).catch(() => ({}));
  graph = process.env.DW_ONNX_MODEL || config.graph || await findModel(join(app.isPackaged ? process.resourcesPath : join(here, 'resources'), 'model'));
  // A stable saved port preserves IndexedDB projects and localStorage across restarts.
  const port = Number(await readFile(join(data, 'port'), 'utf8').catch(() => 0));
  server = createAppServer({ desktopHandler: desktopHandler(() => backend) });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(port, '127.0.0.1', resolve); });
  const address = server.address();
  origin = `http://127.0.0.1:${address.port}`;
  await writeFile(join(data, 'port'), String(address.port));
  server.on('error', (error) => dialog.showErrorBox('Desktop server failed', error.message));
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    ...(process.platform === 'darwin' ? [{ role: 'appMenu' }] : []),
    { label: 'Model', submenu: [{ label: 'Install ONNX model…', click: () => void importModel() }, { label: 'Restart inference service', click: () => void startBackend() }, { label: 'Open application data', click: () => void shell.openPath(data) }] },
    { role: 'editMenu' }, { role: 'viewMenu' }, { role: 'windowMenu' },
    { label: 'Help', submenu: [{ label: 'About DepthWizard', click: () => void dialog.showMessageBox(window, { message: `DepthWizard ${app.getVersion()}`, detail: 'Local ONNX inference, 3D terrain, analysis and exports. Install a model using the Model menu. Online basemaps and OSM need internet access.' }) }, { role: 'quit' }] },
  ]));
  createWindow();
  await startBackend();
}).catch((error) => { dialog.showErrorBox('Could not start DepthWizard', error.message); app.quit(); });
