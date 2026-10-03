const path = require('node:path');
const fs = require('node:fs');

module.exports = {
  appId: 'org.depthwizard.desktop',
  productName: 'DepthWizard',
  artifactName: 'DepthWizard-${version}-${os}-${arch}.${ext}',
  directories: { output: 'release' },
  asar: true,
  npmRebuild: false,
  files: ['dist/**/*', '!dist/**/*.map', 'desktop/*.mjs', '!desktop/*.test.mjs', '!desktop/smoke.mjs', 'desktop/icons/icon.png', 'server/*.mjs', '!server/*.test.mjs', 'package.json', '!node_modules/**/*'],
  extraResources: [
    { from: 'desktop/resources/backend', to: 'backend', filter: ['**/*', '!.gitkeep'] },
    { from: process.env.DW_MODEL_DIR || 'desktop/resources/model', to: 'model', filter: ['**/*', '!.gitkeep'] },
  ],
  beforePack: async () => {
    const executable = process.platform === 'win32' ? 'depthwizard-inference.exe' : 'depthwizard-inference';
    if (!fs.existsSync(path.join(__dirname, 'desktop/resources/backend/depthwizard-inference', executable))) {
      throw new Error('Build the native inference runtime first: npm run desktop:backend (see desktop/README.md).');
    }
  },
  win: { target: [{ target: 'nsis', arch: ['x64'] }], executableName: 'DepthWizard', icon: 'desktop/icons/icon.ico' },
  nsis: { oneClick: false, perMachine: false, allowToChangeInstallationDirectory: true, createDesktopShortcut: true, createStartMenuShortcut: true, deleteAppDataOnUninstall: false },
  mac: { target: ['dmg', 'zip'], category: 'public.app-category.graphics-design', hardenedRuntime: true, icon: 'desktop/icons/icon.png' },
  linux: { target: [{ target: 'AppImage', arch: ['x64'] }], category: 'Graphics', executableName: 'depthwizard', icon: 'desktop/icons/icon.png' },
  publish: null,
};
