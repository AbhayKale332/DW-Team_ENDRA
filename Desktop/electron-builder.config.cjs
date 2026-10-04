const path = require('node:path');
const fs = require('node:fs');
const platform = { darwin: 'mac', win32: 'windows', linux: 'linux' }[process.platform];

module.exports = {
  appId: 'org.depthwizard.desktop',
  productName: 'DepthWizard',
  artifactName: 'DepthWizard-${version}-${os}-${arch}.${ext}',
  directories: { output: `${platform}/release` },
  asar: true,
  npmRebuild: false,
  files: [
    'shared/*.mjs', '!shared/*.test.mjs', '!shared/smoke.mjs', 'shared/icons/icon.png',
    'package.json', '!node_modules/**/*',
    { from: '../Frontend/dist', to: 'dist', filter: ['**/*', '!**/*.map'] },
    { from: '../Frontend/server', to: 'server', filter: ['*.mjs', '!*.test.mjs'] },
  ],
  extraResources: [
    { from: 'shared/resources/backend', to: 'backend', filter: ['**/*', '!.gitkeep'] },
    { from: process.env.DW_MODEL_DIR || 'shared/resources/model', to: 'model', filter: ['**/*', '!.gitkeep'] },
  ],
  beforePack: async () => {
    const executable = process.platform === 'win32' ? 'depthwizard-inference.exe' : 'depthwizard-inference';
    if (!fs.existsSync(path.join(__dirname, 'shared/resources/backend/depthwizard-inference', executable))) {
      throw new Error('Build the native inference runtime first: npm run desktop:backend (see Desktop/README.md).');
    }
  },
  ...require('./windows/config.cjs'),
  ...require('./mac/config.cjs'),
  ...require('./linux/config.cjs'),
  publish: null,
};
