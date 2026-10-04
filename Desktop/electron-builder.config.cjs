const path = require('node:path');
const fs = require('node:fs');
const bundleModel = process.env.DW_BUNDLE_MODEL !== '0';
const variant = bundleModel ? 'with-model' : 'without-model';
const platform = { darwin: 'mac', win32: 'windows', linux: 'linux' }[process.platform];

module.exports = {
  appId: 'org.depthwizard.desktop',
  productName: 'DepthWizard',
  artifactName: 'DepthWizard-${version}-${os}-${arch}-' + variant + '.${ext}',
  directories: { output: `${platform}/release/${variant}` },
  asar: true,
  npmRebuild: false,
  files: [
    'shared/*.mjs', '!shared/*.test.mjs', '!shared/smoke.mjs', 'shared/icons/icon.png',
    'package.json', '!node_modules/**/*',
    { from: '../Frontend/dist', to: 'dist', filter: ['**/*', '!**/*.map', '!samples/**/*'] },
    // Offline installers include only these two projects and their quick looks.
    // The desktop server generates the index from the packaged folders.
    ...['PNG-Wankhede_Stadium_Mumbai', 'GeoReferenced .tif From Cartosat2S'].map((sample) => ({
      from: `../Frontend/dist/samples/${sample}`, to: `dist/samples/${sample}`,
      filter: ['*.dwproj', 'input.png', 'input.jpg', 'input.jpeg', 'height.png'],
    })),
    { from: '../Frontend/server', to: 'server', filter: ['*.mjs', '!*.test.mjs'] },
  ],
  extraResources: [
    { from: 'shared/resources/backend', to: 'backend', filter: ['**/*', '!.gitkeep'] },
    ...(bundleModel ? [{ from: process.env.DW_MODEL_DIR || 'shared/resources/model', to: 'model', filter: ['**/*', '!.gitkeep'] }] : []),
  ],
  beforePack: async () => {
    if (bundleModel) {
      const directory = path.resolve(__dirname, process.env.DW_MODEL_DIR || 'shared/resources/model');
      const graphs = fs.existsSync(directory) ? fs.readdirSync(directory).filter((name) => name.endsWith('.onnx')) : [];
      if (graphs.length !== 1 || !fs.existsSync(path.join(directory, `${graphs[0]}.json`))) {
        throw new Error('Model-included installers require one ONNX graph and its .onnx.json metadata. Stage a verified model first.');
      }
    }
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
