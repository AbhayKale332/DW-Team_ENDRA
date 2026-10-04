module.exports = {
  win: { target: [{ target: 'nsis', arch: ['x64'] }], executableName: 'DepthWizard', icon: 'windows/icons/icon.ico' },
  nsis: { oneClick: false, perMachine: false, allowToChangeInstallationDirectory: true, createDesktopShortcut: true, createStartMenuShortcut: true, deleteAppDataOnUninstall: false },
};
