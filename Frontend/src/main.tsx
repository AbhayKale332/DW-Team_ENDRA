import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { MantineProvider, localStorageColorSchemeManager } from '@mantine/core';
import { ModalsProvider } from '@mantine/modals';
import { Notifications } from '@mantine/notifications';
import '@fontsource/inter/400.css';
import '@fontsource/inter/500.css';
import '@fontsource/inter/600.css';
import '@fontsource/inter/700.css';
import '@fontsource/jetbrains-mono/400.css';
import '@mantine/core/styles.css';
import '@mantine/dropzone/styles.css';
import '@mantine/notifications/styles.css';
import './theme/global.css';
import { theme } from './theme/theme';
import App from './App';
import { configureDesktopInference } from './store/settings';

await configureDesktopInference();
window.addEventListener('dw-model-installed', () => void configureDesktopInference().catch(console.error));

const colorSchemeManager = localStorageColorSchemeManager({ key: 'dw.color-scheme' });

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <MantineProvider theme={theme} defaultColorScheme="auto" colorSchemeManager={colorSchemeManager}>
      <ModalsProvider>
        <Notifications position="bottom-right" limit={4} containerWidth={360} />
        <App />
      </ModalsProvider>
    </MantineProvider>
  </StrictMode>,
);
