import { createTheme, type MantineColorsTuple, rem } from '@mantine/core';

/** DepthWizard blue — anchored on #2d6cdf (the backend/report accent). */
const dwBlue: MantineColorsTuple = [
  '#eaf1fd',
  '#d3e1f9',
  '#a6c2f2',
  '#76a1eb',
  '#4f86e5',
  '#3a76e2',
  '#2d6cdf',
  '#205bc6',
  '#1850b1',
  '#08449d',
];

const dwOrange: MantineColorsTuple = [
  '#fff1e8',
  '#fde1d2',
  '#f8c0a3',
  '#f39d70',
  '#ef8046',
  '#ed6d2b',
  '#e5793a',
  '#d1561b',
  '#bb4b16',
  '#a33e0f',
];

/** Sharp, dense, instrument-panel styling: 2 px radii, hairline borders, restrained shadows. */
export const theme = createTheme({
  // Whole UI at 90% (it read too large at 100%). Every rem-based Mantine size follows this; the px values in
  // global.css and the CSS modules, and the icon zoom in global.css, are already at the same 90%.
  scale: 0.9,
  // Breakpoints shrink with it, so layouts switch at the same widths as they would under 90% browser zoom.
  breakpoints: { xs: '32.4em', sm: '43.2em', md: '55.8em', lg: '67.5em', xl: '79.2em' },
  primaryColor: 'dwBlue',
  primaryShade: { light: 6, dark: 5 },
  colors: { dwBlue, dwOrange },
  fontFamily: 'Inter, "Segoe UI", system-ui, -apple-system, sans-serif',
  fontFamilyMonospace: '"JetBrains Mono", ui-monospace, Consolas, monospace',
  headings: { fontFamily: 'Inter, "Segoe UI", system-ui, sans-serif', fontWeight: '600' },
  defaultRadius: 'xs',
  radius: { xs: rem(2), sm: rem(3), md: rem(4), lg: rem(6), xl: rem(8) },
  fontSizes: { xs: rem(12), sm: rem(13), md: rem(14), lg: rem(16), xl: rem(18) },
  spacing: { xs: rem(6), sm: rem(10), md: rem(14), lg: rem(20), xl: rem(28) },
  shadows: {
    xs: '0 1px 2px rgb(15 23 42 / 0.06)',
    sm: '0 1px 3px rgb(15 23 42 / 0.10), 0 1px 2px rgb(15 23 42 / 0.06)',
    md: '0 4px 14px rgb(15 23 42 / 0.12)',
    lg: '0 10px 30px rgb(15 23 42 / 0.16)',
    xl: '0 18px 50px rgb(15 23 42 / 0.2)',
  },
  cursorType: 'pointer',
  focusRing: 'auto',
  autoContrast: true,
  components: {
    Button: { defaultProps: { size: 'sm' } },
    ActionIcon: { defaultProps: { variant: 'subtle', color: 'gray' } },
    Tooltip: { defaultProps: { openDelay: 350, withArrow: false, fz: 'xs' } },
    Menu: { defaultProps: { shadow: 'md', radius: 'xs', transitionProps: { duration: 90 } } },
    Popover: { defaultProps: { shadow: 'md', radius: 'xs' } },
    Modal: { defaultProps: { radius: 'xs', centered: true, overlayProps: { backgroundOpacity: 0.45, blur: 2 } } },
    Paper: { defaultProps: { radius: 'xs' } },
    SegmentedControl: { defaultProps: { radius: 'xs', size: 'xs' } },
    Slider: { defaultProps: { size: 'sm', radius: 'xs', thumbSize: 14 } },
    Select: { defaultProps: { size: 'xs', radius: 'xs', allowDeselect: false } },
    NumberInput: { defaultProps: { size: 'xs', radius: 'xs' } },
    TextInput: { defaultProps: { size: 'xs', radius: 'xs' } },
    PasswordInput: { defaultProps: { size: 'xs', radius: 'xs' } },
    Switch: { defaultProps: { size: 'sm' } },
    Tabs: { defaultProps: { radius: 'xs' } },
    Badge: { defaultProps: { radius: 'xs', variant: 'light' } },
    Notification: { defaultProps: { radius: 'xs' } },
  },
});
