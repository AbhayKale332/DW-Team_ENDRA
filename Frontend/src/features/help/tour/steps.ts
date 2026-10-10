/** One stop of the guided tour. `target` names a `data-tour` attribute; a step without one is a centred card. */
export interface TourStep {
  target?: string;
  title: string;
  body: string;
  /** Preferred side for the card; it flips when that side has no room. */
  side?: 'top' | 'bottom' | 'left' | 'right';
  /** Side panel the target lives in; the tour opens it first. */
  panel?: 'project' | 'inspector';
}

/** Plain-language walkthrough of the screen, in the order a first-time user meets it. */
export const TOUR_STEPS: TourStep[] = [
  {
    title: 'Welcome to DepthWizard',
    body: 'DepthWizard turns a single satellite or aerial photo into a 3D map that shows how tall buildings and trees are. This short tour shows you around. Use Next and Previous to move, or End tour at any time.',
  },
  {
    target: 'input',
    panel: 'project',
    side: 'right',
    title: 'Add your photo',
    body: 'Start here. Drop a photo into this box or click it to pick one from your computer. PNG, JPG and GeoTIFF files all work.',
  },
  {
    target: 'params',
    panel: 'project',
    side: 'right',
    title: 'Check the settings',
    body: 'Tell DepthWizard how much ground one pixel covers. It fills this in by itself when the file knows. Turn on higher quality for a sharper result, or the cloud mask to ignore clouds.',
  },
  {
    target: 'run',
    panel: 'project',
    side: 'right',
    title: 'Make the 3D map',
    body: 'Press Estimate heights. After a few seconds your photo becomes a 3D map.',
  },
  {
    target: 'viewport',
    side: 'left',
    title: 'Explore your map',
    body: 'Your 3D map appears here. Drag to turn it, right-drag to slide it around, and scroll to zoom. Hover over any spot to see how tall it is.',
  },
  {
    target: 'view-switcher',
    side: 'bottom',
    title: 'Change how it looks',
    body: 'Switch between the 3D view, a colour map where colour shows height, and your original photo.',
  },
  {
    target: 'nav-controls',
    side: 'left',
    title: 'Zoom and full screen',
    body: 'Zoom in and out, fill the whole screen with the map, or jump back to the starting view.',
  },
  {
    target: 'inspector',
    panel: 'inspector',
    side: 'left',
    title: 'Layers and details',
    body: 'Layers turns parts of the map on and off, like buildings, trees or nearby roads. Validation compares the result with known heights. Info shows where the photo was taken.',
  },
  {
    target: 'scenarios',
    side: 'bottom',
    title: 'Try a real-world scenario',
    body: 'See what your map can be used for: where a phone tower’s signal would reach, or which areas would flood first.',
  },
  {
    target: 'measure',
    side: 'bottom',
    title: 'Measure things',
    body: 'Measure the distance between two points, the size of an area, or the height of a building.',
  },
  {
    target: 'gcp',
    side: 'bottom',
    title: 'Add known heights',
    body: 'If your photo has no location information, mark a few spots whose real height you know. DepthWizard then adjusts the whole map to match.',
  },
  {
    target: 'raw',
    side: 'bottom',
    title: 'See the raw result',
    body: 'Show the heights exactly as they came out, without the tidy 3D buildings and trees added on top.',
  },
  {
    target: 'menubar',
    side: 'bottom',
    title: 'Open, save and share',
    body: 'File opens photos and sample scenes, saves your project and exports the map. View and Tools hold the same options as the buttons, plus walk, fly and drone-tour modes.',
  },
  {
    target: 'status',
    side: 'bottom',
    title: 'Is the engine ready?',
    body: 'This light shows whether the height engine is online. Green means ready. Click it to change the connection.',
  },
  {
    target: 'panels',
    side: 'bottom',
    title: 'Make more room',
    body: 'Hide or show the side panels to give the map more space.',
  },
  {
    target: 'help',
    side: 'bottom',
    title: 'You’re all set',
    body: 'Open Help and choose Take the tour to see this again. Help also lists the keyboard shortcuts.',
  },
];
