# DepthWizard Docs

Documentation site for DepthWizard, built with [Astro Starlight](https://starlight.astro.build).

```bash
npm install
npm run dev       # http://localhost:4321
npm run build     # static site in dist/ (fails on broken internal links)
npm run preview   # serve dist/
```

Requires Node 22.12 or later (see `.nvmrc`).

## Layout

| Path | Contents |
|---|---|
| `src/content/docs/` | Pages (MDX). The sidebar order is set in `astro.config.mjs` |
| `src/components/` | `Chart` (ECharts), `DataTable`, `Stat`/`StatGrid`, `Figure`, `ImageGrid`, `Math` (KaTeX), `VersionTimeline`, `Swatches`, `PrelimBadge`, `CampusViewer` (interactive three.js mini viewer) |
| `src/components/diagrams/` | Hand-drawn, theme-aware SVG diagrams |
| `src/data/metrics.json` | Every charted number, with source paths. Generated, do not edit |
| `src/assets/` | Figures and gallery renders |
| `scripts/extract_metrics.py` | Rebuilds `metrics.json` from `Model_Traning/**/metrics.json`, the MVS3DM summary and the GSD measurements |
| `scripts/make_gallery.py` | Renders gallery images from model output folders |
| `scripts/make_campus_sample.py` | Packs the web app's campus sample into `public/samples/campus/` for the mini viewer |

Flowcharts use Mermaid fenced blocks (` ```mermaid `). Styling follows the site theme through `src/styles/theme.css`.

## Updating results

1. Add or refresh run artefacts under `Model_Traning/`.
2. Edit `RUNS` in `scripts/extract_metrics.py` if a new run should be charted, then run `npm run extract-metrics`.
3. Update the prose on the relevant pages. Label anything that has not been evaluated on held-out data with `<PrelimBadge />`.

## Deploy (Vercel)

Import the repository, set **Root Directory** to `Docs-Site`, and keep the Astro preset. `vercel.json` sets the build command and output directory.
