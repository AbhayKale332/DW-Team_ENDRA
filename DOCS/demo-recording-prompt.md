# Prompt: record the DepthWizard app demos with ScreenKit

Copy everything below the line into the recording agent.

---

You are recording screen demos of **DepthWizard**, a web app made by Team ENDRA for Smart India Hackathon 2026 (ISRO, problem statement 26175). The clips go into the team's video, which has a voice-over. Part 6 of that video (3:25–4:05) and one shot in Part 5 must be real screen recordings of the app. Use the **ScreenKit MCP** to record them.

## 0. Before you record anything

1. **List the ScreenKit MCP tools** and read their schemas. Work out how to: choose the screen, window or region; set resolution and frame rate; start, pause and stop a recording; save to a path; and add cursor highlights or zoom, if it supports them. Use only tools that exist. Do not guess tool names.
2. **Make one 5-second test recording** of the app window. Check it saved, plays, is 1920×1080, and that the 3D view is smooth, with no black canvas or tearing. Fix settings before going on.
3. If ScreenKit cannot drive the mouse and keyboard, then use whatever browser or computer-control tools you have for input, and use ScreenKit only to record. If you have no input tool at all, stop and tell the user. Do not record a static screen.

## 1. App setup

- Repo: `/home/abhay/F/7th SEM/SIH/DepthWizard`. App: `Frontend/` (React + Vite + three.js).
- Start it: `cd Frontend && npm run dev` → `http://localhost:5173`. `node_modules` is already installed. Do not reinstall it, because the disk is nearly full.
- **Live model runs need a Hugging Face token.** `Frontend/.env` does not exist yet, only `.env.example`. If the user has not made `.env` with `HF_TOKEN=...`, ask them to. Never print, log or show the token on screen. Check the model is reachable: the status dot in the header should be green, or use **File → Settings → Test connection**.
- **No token? Use the precomputed sample scenes** (they work offline): the viewport's *Try a sample scene* button or **File → Sample scenes**. The two samples are `buildings_large_campus` (with OpenStreetMap and points-of-interest data) and `Wankhede_Stadium_Mumbai`.
- Input images for a live run: `cartosat_2S_Sample/testing_crops/`. Each one is a `.png` and a georeferenced `.tif`. Good choices: `2s_stadium.tif`, `2s_buildings_dense_city.tif`, `2s_buildings_large_campus.tif`, `2s_commercial_blocks.tif`, `2e_forest_hills.tif` (has terrain). Use a `.tif` so the georeferenced features (basemap, OSM, GeoTIFF export) work.

## 2. Recording rules

- **1920×1080, 60 fps** (30 fps if 60 stutters), H.264 MP4. Record the browser window's content only: no OS taskbar, no browser tabs or address bar. Use full screen (F11) or a kiosk window.
- Browser zoom at 100%. Light theme, which is the app default. Turn off notifications. Close other tabs. No bookmarks bar.
- **Move the mouse slowly and on purpose.** Pause about 1 s before and after each click. No hunting around. Viewers need to follow every click.
- Each clip is **one action, 4–12 s long**, with 1 s of still frame at both ends so the editor can cut cleanly. Record each clip as its own file. Do not record one long take.
- Do 2 takes of each clip and keep the better one. Re-record any take with a loading spinner stuck on screen, an error toast, a half-loaded scene or a visible token.
- Nothing fake: no mock mode (`?mock`) in the final clips, and no edited numbers. Judges must see the real app.
- Save to `DOCS/demo-recordings/` with the file names below. Do not commit anything.

## 3. Shot list

Each clip lists the voice-over line it plays under, so match its length and pacing.

### Part 6 — The 3D viewer (main clips)

| File | Voice-over line | What to record |
|---|---|---|
| `D1_upload_to_3d.mp4` | "In our app, you just upload an image. DepthWizard builds a 3D model and drapes the original photo on top." | Empty app → drop `2s_stadium.tif` on *Drop an image or click to choose one* → click run in the project panel → progress stepper → 3D model appears. Model wait over ~10 s: record it all; the editor will speed it up. No token: open a sample scene from **File → Sample scenes** instead and record its loading card → 3D. |
| `D2_orbit.mp4` | "You can orbit around it…" | Slow orbit drag around the scene for about 180°, then a gentle zoom in toward the tallest building. |
| `D3_walk.mp4` | "…walk through in first-person mode…" | Press **F** (first-person walk). Click into the canvas, then walk slowly between buildings for 6–8 s. Press **Esc** to leave. |
| `D4_drone_tour.mp4` | "…or take an automatic drone tour." | Press **T** (drone tour). Let it fly 8–10 s untouched, then **Esc**. |
| `D5_probe_height.mp4` | "Click any point to see its height." | Pick the *Probe height* tool. Click a tall building roof, then a road, so the viewer sees two different heights. |
| `D6_measure_slope.mp4` | "Measure the slope between two points…" | *Measure distance & slope* tool: click the ground, then a rooftop. Hold on the distance/slope readout for 2 s. |
| `D7_profile.mp4` | "…or draw an elevation profile." | *Elevation profile* tool: draw a line across several buildings. Hold on the profile chart, and move the cursor along the chart so the marker moves on the terrain. |
| `D8_validation.mp4` | "If you have reference data, the app compares it and shows the error live." | **Inspector → Validation** tab: load a reference raster (see *Open questions*) → metrics appear (RMSE etc.) → turn on *Compare swipe (prediction \| reference)* and drag the swipe slowly across. |
| `D9_offline_sample.mp4` | "And it runs fully offline…" | See *Open questions* first. If cleared: turn Wi-Fi off on screen, then open a sample scene and orbit it. |

### Part 5 — the QGIS shot

| File | Voice-over line | What to record |
|---|---|---|
| `D10_export_geotiff.mp4` | "The result is a standard GeoTIFF…" | With a georeferenced result open, press **Ctrl+E** (or **File → Export**), choose *GeoTIFF — height above ground (nDSM)* and save it to `DOCS/demo-recordings/`. |
| `D11_qgis.mp4` | "…that opens in any GIS software." | Open that GeoTIFF in QGIS over an OpenStreetMap base layer, to show it lands in the right place. Apply a pseudocolour ramp. If QGIS is not installed, skip this clip and tell the user. Do not install it. |

### B-roll extras (record if time allows; the editor uses them as cutaways)

| File | What to record |
|---|---|
| `B1_view_modes.mp4` | Press **1** (3D DSM) → **2** (height map) → **3** (input image) → **1**. |
| `B2_layers.mp4` | **Inspector → Layers**: switch the drape layer through optical, height tint, relief shading, slope and contours. |
| `B3_3d_objects.mp4` | Toggle the 3D object layers (buildings, trees) from the floating toolbar; press **O** to switch all off and back on. |
| `B4_surroundings.mp4` | Turn on *Surrounding basemap* and the *OpenStreetMap overlay* / *Facilities*. Needs a georeferenced scene; `buildings_large_campus` has the OSM data. |
| `B5_flood.mp4` | Use-cases panel → **Flood** scenario: raise *Water level rise* slowly so water spreads over low ground. |
| `B6_telecom.mp4` | Use-cases panel → **Telecom** scenario: add a tower at the centre and show coverage on the terrain. |
| `B7_flight.mp4` | Press **G** (flight simulator) and fly a short pass over the scene, then **Esc**. |
| `B8_exports_menu.mp4` | Open the export list and hover slowly over GeoTIFF, GLB, OBJ, PLY, STL, PNG. Do not save anything. |

### Shortcuts you will need

`1/2/3` view mode · `R` reset camera · `N` face north · `+/−` zoom · `F` walk · `G` flight · `T` tour · `Esc` leave mode · `O` objects on/off · `[` / `]` hide panels · `Ctrl+O` open · `Ctrl+E` export GeoTIFF · `?` shortcut list.

Hide the side panels (`[` and `]`) for the cinematic clips (D2, D3, D4, B7) so the 3D view fills the frame. Keep them open for the tool and validation clips.

## 4. Open questions (ask the user, do not guess)

1. **Reference raster for D8.** Which file is the ground-truth DSM/nDSM for the scene you record? It must cover the same area as the input. If none exists, skip D8 and say so.
2. **The "runs fully offline" line (D9).** Live inference runs on a Hugging Face Space, so a new image needs internet. Only the precomputed sample scenes open offline. Tell the user this and let them choose: reword the line (for example, "sample scenes open offline, and the model can run on a laptop without a GPU"), or drop D9. Do not record a shot that suggests live inference works offline.
3. **HF token.** If there is no `Frontend/.env`, D1 uses the sample-scene fallback unless the user adds a token.

## 5. When you are done

Report back with a table: file name, length in seconds, and which take you kept, with anything skipped and why. Then stop the dev server, and delete your test recordings and failed takes.
