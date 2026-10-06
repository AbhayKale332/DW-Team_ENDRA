# SIH submission presentation review

Reviewed 5 October 2026: `SIH2026-IDEA-Presentation-Format.pptx (2).pdf`, all six rendered pages, against `CompetitionContext/ProblemStatment.json` and `CompetitionContext/CompitionHostMessage.md`. Supporting checks used the current host FAQ, local evaluation reports, published metric artifacts, desktop documentation, and selected implementation files. The PDF was not edited. This is a presentation and evidence review, not a fresh model benchmark or an end-to-end application test.

The concept covers most requested functionality. The highest priority is to make the accuracy evidence precise, show the missing evaluation/deployment evidence, and remove comparisons that overstate what the current results establish.

## Requirement coverage

| Requirement | Evidence in slides | Assessment / improvement |
|---|---|---|
| Single RGB image as input | 2–3 | Covered. |
| PNG/JPG and TIFF input | 2–3 | Covered; route using actual spatial metadata, since TIFF alone does not guarantee georeferencing. |
| Non-georeferenced input → rDSM | 2–3 | Covered, but small. Make this branch visible in the solution summary. |
| Georeferenced input → absolute metric DSM | 2–3, 5 | Covered as a feature; current slide 4 metrics do not validate this output. |
| Pretrained model and remote-sensing adaptation | 2–3 | Covered. DINOv3 is permitted under the host's clarified methodology rules. |
| DEM/GCP calibration | 2–3 | Covered; fix the simplistic `nDSM + DEM` explanation. |
| Original optical texture projected onto mesh | 2–3, 6 | Covered. Demonstrate texture alignment with an input/output pair. |
| First-person navigation and aerial perspectives | 2–3, 5 | Covered. |
| Structural heights and slopes | 2–3, 5 | Covered. Distinguish above-ground height from absolute elevation. |
| RMSE, MAE, correlation | 4 | Present, but target, split, inference settings and correlation aggregation need correction. |
| Urban, sparse, hilly, forested evaluation | 4 | Urban/sparse/forested shown; hilly terrain missing from the table. |
| Rendering fidelity, usability and stability | Feature lists and screenshots | Weak evidence. Add a compact demonstration/check summary. Numeric FPS is a useful addition, not an explicitly required metric. |
| Successful standalone deployment | Offline claim; Docker/ONNX mentioned | Ambiguous. Name the actual local application and link the distributable or launch instructions. |
| Standard geospatial output | 3 flowchart | GeoTIFF DSM/nDSM/DTM export is already present. Make CRS, grid and vertical datum handling explicit. |
| Complete source code and documentation | 6 | Links present; user confirmed the source repository is intentionally private. Arrange judge access when required. |
| Submission format | Entire file | Six slides including cover, PDF: matches the instructions in the supplied template. |

## Priority corrections

### 1. Slide 4: the table measures nDSM, not absolute DSM

The heading “DSM Accuracy vs LiDAR / Reference” overstates the meaning of the numbers. Your own [landscape documentation](../Docs-Site/src/content/docs/results/landscape-accuracy.mdx) identifies these targets as height above ground (nDSM).

Use **“Above-ground height accuracy (nDSM), validation data”** for this table. Give absolute DSM agreement its own label and evidence. Reference DSM accuracy after DEM anchoring cannot be inferred from the model's nDSM RMSE alone.

The current table combines two protocols:

- First three numeric columns: epoch-7, single-pass, four validation datasets, 718 tiles and 183,717,738 valid pixels.
- GAMUS RMSE column: all 859 GAMUS validation tiles, deterministic 512-pixel centre crops, D4 averaging plus 1.5× input scaling, 222,847,123 valid pixels.

The `~184 M pixels` footnote does not describe the GAMUS column. Either separate the tables or provide two explicit footnotes. Replace “held-out sets” with “validation splits”; MVS3DM and NEON validation contributed to checkpoint selection. These results are not a separate held-out test score.

### 2. Slide 4: explain the correlation calculation

The presentation uses forested `r = 0.84` and overall `r = 0.89`. Those correspond to an unweighted mean of dataset correlations. Your current documentation uses a pixel-weighted mean, which gives `0.83` and `0.88`. Neither averaging method produces exact pooled Pearson correlation.

If keeping the table, match the documented method and label the column **“Weighted mean dataset r”**. Alternatively, compute exact pooled Pearson r from the full sufficient statistics and label it accordingly. Do not change only the numbers while retaining the implication of an exact pooled statistic.

Corrected four-dataset table under the documented weighted-mean convention:

| Landscape | nDSM RMSE (m) | MAE (m) | Weighted mean dataset r |
|---|---:|---:|---:|
| Urban | 3.85 | 1.69 | 0.90 |
| Sparse | 1.32 | 0.32 | 0.63 |
| Forested | 2.85 | 1.60 | 0.83 |
| Overall | 3.04 | 1.32 | 0.88 |

For a simpler presentation, use the internally consistent GAMUS-only table below and link the four-dataset results:

| Landscape | nDSM RMSE (m) | MAE (m) | Pearson r |
|---|---:|---:|---:|
| Urban | 3.06 | 1.64 | 0.930 |
| Sparse | 1.25 | 0.32 | 0.736 |
| Forested | 2.05 | 0.98 | 0.798 |
| Overall | 2.59 | 1.26 | 0.924 |

Caption: **“GAMUS validation: 859 tiles, 512 px centre crops, D4 + 1.5× inference; above-ground heights.”** Numbers come from [the saved GAMUS evidence](../Docs-Site/public/evidence/v5-gamus-d4/probe_metrics.json). Keep the slower inference setting attached to the accuracy claim; do not pair its accuracy with a faster configuration's timing.

### 3. Slides 3–4: add hilly and absolute-elevation evidence

Mentioning NEON as “forested + hilly” does not demonstrate hilly-landscape accuracy. Your classifier omits hilly terrain because terrain is removed from the nDSM targets.

You already have relevant preliminary evidence in [the Cartosat report](CompetitionContext/V5_Cartosat_Test_Results.md): nine Cartosat-2E hill crops, with 3.849 m RMSE against SRTM at approximately 30 m block resolution. This can be presented separately as **“Coarse absolute DSM agreement on hilly Cartosat scenes.”** It does not establish high-resolution terrain or building-height accuracy. Add MAE and correlation from the corresponding artifacts if presenting it as a complete evaluation row.

The report also includes 13 Cartosat-2S crops with 2.783 m SRTM agreement RMSE. This is useful alignment with the stated evaluation sensor, subject to the same qualifications.

Do not advertise near-zero agreement against the very Copernicus reference used to anchor the output as independent accuracy. Your report explicitly calls this self-consistency.

### 4. Slides 2–3: correct the calibration explanation

The slide 3 box says `nDSM + DEM → Absolute DSM`. That is too broad: Copernicus DEM already contains structures and vegetation. Adding structure heights directly can count them twice. The [Copernicus product description](https://registry.opendata.aws/copernicus-dem/) explicitly identifies it as a surface model.

Your [calibration code](../Model_Traning/v5/geo/calibrate.py) supports a more careful explanation. Suggested box text:

> Align imagery and elevation reference → anchor coarse absolute elevation → preserve predicted local height detail → export metric DSM.

If describing the terrain-fitting mode, use `estimated DTM + nDSM → DSM`. Match the diagram to the mode actually demonstrated; slide 2's ground-mask description and the DEM-anchored mode are different routes.

Add a short output statement: **“GeoTIFF with CRS, geotransform, metre units, vertical datum and NoData mask.”** Treat missing DEM/GCP support explicitly; a GeoTIFF's horizontal location metadata alone does not supply vertical elevation.

### 5. Slides 2–4: make standalone deployment visible

“Works without internet” is too broad beside a flowchart that names Hugging Face GPU inference. Your [desktop documentation](../Desktop/README.md) already explains the distinction:

> Desktop/local mode: bundled ONNX model and local DEM; hosted GPU inference available online.

Use a download or local-launch link and state the platform actually verified. The desktop README describes Windows, macOS and Linux targets, but source documentation alone is not evidence that every installer has been demonstrated successfully. Live OSM/basemaps and remote DEM downloads require connectivity unless data is supplied locally.

### 6. Slide 4: shadow checks require optional metadata

The host FAQ says evaluation imagery should be expected without sun angles or acquisition timestamps. Replace “Shadow-length scale check using sun angle” with:

> Optional shadow consistency check when sun-angle metadata is available; DEM/GCP calibration is the primary route.

This is also consistent with your shadow module's warning that unknown sun angle and height scale are ambiguous.

### 7. Slides 4–5: replace unsupported or mismatched comparisons

The cost and timing graphics use incompatible scopes:

- Slide 4 compares a satellite scene priced at ₹5,000 with UAV LiDAR at about ₹5 crore, while slide 5 advertises 59× savings using $129/km² and $2.2/km². These are different comparisons; the presentation does not reconcile geography, area, acquisition, processing or output quality.
- The linked NRSC pricing workshop is from 2022. It is not enough to establish a universal current scene-price ceiling.
- “Every satellite pass (~4 days)” is not a guarantee of a usable new image for every location.
- An eight-year LiDAR refresh cycle and a satellite revisit period are not comparable processing capabilities of your software.
- “~5 minutes for 100 acres” lacks the machine, image resolution, TTA mode and timing scope.
- The `2.8 m` vertical-error bar does not identify its dataset or why it differs from `3.04 m` and `2.59 m` on slide 4. It compares a model benchmark to a LiDAR accuracy figure without a shared evaluation protocol.

Replace these plots with measured project evidence: image dimensions/GSD, hardware, inference setting, time to first usable 3D scene, and appropriately labelled height accuracy. Where no measurement exists, describe the intended benefit qualitatively or mark a target clearly.

### 8. Slide 4: revise the blanket licence claim

DINOv3 uses its own [DINOv3 licence](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md), as also stated in the [official model card](https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m). The whole stack should not be characterized as exclusively BSD/MIT/Apache. Prefer:

> Open-source application libraries; DINOv3 and each dataset used under their respective licence terms.

Keep detailed licence references in documentation. “All training data is CC BY 4.0 / MIT / public” needs a dataset-by-dataset record; public download access alone does not establish a licence category.

### 9. Slide 6: plan judge access to the private source repository

`https://github.com/AbhayKale332/DW-Team_ENDRA` returned HTTP 404 in an unauthenticated request. The user confirmed that the repository is intentionally private, so this is an access consideration rather than an incorrect link. Keep it private during preparation as intended; arrange judge access or provide the source through the authorized submission channel when required. No repository visibility change was requested or made.

The live demo, judges documentation, landscape-accuracy page and references page returned HTTP 200. This checks reachability only, not successful inference, installer execution or video playback.

### 10. Slide 4: describe cloud handling accurately

The slide repeats detection/masking three times and says clouds are excluded from inference. The inspected implementation detects likely clouds using RGB heuristics, fills the input beneath the mask, runs inference, and fills predicted heights in masked areas. It does not recover observed ground geometry hidden by clouds.

Suggested wording: **“Detect likely clouds, mask and fill affected regions, and mark those heights as uncertain.”** If claiming exclusion from validation, verify that the validation mask actually enforces it; the detector alone does not prove this.

## Improvements to evidence and slide balance

Rendering and UX receive half the evaluation weight. Give judges a compact demonstration of texture alignment, fly/walk controls, height/slope measurement, reference upload and stability. Add measured FPS, time to first scene and scene size if available; otherwise label them as planned checks. A short clip of the whole upload → 3D → measurement → validation → export workflow would substantiate more than extra feature names.

Keep flood visualization as an application example, but name it according to what it does. If it raises a water plane and marks intersections, describe an inundation scenario rather than implying validated flood forecasting. Make the connection to emergency assessment clear.

The host's [current FAQ](https://github.com/IMG-PROCESS-SAC/SIH-DepthWizard-2026/issues/1) permits alternative model methods and local web or desktop deployment. It also specifies Cartosat-2S at 0.6 m, asks for broader 0.35–10 m handling, and says additional features receive no extra marks. Summarize this generalization plan without claiming uniform accuracy across that entire resolution range. GAMUS is recommended for development; final evaluation uses separate imagery.

For initial submission, the FAQ places preliminary work on the last slide or in the linked repository. Retain the supplied six-slide structure and consider this distribution:

| Slide | Recommended emphasis |
|---|---|
| 1 | Required identification, live demo and presentation links. |
| 2 | Problem, two input/output branches, and 2–3 specific contributions. |
| 3 | One clear pipeline, dataset roles, calibration and local deployment. |
| 4 | Principal risks and mitigations; implementation feasibility. |
| 5 | Disaster-management benefit and defensible cost/runtime assumptions. |
| 6 | Preliminary results, two clear screenshots, source/docs links, compact research references. |

Your distinctive contribution is better expressed as the combination of satellite features, height/semantic heads, elevation anchoring and an integrated analysis workflow. Generic monocular prediction and a flythrough are requested capabilities. Label OSM, telecom and project sharing as additional features unless you substantiate a specific novelty claim.

## Readability and copy corrections

Slides 2–4 are crowded. The architecture and user-flow diagrams require zooming, while clip art occupies space that could make the technical content readable. Shorten prose, remove repeated claims and decorative pictures, and enlarge the main pipeline. On slide 6, reduce library descriptions to make the prototype evidence and references legible. Preserve required template headings; add the actual product/idea title beside or beneath “IDEA TITLE.”

| Current text | Suggested text |
|---|---|
| `Potential Challanges` | `Potential Challenges` |
| `Global Contex` / `topography contex` | `Global Context` / `topographic context` |
| `accross` | `across` |
| `NVDIA T4` | `NVIDIA T4` |
| `How Our Solution is better?` | `How does our solution improve on existing approaches?` |
| `digital surface model(DSM)` | `digital surface model (DSM)` |
| `Low resolution DEM` | `low-resolution DEM` |
| `.dwproject` on slide 2, `.dwproj` in flowcharts | Use `.dwproj` consistently; this matches the project implementation. |
| `Ungeoreferenced` | `Non-georeferenced` |
| `DSM Model` for the stadium screenshot | `Relative surface reconstruction (rDSM)` |
| `Screen-Shot` | `screenshot` |
| `Github` | `GitHub` |
| `Figure 3 :User Usage Flowchart` | `Figure 3: User workflow` |
| `Edges over-smoothed trees are the worst class` | `Tree boundaries are oversmoothed; tree-height errors remain high.` |
| `Combine the views` | `Fuse multiscale features` to avoid implying multiple input images. |

“Tree RMSE 4.19 m” needs a dataset/checkpoint/inference label. It is not necessarily inconsistent with a lower forest-landscape score, because a tree class and an entire forest tile are different cohorts.

Suggested opening solution text:

> DepthWizard reconstructs a navigable 3D surface from one optical RGB image. Non-georeferenced images produce relative surfaces; georeferenced images use DEM/GCP anchoring to produce metric DSMs. Users inspect heights and slopes, compare reference elevations, and export geospatial results.

## Review limits

No new model evaluation, runtime benchmark, installer test or full UI test was performed. Numerical corrections were checked against existing artifacts. Team ID, portal upload-size limits, all dataset sizes/licences and every external reference were not independently verified. The file is about 10.05 MB (9.58 MiB); check the actual portal limit if relevant. The unauthenticated source-link response is explained by the intentionally private repository.
