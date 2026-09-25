"""The validation report — one self-contained HTML file per run.

"Validate estimated structural heights against reference datasets" is an explicit
deliverable, and "RMSE, MAE, and correlation against LiDAR/reference data,
including performance stability across urban, sparse, hilly, and forested
landscapes" is the 50 % accuracy rubric verbatim.  This renders exactly that, from
`metrics.json`, with every figure inlined as a data URI so the file can be
emailed, printed to PDF, or opened on an air-gapped machine with no server and no
sibling directory.

It is deliberately blunt about what is *not* known: the class-id mapping is
unverified, the landscape labels are heuristics, and any number that came from a
proxy nDSM says so.  A panel that catches an overclaim discounts everything else
you said; a report that flags its own limits does the opposite.
"""

from __future__ import annotations

import base64
import html
import json
from datetime import datetime, timezone
from pathlib import Path

_CSS = """
:root{color-scheme:light}
*{box-sizing:border-box}
body{margin:0;font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
     color:#1b1f24;background:#f6f7f9}
.wrap{max-width:1000px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:26px;margin:0 0 4px}
h2{font-size:17px;margin:34px 0 10px;padding-bottom:6px;border-bottom:1px solid #e1e4e8}
h3{font-size:14px;margin:20px 0 6px;color:#39424c}
.sub{color:#6a737d;margin:0 0 22px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}
.card{flex:1 1 150px;background:#fff;border:1px solid #e1e4e8;border-radius:10px;padding:14px 16px}
.card .k{color:#6a737d;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.card .v{font-size:24px;font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums}
.card .u{font-size:13px;font-weight:400;color:#6a737d}
.card.warn{border-color:#f0b429;background:#fffaf0}
table{border-collapse:collapse;width:100%;background:#fff;border:1px solid #e1e4e8;
      border-radius:10px;overflow:hidden;font-variant-numeric:tabular-nums}
th,td{padding:7px 11px;text-align:right;border-bottom:1px solid #eef0f2}
th:first-child,td:first-child{text-align:left}
thead th{background:#f0f2f5;font-weight:600;font-size:12px;color:#39424c}
tbody tr:last-child td{border-bottom:0}
figure{margin:16px 0;background:#fff;border:1px solid #e1e4e8;border-radius:10px;padding:12px}
figure img{width:100%;display:block;border-radius:6px}
figcaption{color:#6a737d;font-size:12px;margin-top:8px}
.note{background:#fff8e6;border:1px solid #f0d9a0;border-left:3px solid #f0b429;
      border-radius:6px;padding:10px 14px;margin:14px 0;font-size:13px}
.ok{background:#eefaf0;border-color:#bfe5c8;border-left-color:#34a853}
code{background:#eef0f2;padding:1px 5px;border-radius:4px;font-size:12.5px}
footer{margin-top:40px;color:#6a737d;font-size:12px;border-top:1px solid #e1e4e8;padding-top:14px}
@media print{body{background:#fff}.wrap{max-width:none}figure,table,.card{break-inside:avoid}}
"""


def _img(path: Path) -> str:
    b = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    mime = "jpeg" if Path(path).suffix.lower() in (".jpg", ".jpeg") else "png"
    return f"data:image/{mime};base64,{b}"


# What each gallery source *is*, so a reader knows what they are looking at.
# Matched on the store-name prefix (`dfc23_g050` -> `dfc23`).
_GALLERY_ABOUT = {
    "synrs3d": "SynRS3D — synthetic remote-sensing scenes rendered with exact "
               "per-pixel heights. Clean labels, but not real imagery.",
    "dfc23": "DFC23 Track 2 — real very-high-resolution satellite imagery with "
             "reference nDSMs, the closest open data to the deployment sensor.",
    "india_labeled": "India (labelled) — New Delhi tiles carved out of DFC23, the "
                     "only Indian imagery here with per-pixel reference heights.",
    "gamus": "GAMUS — US aerial imagery, the primary validation set.",
}


def _gallery(figs: Path) -> str:
    idx_path = figs / "gallery.json"
    if not idx_path.is_file():
        return ""
    try:
        index = json.loads(idx_path.read_text())
    except (OSError, ValueError):
        return ""
    body = []
    for e in index:
        store = str(e.get("store", ""))
        about = next((v for k, v in _GALLERY_ABOUT.items()
                      if store == k or store.startswith(k + "_")), "")
        where = f"<code>{html.escape(store)}/{html.escape(str(e.get('split', '')))}</code>"
        if e.get("seen_in_training"):
            where += (" — <b>training tiles</b>: the model saw these, so their error "
                      "flatters it")
        rm = [t["rmse_m"] for t in e.get("tiles") or [] if t.get("rmse_m") is not None]
        tail = f" Mean per-tile RMSE {sum(rm) / len(rm):.2f} m over {len(rm)} tiles." if rm else ""
        fig = _fig(figs / str(e.get("file", "")),
                   f"{html.escape(about)} Tiles from {where}.{tail}")
        if fig:
            body.append(f"<h3>{html.escape(store)}</h3>{fig}")
    if not body:
        return ""
    return ("<h2>Sample tiles by dataset</h2>"
            "<p>A seeded handful of tiles from each source, predicted at native GSD. "
            "Reference and prediction share one height scale per row; grey is "
            "NoData in the reference. These are for looking at, not scoring — a "
            "few tiles say nothing about a dataset's RMSE.</p>" + "".join(body))


def _fig(path: Path, caption: str) -> str:
    if not Path(path).is_file():
        return ""
    return (f'<figure><img alt="{html.escape(caption)}" src="{_img(path)}">'
            f'<figcaption>{caption}</figcaption></figure>')


def _card(k, v, unit="", warn=False) -> str:
    cls = "card warn" if warn else "card"
    u = f'<span class="u"> {html.escape(unit)}</span>' if unit else ""
    return (f'<div class="{cls}"><div class="k">{html.escape(k)}</div>'
            f'<div class="v">{html.escape(str(v))}{u}</div></div>')


def _num(x, nd=3, dash="—"):
    return dash if x is None else f"{float(x):.{nd}f}"


def _table(headers, rows) -> str:
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


# ---------------------------------------------------------------------
def _pick_result(metrics: dict) -> tuple[str, dict]:
    # A held-out test result outranks every val result: `best.pt` was selected
    # on the val prefix, so a `final_*` headline is scored on the set that
    # picked the checkpoint.  `test_*` is not.
    for suffix, label in (("_sliding_tta", "held-out test — sliding window + TTA, native GSD"),
                          ("_tta", "held-out test — centre crop + TTA"),
                          ("_plain", "held-out test — centre crop")):
        for key in metrics:
            if key.startswith("test_") and key.endswith(suffix) and metrics[key]:
                store = key[len("test_"):-len(suffix)].replace("_", "/", 1)
                return f"{label} ({store})", metrics[key]
    for key, label in (("final_sliding_tta", "full-tile sliding window + TTA, native GSD"),
                       ("final_tta", "centre crop + TTA"),
                       ("final_plain", "centre crop")):
        if metrics.get(key):
            return label, metrics[key]
    for h in reversed(metrics.get("history") or []):
        if h.get("val"):
            return f"last validation (epoch {h['epoch']})", h["val"]
    return "no result", {}


def build(run_dir: str | Path, out_path: str | Path | None = None,
          fig_dir: str | Path | None = None, title: str = "DepthWizard v4") -> Path:
    run_dir = Path(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    figs = Path(fig_dir) if fig_dir else run_dir / "figures"
    out_path = Path(out_path) if out_path else run_dir / "validation_report.html"

    cfg = metrics.get("config") or {}
    pre = metrics.get("preproc") or {}
    label, res = _pick_result(metrics)
    g = res.get("global") or {}

    parts = [f"<h1>{html.escape(title)} — validation report</h1>",
             f'<p class="sub">{html.escape(label)} · '
             f'{html.escape(str(cfg.get("datasets", "?")))} · '
             f'{_num(metrics.get("elapsed_min"), 0)} min · '
             f'generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}</p>']

    # ---- headline -----------------------------------------------------
    tall = res.get("tall_gt15m") or {}
    flat = res.get("flat_lt1m") or {}
    parts.append('<div class="cards">'
                 + _card("RMSE", _num(g.get("rmse_m"), 3), "m")
                 + _card("MAE", _num(g.get("mae_m"), 3), "m")
                 + _card("Pearson r", _num(g.get("pearson_r"), 3))
                 + _card("δ₁ < 1.25", _num(g.get("delta1"), 3))
                 + _card("balanced RMSE", _num(res.get("balanced_rmse_m"), 3), "m",
                         warn=(res.get("balanced_rmse_m") or 0) >
                              1.3 * (g.get("rmse_m") or 1e9))
                 + "</div>")
    parts.append('<div class="cards">'
                 + _card("tall bias (GT ≥ 15 m)", _num(tall.get("bias_m"), 2), "m",
                         warn=abs(tall.get("bias_m") or 0) > 2.0)
                 + _card("flat bias (GT < 1 m)", _num(flat.get("bias_m"), 2), "m",
                         warn=abs(flat.get("bias_m") or 0) > 1.0)
                 + _card("landscape spread",
                         _num(res.get("landscape_rmse_spread_m"), 2), "m")
                 + _card("worst landscape", res.get("landscape_worst", "—"))
                 + "</div>")
    parts.append(
        '<div class="note">Read <b>balanced RMSE</b> and the two bias numbers, not '
        'the headline alone. A global RMSE is dominated by flat ground — roughly '
        'two thirds of an aerial scene — so a model that underestimates every tall '
        'building still posts a good one. A large negative <b>tall bias</b> means '
        'structures are being compressed; a large positive <b>flat bias</b> means '
        'terrain is being invented from image texture, which is the failure that '
        'turns a flythrough into a crumpled mountain range.</div>')

    # ---- landscape stability -----------------------------------------
    per_l = res.get("per_landscape") or {}
    if per_l:
        parts.append("<h2>Stability across landscapes</h2>")
        parts.append("<p>The rubric grades stability across urban, sparse, hilly and "
                     "forested scenes. No dataset here ships that label, so each "
                     "validation tile is classified from its own ground-truth height "
                     "field (relief, tall-pixel fraction, canopy roughness — see "
                     "<code>eval/landscape.py</code>). The rule and its descriptors are "
                     "in <code>metrics.json</code>; these are heuristics, not "
                     "annotations.</p>")
        parts.append(_table(
            ["landscape", "tiles", "pixels", "RMSE (m)", "MAE (m)", "bias (m)", "r"],
            [[k, v.get("tiles", "—"), f"{v.get('n', 0):,}", _num(v.get("rmse_m")),
              _num(v.get("mae_m")), _num(v.get("bias_m"), 2), _num(v.get("pearson_r"))]
             for k, v in sorted(per_l.items(), key=lambda kv: -kv[1].get("rmse_m", 0))]))
        parts.append(_fig(figs / "per_landscape.png",
                          "RMSE by landscape class; dashed line is the global RMSE."))

    # ---- height strata -------------------------------------------------
    per_s = {k: v for k, v in (res.get("per_stratum") or {}).items() if v.get("n")}
    if per_s:
        parts.append("<h2>Error by height stratum</h2>")
        parts.append(_table(
            ["stratum", "pixels", "RMSE (m)", "MAE (m)", "bias (m)"],
            [[k, f"{v['n']:,}", _num(v.get("rmse_m")), _num(v.get("mae_m")),
              _num(v.get("bias_m"), 2)] for k, v in per_s.items()]))
        parts.append(_fig(figs / "per_stratum.png",
                          "Per-stratum RMSE and signed bias. Negative bias in the top "
                          "stratum is tall-structure underestimation."))

    # ---- distributions --------------------------------------------------
    dist = "".join([
        _fig(figs / "scatter.png",
             "Predicted against reference height. Mass below y=x at the high end is "
             "compression of tall structures."),
        _fig(figs / "error_hist.png",
             "Signed error, split into flat ground and tall structures — the two "
             "regimes fail in opposite directions and a single histogram hides it."),
    ])
    if dist:
        parts.append("<h2>Error distribution</h2>" + dist)

    # ---- qualitative -----------------------------------------------------
    qual = "".join([
        _fig(figs / "hillshade.png",
             "Hillshade of the predicted and reference surfaces. This is the shape "
             "the 3D flythrough renders; flat ground must read as flat."),
        _fig(figs / "qualitative.png", "RGB, prediction, reference, absolute error."),
    ])
    if qual:
        parts.append("<h2>Qualitative</h2>" + qual)
    parts.append(_gallery(figs))

    # ---- training --------------------------------------------------------
    parts.append("<h2>Training</h2>")
    parts.append(_fig(figs / "curves.png",
                      "Loss and validation error. The dashed line is where the "
                      "encoder was unfrozen."))
    parts.append(_table(
        ["setting", "value"],
        [[k, html.escape(str(cfg.get(k, "—")))] for k in (
            "encoder_model_id", "datasets", "epochs", "freeze_epochs", "batch_size",
            "grad_accum", "learning_rate", "encoder_lr", "llrd", "tile_size",
            "canonical_gsd_m", "stratum_balance_beta", "w_consistency", "ema_decay")]))

    # ---- the inference contract -------------------------------------------
    if pre:
        parts.append("<h2>Inference contract</h2>")
        parts.append("<p>Serialised into every checkpoint, and read from the "
                     "checkpoint (never from a config file) by "
                     "<code>infer/predict.py</code>. Training and inference build "
                     "their tensors with the same code, so a validation number "
                     "measures the path the demo actually runs.</p>")
        parts.append(_table(["field", "value"],
                            [[k, html.escape(str(v))] for k, v in pre.items()]))

    # ---- per class, with the caveat attached ------------------------------
    per_c = res.get("per_class") or {}
    if per_c:
        stats = res.get("class_stats") or {}
        parts.append("<h2>Per semantic class</h2>")
        parts.append('<div class="note">Class <i>ids</i> are reported, not names. The '
                     'name order asserted by v1/v2 is contradicted by the measured '
                     'histogram (one supposed "bridge" id is 16 % of all pixels; '
                     '"ground" is 0.1 %), so the mapping is unverified. Do not attach '
                     'land-cover names to these rows until it is pinned to the GAMUS '
                     'paper.</div>')
        parts.append(_table(
            ["class id", "pixel share", "mean GT height (m)", "RMSE (m)", "MAE (m)"],
            [[k, f"{(stats.get(k, {}).get('px_frac') or 0) * 100:.1f}%",
              _num((stats.get(k, {}) or {}).get("mean_gt_height_m"), 2),
              _num(v.get("rmse_m")), _num(v.get("mae_m"))]
             for k, v in per_c.items()]))

    # ---- limits ------------------------------------------------------------
    parts.append("<h2>What this report does not establish</h2>")
    parts.append("<ul>"
                 "<li>Every labelled training source is foreign (US, German, "
                 "synthetic). No open dataset pairs Indian RGB with per-pixel "
                 "heights; the Indian branch here is unlabeled mean-teacher "
                 "adaptation, which reduces the domain gap but is not evidence of "
                 "Indian accuracy. Numbers on ISRO imagery have to be measured on "
                 "ISRO imagery.</li>"
                 "<li>Landscape labels are heuristics derived from GT geometry, not "
                 "annotations.</li>"
                 "<li>Semantic class ids are unverified (above).</li>"
                 "<li>A GeoNRW-derived nDSM is a morphological proxy, not a survey "
                 "product; treat any GeoNRW row as auxiliary.</li>"
                 "</ul>")

    parts.append(f'<footer>DepthWizard v4 · SIH 2026 PS 26175 · run directory '
                 f'<code>{html.escape(str(run_dir))}</code></footer>')

    doc = (f"<!doctype html><html lang=en><head><meta charset=utf-8>"
           f"<meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>{html.escape(title)} — validation report</title>"
           f"<style>{_CSS}</style></head><body><div class=wrap>"
           + "".join(parts) + "</div></body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    print(f"[report] {out_path}  ({out_path.stat().st_size / 1024:.0f} KB, self-contained)")
    return out_path


if __name__ == "__main__":
    import sys

    build(sys.argv[1] if len(sys.argv) > 1 else "outputs/v4")
