# blog/

The public write-up of DepthWizard's development, kept as a living document.

| File | What it is |
|---|---|
| `depthwizard-journey.md` | **Canonical draft.** Edit this one. Ends with a `SOURCES` comment mapping every number back to the repo. |
| `build_medium.sh` | Strips the SOURCES block and renders `medium-paste.html` via pandoc. |
| `medium-paste.html` | Generated. Open in a browser, select all, copy, paste into a Medium draft — headings, bold and lists survive the paste. |
| `CHANGELOG.md` | One entry per update to the published post. |
| `assets/` | Screenshots and figures. |

## Updating after a new version lands

1. Edit `depthwizard-journey.md` — add to the **Status** section, adjust the v3/v4 sections if
   the architecture changed, and add real numbers only once
   `CompetitionContext/Outcome.md` has a non-pending row for that run.
2. Add the source line for any new number to the `SOURCES` comment at the bottom.
3. `./build_medium.sh`
4. Paste into the existing Medium story and update it — Medium keeps the same URL when you
   edit a published post, so the link you shared stays valid.
5. Log it in `CHANGELOG.md`.

**Images do not survive the paste.** The `<img>` tags point at local relative paths, so Medium
drops them. After pasting, drag the two files from `assets/` into the Medium editor at the
marked spots — the captions come through as italic text and Medium turns them into real
captions if you select them and press the caption style.

## The one rule

Do not state an accuracy number for a version that has not been trained. v3 and v4 are built
and tested, not trained. `FineTunning/v4/README.md` §7 is the list of things the project
cannot currently claim; the post must not contradict it.
