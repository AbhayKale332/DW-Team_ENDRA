#!/usr/bin/env bash
# Render the canonical draft to HTML for pasting into the Medium editor.
# Strips the trailing SOURCES comment block so it cannot leak into the post.
# No pandoc --standalone: it injects a duplicate <h1> from the metadata title,
# which Medium would then treat as a second heading above the real one.
set -euo pipefail
cd "$(dirname "$0")"
python3 - <<'PY'
import re, pathlib
src = pathlib.Path("depthwizard-journey.md").read_text()
pathlib.Path(".medium.tmp.md").write_text(re.sub(r"\n<!--\nSOURCES.*?-->\n", "\n", src, flags=re.S))
PY
{
  printf '%s\n' '<meta charset="utf-8">' \
    '<style>body{max-width:42em;margin:3em auto;padding:0 1.5em;font:17px/1.6 Georgia,serif;color:#222}' \
    'img{max-width:100%}h1{line-height:1.2}blockquote{border-left:3px solid #ddd;margin:0;padding-left:1em;color:#555}' \
    'em{color:#666}code{font:0.9em ui-monospace,monospace;background:#f4f4f4;padding:1px 4px}</style>'
  pandoc -f gfm -t html5 .medium.tmp.md
} > medium-paste.html
rm -f .medium.tmp.md
echo "wrote blog/medium-paste.html — open it, select all, copy, paste into a Medium draft"
