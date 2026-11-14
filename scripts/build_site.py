"""Build the static GitHub Pages site in _site/ from web/ plus the simulation package.

The browser runs exactly the files in microseismic/ (copied unchanged).
Usage:  python scripts/build_site.py   then   python -m http.server 8000 --directory _site
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_site"
if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(ROOT / "web", OUT)
(OUT / "microseismic").mkdir()
for source in sorted((ROOT / "microseismic").glob("*.py")):
    shutil.copy2(source, OUT / "microseismic" / source.name)
(OUT / ".nojekyll").touch()
print(f"Built {OUT}")
