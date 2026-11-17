"""docs/ is what GitHub Pages serves; it must be the current build of web/."""
import pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_site import outputs

def test_docs_is_the_current_build_of_web():
    for name, text in outputs().items():
        path = ROOT / "docs" / name
        assert path.exists(), f"docs/{name} is missing: run python scripts/build_site.py"
        assert path.read_text() == text, f"docs/{name} is stale: run python scripts/build_site.py"

def test_the_page_is_a_complete_document():
    html = outputs()["index.html"]
    assert html.startswith("<!doctype html>") and "<title>a2i Playground</title>" in html.split("</head>")[0]
    assert html.count("<body>") == 1 and html.rstrip().endswith("</html>")
