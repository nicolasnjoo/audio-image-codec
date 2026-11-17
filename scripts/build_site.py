"""Build docs/ (the folder GitHub Pages serves) from web/.

web/index.html is page content without an <html> wrapper, the form the Claude artifact viewer
wants. A normal web server needs a complete document, so this script moves the title, fonts
and styles into a <head> and copies the two scripts next to it.

usage:  python scripts/build_site.py          (tests/test_site.py fails when docs/ is stale)
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESET = (":root{color-scheme:light;padding:env(safe-area-inset-top,0px) 0 env(safe-area-inset-bottom,0px)}"
         "body{margin:0;font:14px system-ui}img{max-width:100%}[hidden]{display:none!important}")

def build_index(content):
    head, body = content.split('<div class="wrap">', 1)
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<meta name="description" content="Turn eight seconds of sound into a picture and read a picture or a '
            'screenshot of it back into sound, entirely on your device.">\n'
            f"<style>{RESET}</style>\n{head.strip()}\n</head>\n<body>\n<div class=\"wrap\">{body.rstrip()}\n</body>\n</html>\n")

def outputs():
    """{file name in docs/: text}"""
    web = ROOT / "web"
    return {"index.html": build_index((web / "index.html").read_text()),
            "a2i.js": (web / "a2i.js").read_text(), "worker.js": (web / "worker.js").read_text(),
            ".nojekyll": ""}

if __name__ == "__main__":
    docs = ROOT / "docs"; docs.mkdir(exist_ok=True)
    for name, text in outputs().items():
        (docs / name).write_text(text)
    print("wrote", ", ".join(sorted(outputs())), "to", docs)
