"""Build docs/index.html (the page GitHub Pages serves) from web/.

web/index.html is page content without an <html> wrapper, the form the Claude artifact viewer
wants, and it loads the codec from two separate files. This script makes one complete,
self-contained web page instead: title, fonts and styles go into a <head>, and the codec
(web/a2i.js plus the worker's message handler) is embedded in the page, which starts its
background worker from that text. The result needs no other file, so it works from GitHub
Pages and also when the file is simply opened from a disk.

usage:  python scripts/build_site.py          (tests/test_site.py fails when docs/ is stale)
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESET = (":root{color-scheme:light;padding:env(safe-area-inset-top,0px) 0 env(safe-area-inset-bottom,0px)}"
         "body{margin:0;font:14px system-ui}img{max-width:100%}[hidden]{display:none!important}")

def codec_source():
    web = ROOT / "web"
    worker = (web / "worker.js").read_text()
    assert 'importScripts("a2i.js");' in worker
    src = (web / "a2i.js").read_text() + "\n" + worker.replace('importScripts("a2i.js");', "")
    assert "</script" not in src.lower()                     # it is embedded in a <script> element
    return src

def build_index(content):
    head, body = content.split('<div class="wrap">', 1)
    body, script = body.split("<script>", 1)
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<meta name="description" content="Turn eight seconds of sound into a picture and read a picture or a '
            'screenshot of it back into sound, entirely on your device.">\n'
            f"<style>{RESET}</style>\n{head.strip()}\n</head>\n<body>\n<div class=\"wrap\">{body.rstrip()}\n\n"
            f'<script type="text/plain" id="codec-src">\n{codec_source()}</script>\n\n<script>{script.rstrip()}\n</body>\n</html>\n')

def outputs():
    """{file name in docs/: text}"""
    return {"index.html": build_index((ROOT / "web" / "index.html").read_text()), ".nojekyll": ""}

if __name__ == "__main__":
    docs = ROOT / "docs"; docs.mkdir(exist_ok=True)
    for name, text in outputs().items():
        (docs / name).write_text(text)
    for old in ("a2i.js", "worker.js"):                      # earlier builds used separate files
        (docs / old).unlink(missing_ok=True)
    print("wrote", ", ".join(sorted(outputs())), "to", docs)
