#!/usr/bin/env python3
"""Write a README.md into every built chapter folder (toolkit/ and financial-ml/).

The README is generated from the chapter's own source so it never drifts:
front matter (title, subtitle, time, prerequisites), the list of files, the
sections, the "Present your solution" results with their real numbers (taken
from the executed notebook), and how to run the notebook.

    python3 tools/readmes.py
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = {"Exercises", "What's next", "Try it out"}
STD = {"numpy", "pandas", "matplotlib", "scipy", "statsmodels", "sklearn", "arch", "yfinance", "plotly", "seaborn", "lightgbm", "imageio"}
PIP = {"sklearn": "scikit-learn"}


def front_matter(text):
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    meta = {}
    for line in m.group(1).splitlines():
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip()
    return meta, text[m.end():]


def results_dict(chap):
    path = os.path.join(chap, ".executed.ipynb")
    if not os.path.exists(path):
        return {}
    for cell in json.load(open(path, encoding="utf-8"))["cells"]:
        for o in cell.get("outputs", []):
            t = o.get("text", "")
            t = "".join(t) if isinstance(t, list) else t
            if "__R__" in t:
                return json.loads(t.split("__R__", 1)[1])
    return {}


def section(body, title):
    m = re.search(r"^## " + re.escape(title) + r"\s*\n(.*?)(?=^## |\Z)", body, re.S | re.M)
    return m.group(1).strip() if m else ""


def html_title(path):
    m = re.search(r"<title>(.*?)</title>", open(path, encoding="utf-8").read(), re.S)
    return m.group(1).strip() if m else os.path.basename(path)


def size(path):
    n = os.path.getsize(path)
    return f"{n / 1e6:.1f} MB" if n > 1e6 else f"{n / 1e3:.0f} kB"


def build(chap):
    text = open(os.path.join(chap, "chapter.md"), encoding="utf-8").read()
    meta, body = front_matter(text)
    R = results_dict(chap)
    fill = lambda s: re.sub(r"\{\{(\w+)\}\}", lambda m: str(R.get(m.group(1), m.group(0))), s)

    code = "\n".join(re.findall(r"```python[^\n]*\n(.*?)```", body, re.S))
    pkgs = sorted({PIP.get(p, p) for p in re.findall(r"^\s*(?:from|import)\s+(\w+)", code, re.M) if p in STD})

    rows = [("`index.html`", "The guide: theory and maths from scratch, all the code with its real outputs, all the figures."),
            (f"`{meta['notebook']}`", "The complete notebook; it reproduces every number and figure in the guide.")]
    for a in sorted(glob.glob(os.path.join(chap, "animations", "*.html"))):
        rows.append((f"`animations/{os.path.basename(a)}`", f"Interactive playground: {html_title(a)}."))
    for d in sorted(glob.glob(os.path.join(chap, "datasets", "*"))):
        rows.append((f"`datasets/{os.path.basename(d)}`", f"Cached data ({size(d)}), so everything runs offline."))

    heads = [h for h in re.findall(r"^## (.+)$", body, re.M) if h not in SKIP]
    present = fill(section(body, "Present your solution"))
    present = "\n".join(l for l in present.splitlines() if not l.startswith((":::", "```")))

    out = [f"# {meta.get('label', '').split('·')[0].strip()} {meta.get('number', '')} — {meta['title']}".replace("  ", " "), "",
           meta.get("subtitle", ""), "",
           f"**Time:** {meta.get('time', '')}. **Before this:** {meta.get('needs', 'nothing')}.", "",
           "| File | What it is |", "|---|---|"] + [f"| {a} | {b} |" for a, b in rows] + ["", "## What you will do", ""]
    out += [f"{i}. {h}" for i, h in enumerate(heads, 1)]
    if present:
        out += ["", "## Results", "", present]
    out += ["", "## Run it", "", "```bash", f"python3 -m pip install -U jupyter {' '.join(pkgs)}", f"jupyter notebook {meta['notebook']}", "```", "",
            "Or open `index.html` in a browser to read the guide with every output already computed.", ""]
    nav = []
    if meta.get("prev"):
        nav.append(f"Previous: [{meta.get('prev_title', 'previous chapter')}]({os.path.dirname(meta['prev'])}/)")
    if meta.get("next"):
        nav.append(f"Next: [{meta.get('next_title', 'next chapter')}]({os.path.dirname(meta['next'])}/)")
    if nav:
        out += [" · ".join(nav), ""]
    open(os.path.join(chap, "README.md"), "w", encoding="utf-8").write("\n".join(out))
    return chap


if __name__ == "__main__":
    for c in sorted(glob.glob(os.path.join(ROOT, "toolkit", "*")) + glob.glob(os.path.join(ROOT, "financial-ml", "*"))):
        if os.path.exists(os.path.join(c, "chapter.md")) and os.path.exists(os.path.join(c, "index.html")):
            print("README:", os.path.relpath(build(c), ROOT))
