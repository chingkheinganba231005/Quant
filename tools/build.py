#!/usr/bin/env python3
"""Build a chapter: literate Markdown source -> executed Jupyter notebook + HTML guide.

Usage:  python3 tools/build.py toolkit/01-numpy-from-zero            # build one chapter
        python3 tools/build.py toolkit/01-numpy-from-zero --no-exec  # reuse the last executed notebook

The chapter folder contains `chapter.md`. Its format:

  ---                                   front matter (simple key: value lines)
  title: NumPy from Zero
  label: Toolkit · Chapter 1
  notebook: numpy_from_zero.ipynb
  ---
  Markdown text with $inline$ and $$display$$ maths.

  ```python                 code that is run, shown in the guide with its output
  ```python hide            run but not shown in the guide (still in the notebook)
  ```python noout           run and shown, output not shown in the guide
  ```python norun           shown only (not executed, not in the notebook as code)

  :::eli5 / :::idea / :::quant / :::trap / :::tip / :::note / :::exercise   ... :::
  ![Figure 1-2. Caption text](images/name.png)       becomes a numbered <figure>
  ::playground animations/file.html 620              embeds an interactive page
  {{key}}                    replaced by R["key"] (a dict filled in by the code cells)

Everything a reader sees is produced from the same cells that the notebook runs,
so every number and every figure in the guide is reproducible.
"""
import html
import json
import os
import re
import sys

import markdown
import nbformat
from nbclient import NotebookClient
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOX_TITLES = {
    "eli5": "Like you’re five", "idea": "The real idea", "quant": "Quant view",
    "trap": "Trap", "tip": "Tip", "note": "Note", "exercise": "Try it",
    "math": "The maths, from scratch", "check": "Check yourself",
}


def parse_source(text):
    meta = {}
    if text.startswith("---"):
        end = text.index("\n---", 3)
        for line in text[3:end].strip().splitlines():
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
        text = text[end + 4:]
    cells, buf, i = [], [], 0
    lines = text.split("\n")
    while i < len(lines):
        m = re.match(r"^```python\s*(\w*)\s*$", lines[i])
        if m:
            if buf:
                cells.append({"type": "md", "src": "\n".join(buf).strip("\n")})
                buf = []
            j = i + 1
            code = []
            while not lines[j].startswith("```"):
                code.append(lines[j])
                j += 1
            cells.append({"type": "code", "src": "\n".join(code), "opt": m.group(1) or "show"})
            i = j + 1
            continue
        buf.append(lines[i])
        i += 1
    if buf:
        cells.append({"type": "md", "src": "\n".join(buf).strip("\n")})
    return meta, [c for c in cells if c["src"].strip() or c["type"] == "code"]


def fill(text, R):
    if isinstance(text, list):
        text = "".join(text)
    def rep(m):
        k = m.group(1)
        if k not in R:
            raise KeyError(f"placeholder {{{{{k}}}}} has no value in R")
        return str(R[k])
    return re.sub(r"\{\{(\w+)\}\}", rep, text)


def build_notebook(meta, cells, chap_dir, execute=True):
    nb_path = os.path.join(chap_dir, meta["notebook"])
    cache = os.path.join(chap_dir, ".executed.ipynb")
    if execute:
        nb = nbformat.v4.new_notebook()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.metadata["language_info"] = {"name": "python"}
        for c in cells:
            if c["type"] == "md":
                nb.cells.append(nbformat.v4.new_markdown_cell(c["src"]))
            elif c["opt"] == "norun":
                nb.cells.append(nbformat.v4.new_markdown_cell("```python\n" + c["src"] + "\n```"))
            else:
                nb.cells.append(nbformat.v4.new_code_cell(c["src"]))
        # a last helper cell exports the placeholder values R
        nb.cells.append(nbformat.v4.new_code_cell(
            "import json as _json\nprint('__R__' + _json.dumps({k: str(v) for k, v in globals().get('R', {}).items()}))"))
        os.makedirs(os.path.join(chap_dir, "images"), exist_ok=True)
        NotebookClient(nb, timeout=3600, kernel_name="python3",
                       resources={"metadata": {"path": chap_dir}}).execute()
        nbformat.write(nb, cache)
    else:
        nb = nbformat.read(cache, as_version=4)
        n_code = sum(1 for c in cells if c["type"] == "code" and c["opt"] != "norun")
        if n_code + 1 != sum(1 for c in nb.cells if c.cell_type == "code"):
            sys.exit("code cells changed since the last run: rebuild without --no-exec")
        # refresh markdown from the source (text edits need no re-run)
        md_src = [c["src"] if c["type"] == "md" else "```python\n" + c["src"] + "\n```"
                  for c in cells if c["type"] == "md" or c["opt"] == "norun"]
        mds = [c for c in nb.cells if c.cell_type == "markdown"]
        for c, s in zip(mds, md_src):
            c.source = s
    R = {}
    for o in nb.cells[-1].outputs:
        if o.get("name") == "stdout" and "__R__" in o.get("text", ""):
            R = json.loads(o["text"].split("__R__", 1)[1])
    saved = nbformat.from_dict(json.loads(nbformat.writes(nb)))
    saved.cells = saved.cells[:-1]
    for c in saved.cells:
        if c.cell_type == "markdown":
            c.source = fill(c.source, R)
    nbformat.write(saved, nb_path)
    return nb, R


# ---------- HTML ----------
FMT = HtmlFormatter(nowrap=True)


def render_code(src):
    return '<pre class="code"><code>' + highlight(src, PythonLexer(), FMT) + "</code></pre>"


def render_outputs(outputs, max_lines=40):
    parts = []
    for o in outputs:
        if o.output_type == "stream" and o.name == "stdout":
            parts.append(("text", o.text))
        elif o.output_type in ("execute_result", "display_data"):
            d = o.get("data", {})
            if "text/html" in d and "<table" in d["text/html"]:
                parts.append(("html", d["text/html"]))
            elif "text/plain" in d and "image/png" not in d:
                parts.append(("text", d["text/plain"]))
        elif o.output_type == "error":
            raise RuntimeError("cell raised an error: " + o.ename + ": " + o.evalue)
    out = []
    for kind, txt in parts:
        if kind == "html":
            txt = re.sub(r'<style[\s\S]*?</style>', '', txt)
            out.append('<div class="out df">' + txt + "</div>")
        else:
            lines = txt.rstrip("\n").split("\n")
            if len(lines) > max_lines:
                lines = lines[:max_lines] + ["…"]
            out.append('<pre class="out">' + html.escape("\n".join(lines)) + "</pre>")
    return "".join(out)


MATH_RE = re.compile(r"(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|(?<![\\$\w])\$(?!\s)[^$\n]+?(?<!\s)\$(?!\w))")


def md_to_html(src, fig_counter, chap_no):
    maths = []

    def keep(m):
        maths.append(m.group(0))
        return f"\u0000M{len(maths) - 1}\u0000"
    src = MATH_RE.sub(keep, src)

    # boxes
    def box(m):
        kind, title, body = m.group(1), (m.group(2) or "").strip(), m.group(3)
        t = title or BOX_TITLES.get(kind, kind.title())
        return f'\n<div class="box {kind}" markdown="1">\n<p class="lab">{html.escape(t)}</p>\n\n{body}\n</div>\n'
    src = re.sub(r"^:::(\w+)[ \t]*(.*)\n([\s\S]*?)^:::\s*$", box, src, flags=re.M)

    # playground
    def pg(m):
        path, h = m.group(1), m.group(2) or "640"
        return (f'\n<div class="playground"><div class="pg-bar"><span>Interactive playground</span>'
                f'<a href="{path}" target="_blank" rel="noopener">open full screen ↗</a></div>'
                f'<iframe src="{path}" loading="lazy" style="height:{h}px" title="Interactive playground"></iframe></div>\n')
    src = re.sub(r"^::playground\s+(\S+)\s*(\d+)?\s*$", pg, src, flags=re.M)

    # figures
    def figure(m):
        fig_counter[0] += 1
        cap, path = m.group(1), m.group(2)
        cap = re.sub(r"^Figure\s+[\d\-–.]+\s*", "", cap)
        n = f"{chap_no}-{fig_counter[0]}"
        return (f'\n<figure><img src="{path}" alt="{html.escape(cap)}" loading="lazy">'
                f'<figcaption><b>Figure {n}.</b> {cap}</figcaption></figure>\n')
    src = re.sub(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", figure, src, flags=re.M)

    out = markdown.markdown(src, extensions=["tables", "md_in_html", "attr_list", "sane_lists", "fenced_code"])

    def put(m):
        return html.escape(maths[int(m.group(1))], quote=False)
    out = re.sub("\u0000M(\\d+)\u0000", put, out)
    return out


def slug(s):
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"[^\w\s-]", "", s.lower()).strip()
    return re.sub(r"[\s_]+", "-", s)


TEMPLATE = open(os.path.join(ROOT, "tools", "template.html"), encoding="utf-8").read()


def build_html(meta, cells, nb, R, chap_dir):
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    ci = 0
    body = []
    fig_counter = [0]
    chap_no = meta.get("number", "1")
    for c in cells:
        if c["type"] == "md":
            body.append(md_to_html(fill(c["src"], R), fig_counter, chap_no))
        elif c["opt"] == "norun":
            body.append(render_code(c["src"]))
        else:
            cell = code_cells[ci]
            ci += 1
            if c["opt"] == "hide":
                continue
            body.append(render_code(c["src"]))
            if c["opt"] != "noout":
                body.append(render_outputs(cell.outputs))
    html_body = "\n".join(body)
    # ids for h2/h3 and a table of contents
    toc, seen = [], set()

    def hid(m):
        lvl, inner = m.group(1), m.group(2)
        s = slug(inner) or "s"
        while s in seen:
            s += "-x"
        seen.add(s)
        if lvl == "2":
            toc.append(f'<li><a href="#{s}">{re.sub("<[^>]+>", "", inner)}</a></li>')
        return f'<h{lvl} id="{s}">{inner}</h{lvl}>'
    html_body = re.sub(r"<h([23])>(.*?)</h\1>", hid, html_body)
    depth = os.path.relpath(ROOT, chap_dir).replace(os.sep, "/")
    page = TEMPLATE
    subs = {
        "TITLE": meta.get("title", ""), "LABEL": meta.get("label", ""),
        "SUBTITLE": meta.get("subtitle", ""), "DESCRIPTION": meta.get("description", meta.get("subtitle", "")),
        "ROOT": depth, "TOC": "\n".join(toc), "BODY": html_body,
        "NOTEBOOK": meta["notebook"], "PREV": meta.get("prev", ""), "NEXT": meta.get("next", ""),
        "PREV_T": meta.get("prev_title", ""), "NEXT_T": meta.get("next_title", ""),
        "META": " · ".join(x for x in [meta.get("time", ""), meta.get("needs", "")] if x),
    }
    for k, v in subs.items():
        page = page.replace("%%" + k + "%%", v)
    with open(os.path.join(chap_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)


def main():
    chap = os.path.abspath(sys.argv[1])
    execute = "--no-exec" not in sys.argv
    src = open(os.path.join(chap, "chapter.md"), encoding="utf-8").read()
    meta, cells = parse_source(src)
    nb, R = build_notebook(meta, cells, chap, execute)
    build_html(meta, cells, nb, R, chap)
    print("built", chap, "| placeholders:", len(R))


if __name__ == "__main__":
    main()
