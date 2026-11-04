#!/usr/bin/env python3
"""Regenerate the hub page (index.html) and README.md from tools/course.json."""
import html
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
course = json.load(open(os.path.join(ROOT, "tools", "course.json"), encoding="utf-8"))

cards = []
for tr in course["tracks"]:
    items = []
    for c in tr["chapters"]:
        kind = c.get("kind", "")
        tag = (f'<span class="kind">{html.escape(kind)}</span>' if kind else "")
        inner = (f'<span class="num">{html.escape(c["n"])}</span>{tag}'
                 f'<h3>{html.escape(c["title"])}</h3><p>{html.escape(c["desc"])}</p>')
        if c["ready"]:
            items.append(f'<li><a class="card" href="{c["path"]}">{inner}<span class="go">Open →</span></a></li>')
        else:
            items.append(f'<li><div class="card soon">{inner}<span class="go">Coming next</span></div></li>')
    cards.append(f'<section class="track"><h2>{html.escape(tr["name"])}</h2><p class="blurb">{html.escape(tr["blurb"])}</p>'
                 f'<ul class="cards">{"".join(items)}</ul></section>')

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quant from Zero</title>
<meta name="description" content="From zero finance knowledge to financial machine learning and algorithmic trading: interactive lessons, a Python data toolkit and end-to-end projects on real market data.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/guide.css">
<script>try{{var t=localStorage.getItem('qfz-theme');if(t)document.documentElement.dataset.theme=t}}catch(e){{}}</script>
<style>
.hub{{max-width:1080px;margin:0 auto;padding:0 16px 80px}}
.hero{{padding:56px 0 18px}}.hero h1{{font-size:clamp(2.2rem,6vw,3.4rem);margin:.2rem 0}}
.hero p{{font-size:1.2rem;color:var(--muted);max-width:62ch}}
.path{{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 6px;font:600 .85rem var(--sans)}}
.path span{{background:var(--accent-soft);color:var(--accent);border-radius:999px;padding:5px 12px}}
.track h2{{border-top:1px solid var(--line)}}.blurb{{color:var(--muted)}}
.cards{{list-style:none;padding:0;margin:1rem 0;display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}}
.card{{display:flex;flex-direction:column;height:100%;border:1px solid var(--line);border-radius:14px;padding:16px 18px;background:var(--surface);color:var(--ink);text-decoration:none;font-family:var(--sans);transition:border-color .2s,transform .2s}}
a.card:hover{{border-color:var(--accent);transform:translateY(-2px)}}
.card h3{{margin:.4rem 0 .3rem;font-size:1.1rem}}.card p{{margin:0 0 .8rem;font-size:.92rem;color:var(--muted);line-height:1.5;flex:1}}
.num{{font:700 .75rem var(--mono);color:var(--accent)}}.kind{{margin-left:8px;font:600 .72rem var(--sans);text-transform:uppercase;letter-spacing:.06em;color:var(--accent2)}}
.go{{font:600 .85rem var(--sans);color:var(--accent)}}.soon{{opacity:.6}}.soon .go{{color:var(--muted)}}
.how{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px;margin:1rem 0}}
.how div{{border:1px solid var(--line);border-radius:12px;padding:12px 14px;background:var(--surface);font:.92rem/1.5 var(--sans)}}
.how b{{display:block;margin-bottom:2px}}
</style></head><body>
<header class="topbar"><a class="brand" href="index.html"><span class="logo" aria-hidden="true"></span>Quant from Zero</a><span class="crumb">Home</span><button class="themebtn" type="button" aria-label="Switch light or dark theme">Theme</button></header>
<main class="hub">
<section class="hero"><p class="eyebrow">A course from zero to financial ML</p><h1>Quant from Zero</h1>
<p>Start knowing nothing about money. Finish able to research, test and run trading strategies with machine learning, honestly. Three parts, taken in order.</p>
<div class="path"><span>0 · Finance from zero</span><span>1 · Python data toolkit</span><span>2 · Financial ML and algorithmic trading</span></div></section>
<section><h2>How every chapter works</h2><div class="how">
<div><b>A guide you read</b>Plain words first, then the maths from scratch, with every step of code and its real output.</div>
<div><b>A notebook you run</b>The same code as a Jupyter notebook that reproduces every number and every figure.</div>
<div><b>Real data</b>Daily prices, rates and factor returns from public sources, saved in the chapter so it also runs offline.</div>
<div><b>A playground</b>An interactive page where you move the sliders and watch the ideas work.</div></div></section>
{"".join(cards)}
<section><h2>Running the notebooks</h2>
<pre class="code"><code>python3 -m pip install -U jupyter numpy pandas matplotlib seaborn plotly scipy scikit-learn statsmodels lightgbm yfinance
jupyter notebook</code></pre>
<p>Each chapter folder holds <code>index.html</code> (the guide), a notebook, <code>images/</code>, <code>datasets/</code> and <code>animations/</code>. The guides and notebooks are generated from one source, <code>chapter.md</code>, by <code>tools/build.py</code>.</p>
<p class="meta">Educational material, not investment advice. Backtests are not predictions; markets change.</p></section>
</main>
<script src="assets/guide.js"></script>
</body></html>
"""
open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8").write(page)

# README
lines = ["# Quant from Zero", "",
         "From **zero finance knowledge** to **financial machine learning and algorithmic trading**: interactive lessons, "
         "a Python data toolkit, and end-to-end projects on real market data.", "",
         "**Start here:** open [`index.html`](index.html) in a browser (or enable GitHub Pages for this repository).", ""]
for tr in course["tracks"]:
    lines += [f"## {tr['name']}", "", tr["blurb"], "", "| # | Chapter | What you learn |", "|---|---|---|"]
    for c in tr["chapters"]:
        t = f"[{c['title']}]({c['path']})" if c["ready"] else f"{c['title']} *(coming)*"
        lines.append(f"| {c['n']} | {t} | {c['desc']} |")
    lines.append("")
lines += ["## Running the notebooks", "", "```bash",
          "python3 -m pip install -U jupyter numpy pandas matplotlib seaborn plotly scipy scikit-learn statsmodels lightgbm yfinance",
          "jupyter notebook", "```", "",
          "Each chapter folder contains `index.html` (the guide), a Jupyter notebook that reproduces every number and figure, "
          "`images/`, `datasets/`, `animations/` (interactive playgrounds) and a `README.md` with its files and results. "
          "Guides and notebooks are generated from a single literate source, `chapter.md`, with "
          "`python3 tools/build.py <chapter-folder>`; the hub and this README with `python3 tools/hub.py`; the chapter "
          "READMEs with `python3 tools/readmes.py`.", "",
          "## How the projects are structured", "",
          "The project chapters follow the end-to-end workflow of Chapter 2 of *Hands-On Machine Learning* (as in the "
          "companion Romy and surrogate-modelling courses), adapted to trading: **look at the big picture** (objective, "
          "performance measure, assumptions), **get the data**, **explore** it, **build** and validate models without "
          "look-ahead, **fine-tune honestly** (walk-forward, purged cross-validation, multiple-testing checks), **present "
          "your solution** with its limitations, and **launch, monitor and maintain** it. Each chapter ends with an "
          "interactive playground and exercises with solutions.", "",
          "*Educational material, not investment advice.*", ""]
open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8").write("\n".join(lines))
print("hub and README written")
