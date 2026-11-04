# Toolkit 3 — Plotting from Zero

From your first line chart to a one-page market report: Matplotlib’s model of figures and axes, every chart a quant needs, styling that tells the truth, seaborn for statistics, Plotly for interactive dashboards, and animations.

**Time:** About 2 to 3 hours with the exercises. **Before this:** Toolkit 1 and 2.

| File | What it is |
|---|---|
| `index.html` | The guide: theory and maths from scratch, all the code with its real outputs, all the figures. |
| `plotting_from_zero.ipynb` | The complete notebook; it reproduces every number and figure in the guide. |
| `animations/spy_candles.html` | Interactive playground: spy_candles.html. |
| `datasets/fred_rates.csv` | Cached data (108 kB), so everything runs offline. |
| `datasets/prices_long.csv` | Cached data (2.3 MB), so everything runs offline. |

## What you will do

1. Setup
2. The anatomy of a figure
3. Choosing the right chart
4. Several panels: `subplots`, shared axes and `GridSpec`
5. Annotations, twin axes and reference lines
6. Heatmaps: a correlation matrix
7. Style: make the data the loudest thing
8. Saving figures
9. pandas plotting: the quick look
10. seaborn: statistical charts in one line
11. Plotly: interactive charts for the browser
12. Animations
13. Putting it together: a one-page market report

## Run it

```bash
python3 -m pip install -U jupyter matplotlib numpy pandas plotly seaborn
jupyter notebook plotting_from_zero.ipynb
```

Or open `index.html` in a browser to read the guide with every output already computed.

Previous: [Toolkit 2 · pandas from Zero](../02-pandas-from-zero/) · Next: [Financial ML 1 · The Statistics of Returns](../../financial-ml/01-statistics-of-returns/)
