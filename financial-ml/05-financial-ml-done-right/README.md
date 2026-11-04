# Financial ML 5 — Financial Machine Learning Done Right

Why the standard machine-learning recipe fails on market data, and the fixes: triple-barrier labels, sample uniqueness, purged and embargoed cross-validation, meta-labelling, honest feature importance, and the probability that your best backtest is overfit.

**Time:** About 5 hours with the exercises. **Before this:** Chapters 1–4; scikit-learn basics.

| File | What it is |
|---|---|
| `index.html` | The guide: theory and maths from scratch, all the code with its real outputs, all the figures. |
| `financial_ml_done_right.ipynb` | The complete notebook; it reproduces every number and figure in the guide. |
| `animations/cv_lab.html` | Interactive playground: CV Lab. |
| `datasets/spy_ohlc.csv` | Cached data (453 kB), so everything runs offline. |
| `datasets/vix.csv` | Cached data (148 kB), so everything runs offline. |

## What you will do

1. Setup and data
2. Labels: what exactly are we predicting?
3. Overlapping labels and sample uniqueness
4. Cross-validation that does not leak
5. Project: predicting SPY's direction honestly
6. Meta-labelling: let ML size the bet, not choose the side
7. Feature importance, honestly
8. How likely is your best backtest to be overfit?
9. The checklist

## Run it

```bash
python3 -m pip install -U jupyter matplotlib numpy pandas scikit-learn scipy
jupyter notebook financial_ml_done_right.ipynb
```

Or open `index.html` in a browser to read the guide with every output already computed.

Previous: [Chapter 4 · Portfolio Construction out of Sample](../04-portfolio-construction/) · Next: [Chapter 6 · Cross-Sectional Factors and ML Ranking](../06-cross-sectional-ml/)
