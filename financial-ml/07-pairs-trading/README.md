# Financial ML 7 — Pairs Trading and Statistical Arbitrage

When two related prices drift apart, bet that they come back together. The mathematics of mean reversion (AR(1), Ornstein–Uhlenbeck, half-lives), unit-root and cointegration tests written from scratch, the spurious-regression trap, a Kalman-filter hedge ratio, and an honest test of how long relative-value edges survive.

**Time:** About 4 hours with the exercises. **Before this:** Chapters 1–5.

| File | What it is |
|---|---|
| `index.html` | The guide: theory and maths from scratch, all the code with its real outputs, all the figures. |
| `pairs_trading.ipynb` | The complete notebook; it reproduces every number and figure in the guide. |
| `animations/pairs_lab.html` | Interactive playground: Pairs Lab. |
| `datasets/pairs_universe.csv` | Cached data (1.0 MB), so everything runs offline. |

## What you will do

1. Look at the big picture
2. Setup and data
3. The mathematics of mean reversion
4. Finding pairs, and the multiple-testing problem
5. The trading rule
6. A hedge ratio that adapts: the Kalman filter
7. Present your solution
8. Launch, monitor and maintain

## Results

- **What worked.** The mathematics: from-scratch ADF and Engle–Granger tests that match `statsmodels`, half-lives, and a Kalman filter for drifting hedge ratios.
- **What the data say.** A screen of 253 pairs finds only a few more “cointegrated” pairs than chance predicts, and none survives a Bonferroni correction; most relationships found in 2010–2017 did not survive to 2018–2026, and the selected pairs' out-of-sample Sharpe ratio (0.21) is far below the in-sample one (0.40). Classic pairs trading of large stocks has decayed since the 1990s (Gatev, Goetzmann and Rouwenhorst, 2006, documented the decline), as many funds compete for the same gaps.
- **How professionals do it now.** Many assets at once (baskets and residuals of factor models, which is Chapter 6's cross-section combined with this chapter's mean reversion), short holding periods with careful cost control, and economic reasons for every relationship. The stop-loss and the decision to retire a pair are as important as the entry rule.

## Run it

```bash
python3 -m pip install -U jupyter matplotlib numpy pandas scipy statsmodels
jupyter notebook pairs_trading.ipynb
```

Or open `index.html` in a browser to read the guide with every output already computed.

Previous: [Chapter 6 · Cross-Sectional Factors and ML Ranking](../06-cross-sectional-ml/) · Next: [Chapter 8 · Options, Volatility and Neural Pricers](../08-options-and-neural-pricers/)
