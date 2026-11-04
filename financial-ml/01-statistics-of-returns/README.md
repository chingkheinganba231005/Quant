# Financial ML 1 — The Statistics of Financial Returns

What real returns look like, the maths to describe and test them (written from scratch), and the uncomfortable truth at the centre of quant finance: most of what looks like skill is luck.

**Time:** About 4 hours with the exercises. **Before this:** Toolkit 1–3; Finance From Zero lessons 8 and 15.

| File | What it is |
|---|---|
| `index.html` | The guide: theory and maths from scratch, all the code with its real outputs, all the figures. |
| `statistics_of_returns.ipynb` | The complete notebook; it reproduces every number and figure in the guide. |
| `animations/luck_lab.html` | Interactive playground: Luck Lab. |
| `datasets/assets_daily.csv` | Cached data (420 kB), so everything runs offline. |
| `datasets/ff_factors_daily.csv` | Cached data (1.3 MB), so everything runs offline. |
| `datasets/sp500_index_daily.csv` | Cached data (346 kB), so everything runs offline. |

## What you will do

1. Setup
2. Get the data
3. From prices to returns
4. Stylised fact 1: heavy tails
5. Stylised fact 2: no linear predictability, strong volatility predictability
6. Stylised fact 3: aggregational Gaussianity
7. The same facts in seven assets
8. Is the market a random walk? The variance-ratio test
9. Maximum likelihood: fitting a fat-tailed distribution
10. How much can a sample tell you?
11. The bootstrap, and why blocks matter
12. Luck versus skill: multiple testing
13. Summary: what every model must respect

## Run it

```bash
python3 -m pip install -U jupyter matplotlib numpy pandas scipy statsmodels
jupyter notebook statistics_of_returns.ipynb
```

Or open `index.html` in a browser to read the guide with every output already computed.

Previous: [Toolkit 3 · Plotting from Zero](../../toolkit/03-plotting-from-zero/) · Next: [Chapter 2 · A Backtester from Scratch](../02-backtesting-trend-following/)
