# Financial ML 4 — Portfolio Construction out of Sample

Markowitz’s beautiful theory meets noisy data. Derive the efficient frontier from scratch, see how estimation error turns optimisers into error maximisers, and fix it with shrinkage, risk parity and hierarchical risk parity, all judged in a 50-year walk-forward test on 30 industries.

**Time:** About 5 hours with the exercises. **Before this:** Chapters 1–3; Finance From Zero lessons 9 and 10.

| File | What it is |
|---|---|
| `index.html` | The guide: theory and maths from scratch, all the code with its real outputs, all the figures. |
| `portfolio_construction.ipynb` | The complete notebook; it reproduces every number and figure in the guide. |
| `animations/frontier_lab.html` | Interactive playground: Frontier Lab. |
| `datasets/ff_factors_daily.csv` | Cached data (653 kB), so everything runs offline. |
| `datasets/industry30_daily_vw.csv` | Cached data (3.6 MB), so everything runs offline. |

## What you will do

1. Look at the big picture
2. Get the data
3. The theory: the efficient frontier, from scratch
4. Estimation error: why optimisers maximise error
5. Better estimates: shrinkage
6. Portfolios that need fewer estimates
7. The walk-forward test
8. Present your solution
9. Launch, monitor and maintain

## Results

| Method | Inputs it trusts | Strength | Weakness |
|---|---|---|---|
| 1/N | none | robust, diversified, cheap to run | ignores risk differences |
| Minimum variance (shrunk) | covariance | lowest risk; low-vol anomaly | concentrated; tracking error to the market |
| Mean–variance (sample) | means and covariance | optimal *if* inputs were known | error maximiser; high turnover |
| Risk parity | covariance | balanced risk; no means | needs leverage to reach equity-like returns |
| HRP | covariance (via a tree) | no inversion; stable; diversified | heuristic; depends on clustering choices |

**Recommendation.** Over fifty years, the method that ignored expected returns completely and minimised risk (Sharpe 0.61) beat both 1/N (0.55) and the mean–variance optimiser (0.56), with a third less volatility and much smaller drawdowns in long bear markets. Its weakness is concentration (about four effective holdings), so in practice: use minimum variance with a shrunk covariance and weight caps if the goal is low risk, or HRP (Sharpe 0.59) if broad diversification matters; never feed raw sample means into an optimiser; and treat the predicted risk of optimised portfolios as optimistic. Expected returns should come from a separately validated model (Chapters 5 and 6), shrunk heavily towards zero, or from priors such as the Black–Litterman equilibrium (Exercise 4). Note how close the Sharpe ratios are: the differences between methods are smaller than the estimation error of a 50-year Sharpe ratio (about ±0.15), so robustness and risk, not the ranking, should decide.

## Run it

```bash
python3 -m pip install -U jupyter matplotlib numpy pandas scikit-learn scipy
jupyter notebook portfolio_construction.ipynb
```

Or open `index.html` in a browser to read the guide with every output already computed.

Previous: [Chapter 3 · Forecasting Volatility](../03-forecasting-volatility/) · Next: [Chapter 5 · Financial ML Done Right](../05-financial-ml-done-right/)
