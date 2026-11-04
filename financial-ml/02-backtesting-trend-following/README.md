# Financial ML 2 — A Backtester from Scratch: Trend Following

Build an honest backtesting engine, test one of the oldest strategies in markets (time-series momentum) on ten real ETFs, and find out how much of the result survives costs, parameter choice, walk-forward testing and the question “is it luck?”.

**Time:** About 5 hours with the exercises. **Before this:** Chapter 1; Toolkit 1–3; Finance From Zero lesson 16.

| File | What it is |
|---|---|
| `index.html` | The guide: theory and maths from scratch, all the code with its real outputs, all the figures. |
| `trend_following_backtest.ipynb` | The complete notebook; it reproduces every number and figure in the guide. |
| `animations/trend_lab.html` | Interactive playground: Trend Lab. |
| `datasets/etf_adjusted_close.csv` | Cached data (502 kB), so everything runs offline. |
| `datasets/tbill_3m.csv` | Cached data (94 kB), so everything runs offline. |

## What you will do

1. Look at the big picture
2. Get the data
3. Discover the data: is there trend?
4. Build the backtester
5. A first strategy: the 200-day moving average on SPY
6. The main strategy: diversified time-series momentum
7. Fine-tune the strategy, honestly
8. Is it luck?
9. Present your solution
10. Launch, monitor and maintain

## Results

| | TSMOM (10 ETFs) | SPY | 60/40 |
|---|---|---|---|
| Sharpe (net of 5 bps) | 0.37 | 0.53 | 0.62 |
| Max drawdown | -42% | -56% | -33% |
| Correlation with SPY | -0.08 | 1 | +0.97 |
| Average month in SPY's worst fifth | +1.5% | -5.9% | -3.3% |
| 50/50 with SPY: Sharpe | 0.67 | | |

**Recommendation.** As a stand-alone investment the evidence is weak in this sample: a Sharpe ratio of 0.37 with a bootstrap interval that includes zero. As a **diversifier** it is attractive, because it earned in the months stocks suffered most: a 50/50 mix with SPY has a Sharpe ratio of 0.67. The short positions lost money over this period (Exercise 4: the long-only version scores higher but is more correlated with stocks), which is a known feature of an era when stocks and bonds mostly rose. If we trade it: fix the rules now (12-month or averaged lookbacks, 40% per-asset volatility scaling, monthly rebalancing), size the allocation for its drawdowns, and evaluate live results against the bootstrap interval rather than against the backtest's point estimate.

## Run it

```bash
python3 -m pip install -U jupyter matplotlib numpy pandas scipy
jupyter notebook trend_following_backtest.ipynb
```

Or open `index.html` in a browser to read the guide with every output already computed.

Previous: [Chapter 1 · The Statistics of Returns](../01-statistics-of-returns/) · Next: [Chapter 3 · Forecasting Volatility](../03-forecasting-volatility/)
