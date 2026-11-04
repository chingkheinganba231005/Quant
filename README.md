# Quant from Zero

From **zero finance knowledge** to **financial machine learning and algorithmic trading**: interactive lessons, a Python data toolkit, and end-to-end projects on real market data.

**Start here:** open [`index.html`](index.html) in a browser (or enable GitHub Pages for this repository).

## Part 0 · Finance from zero

No finance knowledge assumed. Money, banks, interest, bonds, stocks, markets, risk, portfolios, CAPM, accounts, derivatives, Black–Scholes, efficient markets, backtesting and the basics of financial ML, in 18 interactive lessons and a roadmap.

| # | Chapter | What you learn |
|---|---|---|
| 0 | [Finance From Zero](00-finance-from-zero/) | 18 lessons from “what is money?” to options pricing, efficient markets and your first backtest. Every lesson has a like-you’re-five explanation, the real idea, a quant view and interactive demos. |

## Part 1 · The Python data toolkit

The three libraries every data scientist and quant uses daily, from the first line to the advanced tricks: NumPy for arrays and maths, pandas for tables and time series, Matplotlib (with seaborn and Plotly) for charts. All examples use financial data.

| # | Chapter | What you learn |
|---|---|---|
| T1 | [NumPy from Zero](toolkit/01-numpy-from-zero/) | Arrays and dtypes, slicing and the view trap, vectorisation, axes, broadcasting, linear algebra for portfolios (covariance, solve, PCA, Cholesky, least squares), random numbers and Monte Carlo, the bootstrap, rolling windows and einsum. |
| T2 | [pandas from Zero](toolkit/02-pandas-from-zero/) | Series and DataFrames, loc and iloc, cleaning and missing data, groupby, merge and reshape, and the time-series toolkit: dates, resampling, rolling and exponential windows, shifting without look-ahead, panel data of many stocks. |
| T3 | [Plotting from Zero](toolkit/03-plotting-from-zero/) | Matplotlib’s figure and axes model, every common chart, dates, twin axes and annotations, publication styling, then seaborn for statistical plots and Plotly for interactive dashboards, ending with a full market report. |

## Part 2 · Financial ML and algorithmic trading

Theory and maths from scratch, then end-to-end projects on real market data, each one harder than the last. Every chapter has a guide, a notebook that reproduces every number and figure, the dataset, and an interactive playground.

| # | Chapter | What you learn |
|---|---|---|
| 1 | [The Statistics of Financial Returns](financial-ml/01-statistics-of-returns/) | Returns and stationarity, fat tails, volatility clustering, autocorrelation tests, maximum likelihood, the bootstrap and block bootstrap, and why one backtest proves almost nothing (multiple testing, the deflated Sharpe ratio). |
| 2 | [A Backtester from Scratch: Trend Following](financial-ml/02-backtesting-trend-following/) | Build an honest vectorised backtester, test time-series momentum on real ETFs, add costs and volatility targeting, walk forward, and measure how much of the result is luck. |
| 3 | [Forecasting Volatility](financial-ml/03-forecasting-volatility/) | Realised volatility, EWMA, GARCH(1,1) fitted by maximum likelihood from scratch, HAR and gradient boosting, judged with the right loss, then used to size positions. |
| 4 | [Portfolio Construction out of Sample](financial-ml/04-portfolio-construction/) | Mean–variance and why it fails on real estimates, shrinkage, minimum variance, risk parity and hierarchical risk parity, compared in a walk-forward test. |
| 5 | [Financial Machine Learning Done Right](financial-ml/05-financial-ml-done-right/) | Labels (fixed horizon, triple barrier), meta-labelling, overlapping samples, purged and embargoed cross-validation, feature importance and the probability of backtest overfitting. |
| 6 | [Cross-Sectional Factors and ML Ranking](financial-ml/06-cross-sectional-ml/) | Momentum, reversal and low-risk signals across a stock universe, information coefficients, Fama–MacBeth regressions, long–short portfolios, and ML rankers judged after costs. |
| 7 | [Pairs Trading and Statistical Arbitrage](financial-ml/07-pairs-trading/) | Cointegration and the ADF test from scratch, the Ornstein–Uhlenbeck spread, a Kalman-filter hedge ratio, and how fast relative-value edges decay. |
| 8 | [Options, Volatility and Neural Pricers](financial-ml/08-options-and-neural-pricers/) | Black–Scholes, trees and Monte Carlo, implied volatility, a delta-hedging simulation, the Heston model, and a neural-network surrogate, written from scratch, that prices options about a hundred times faster. |
| 9 | Execution and Market Making *(coming)* | Price impact, Almgren–Chriss optimal execution, an order-book simulator, Avellaneda–Stoikov market making and a reinforcement-learning trader. |
| 10 | Capstone: From Idea to Paper Trading *(coming)* | A complete research pipeline with regime detection: hypothesis, data, features, validation, portfolio, risk, a research report, and a paper-trading loop you can run. |

## Running the notebooks

```bash
python3 -m pip install -U jupyter numpy pandas matplotlib seaborn plotly scipy scikit-learn statsmodels lightgbm yfinance
jupyter notebook
```

Each chapter folder contains `index.html` (the guide), a Jupyter notebook that reproduces every number and figure, `images/`, `datasets/` and `animations/` (interactive playgrounds). Guides and notebooks are generated from a single literate source, `chapter.md`, with `python3 tools/build.py <chapter-folder>`; the hub and this README with `python3 tools/hub.py`.

*Educational material, not investment advice.*
