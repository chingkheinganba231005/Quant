# Financial ML 3 — Forecasting Volatility

Direction is nearly unpredictable, but risk is not. Measure volatility from prices, forecast it with EWMA, GARCH (fitted by maximum likelihood from scratch), the HAR model, the VIX and gradient boosting, judge the forecasts with the right loss, and use them to run a volatility-targeted portfolio.

**Time:** About 5 hours with the exercises. **Before this:** Chapters 1 and 2.

| File | What it is |
|---|---|
| `index.html` | The guide: theory and maths from scratch, all the code with its real outputs, all the figures. |
| `forecasting_volatility.ipynb` | The complete notebook; it reproduces every number and figure in the guide. |
| `animations/garch_lab.html` | Interactive playground: GARCH Lab. |
| `datasets/spy_ohlc.csv` | Cached data (453 kB), so everything runs offline. |
| `datasets/vix.csv` | Cached data (148 kB), so everything runs offline. |

## What you will do

1. Look at the big picture
2. Get the data
3. Measuring volatility from prices
4. The models
5. Walk-forward evaluation
6. Use the forecasts: volatility targeting
7. Present your solution
8. Launch, monitor and maintain

## Results

- **Deliverable:** a daily SPY 21-day volatility forecast from the **HAR (+VIX)** model, refitted every year, with EWMA as a fallback when inputs are missing.
- **Accuracy:** best out-of-sample QLIKE of nine models over 2005–2026; reacts within days to crises; never negative or explosive by construction (log targets).
- **Use:** volatility targeting cut the variability of risk and the tail of daily losses sharply.
- **Limits:** every model lags sudden jumps (they forecast tomorrow from today), and none predicts the *start* of a crisis. Overnight events (the 2020 lockdowns, the 2025 tariff announcements) hit before any forecast can react.

## Run it

```bash
python3 -m pip install -U jupyter arch lightgbm matplotlib numpy pandas scipy
jupyter notebook forecasting_volatility.ipynb
```

Or open `index.html` in a browser to read the guide with every output already computed.

Previous: [Chapter 2 · A Backtester from Scratch](../02-backtesting-trend-following/) · Next: [Chapter 4 · Portfolio Construction out of Sample](../04-portfolio-construction/)
