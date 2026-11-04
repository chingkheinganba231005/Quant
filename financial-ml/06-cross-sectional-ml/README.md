# Financial ML 6 — Cross-Sectional Factors and ML Ranking

Stop asking whether the market will go up; ask which assets will beat the others. Build the factor researcher’s toolkit from scratch (information coefficients, quintile sorts, Fama–MacBeth regressions, the fundamental law), then compare a theory-driven composite with ridge and gradient-boosted rankers in a 56-year walk-forward on 49 industries.

**Time:** About 5 hours with the exercises. **Before this:** Chapters 1–5; Finance From Zero lesson 10.

| File | What it is |
|---|---|
| `index.html` | The guide: theory and maths from scratch, all the code with its real outputs, all the figures. |
| `cross_sectional_ml.ipynb` | The complete notebook; it reproduces every number and figure in the guide. |
| `animations/factor_lab.html` | Interactive playground: Factor Lab. |
| `datasets/ff_factors_monthly.csv` | Cached data (56 kB), so everything runs offline. |
| `datasets/industry49_monthly_vw.csv` | Cached data (426 kB), so everything runs offline. |

## What you will do

1. Look at the big picture
2. Get the data
3. The fundamental law of active management
4. Signals from scratch
5. Testing signals: the factor researcher's toolkit
6. From signals to a model: the walk-forward
7. Momentum crashes and volatility scaling
8. Does the edge decay?
9. Present your solution
10. Launch, monitor and maintain

## Results

- **Finding.** In the cross-section of industries, the robust signals are **momentum** (12-month and, unlike stocks, 1-month) and **seasonality**. Signals that work for individual stocks (short-term reversal, low volatility, low beta, lottery stocks) do not transfer.
- **Models.** A composite with signs copied from the stock literature failed (Sharpe -0.05). Models that learned weights and signs from past data in a walk-forward (0.42 ridge, 0.41 LightGBM) worked, but did not beat volatility-scaled industry momentum (0.54), and their alphas after the four standard factors are not significant.
- **Why ML did not win here.** With ICs of a few hundredths and 49 correlated assets (about 23 independent ones, Exercise 3), there is little that a flexible model can learn beyond the signs. ML earns its keep with many more assets (thousands of stocks), more varied inputs (fundamentals, analyst data, text) and careful validation, which is where the large quantitative equity funds use it.
- **Risks.** Momentum crashes (use volatility scaling), decay after publication (the IC halved), crowding, and costs: industry portfolios are cheap to trade, individual small stocks are not.

## Run it

```bash
python3 -m pip install -U jupyter lightgbm matplotlib numpy pandas scikit-learn scipy
jupyter notebook cross_sectional_ml.ipynb
```

Or open `index.html` in a browser to read the guide with every output already computed.

Previous: [Chapter 5 · Financial ML Done Right](../05-financial-ml-done-right/) · Next: [Chapter 7 · Pairs Trading and Statistical Arbitrage](../07-pairs-trading/)
