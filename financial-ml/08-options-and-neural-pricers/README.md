# Financial ML 8 — Options, Volatility and Neural Pricers

Price an option four ways (replication on a tree, the Black–Scholes formula, Monte Carlo and Fourier integration), read the market's implied volatility from a real SPY option chain, delta-hedge in simulation, measure the variance risk premium over 33 years, calibrate the Heston stochastic-volatility model, and train a neural network from scratch that replaces the pricer.

**Time:** About 5 hours with the exercises. **Before this:** Chapters 1–3; the probability chapter of the prerequisite guide.

| File | What it is |
|---|---|
| `index.html` | The guide: theory and maths from scratch, all the code with its real outputs, all the figures. |
| `options_and_neural_pricers.ipynb` | The complete notebook; it reproduces every number and figure in the guide. |
| `animations/options_lab.html` | Interactive playground: Options Lab. |
| `datasets/spy_options.csv` | Cached data (318 kB), so everything runs offline. |
| `datasets/spy_vix_daily.csv` | Cached data (335 kB), so everything runs offline. |

## What you will do

1. Look at the big picture
2. Setup and data
3. What an option is worth at expiry, and one law that needs no model
4. Pricing by replication: the binomial tree
5. The Black–Scholes formula
6. Monte Carlo pricing
7. Implied volatility and the smile
8. Delta hedging: where the option's value comes from
9. Project: is selling volatility a free lunch?
10. Stochastic volatility: the Heston model
11. A neural network that replaces the pricer
12. Present your solution
13. Launch, monitor and maintain

## Results

- **What worked.** Four pricing methods that agree with one another (tree, formula, Monte Carlo, Fourier); implied volatilities computed from raw trade prices with parity-implied forwards; a calibrated Heston model that fits SPY from one month to 13 months with an average error of 0.58 vol points; and a from-scratch neural network that reproduces it.
- **What the data say.** The SPY volatility skew is steep and persistent. Over 33 years, implied variance exceeded realised variance in 83% of months; the premium is real but its skewness (-6.4) shows it is compensation for crash risk, not free money.
- **Limitations.** One snapshot of the option chain, last-trade prices instead of synchronised quotes, the variance swap approximated by VIX on SPX against realised variance on SPY, and no transaction costs (option spreads are wide, often several percent of the premium for out-of-the-money options).

## Run it

```bash
python3 -m pip install -U jupyter matplotlib numpy pandas scipy
jupyter notebook options_and_neural_pricers.ipynb
```

Or open `index.html` in a browser to read the guide with every output already computed.

Previous: [Chapter 7 · Pairs Trading and Statistical Arbitrage](../07-pairs-trading/) · Next: [Chapter 9 · Execution and Market Making](../09-execution-and-market-making/)
