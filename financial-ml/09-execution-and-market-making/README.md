# Financial ML 9 — Execution and Market Making

Look inside the market. Intraday volume and volatility from real one- and five-minute bars, a VWAP execution test, the square-root law of price impact, a limit order book simulated from scratch, Almgren–Chriss optimal execution derived and simulated, Avellaneda–Stoikov market making, and a Q-learning agent that rediscovers the optimal trading schedule and then beats it when liquidity changes.

**Time:** About 5 hours with the exercises. **Before this:** Chapters 1–3 and 8 (for Brownian motion and Monte Carlo).

| File | What it is |
|---|---|
| `index.html` | The guide: theory and maths from scratch, all the code with its real outputs, all the figures. |
| `execution_and_market_making.ipynb` | The complete notebook; it reproduces every number and figure in the guide. |
| `animations/execution_lab.html` | Interactive playground: Execution Lab. |
| `datasets/bars_1min.csv` | Cached data (1.4 MB), so everything runs offline. |
| `datasets/bars_5min.csv` | Cached data (2.2 MB), so everything runs offline. |

## What you will do

1. Look at the big picture
2. Get the data
3. How a market works: the limit order book
4. Explore the data: the intraday rhythm of markets
5. Project, part 1: trading the VWAP
6. Price impact: the square-root law
7. Project, part 2: optimal execution with Almgren–Chriss
8. Market making: Avellaneda–Stoikov
9. Reinforcement learning: learning to execute from experience
10. Present your solution
11. Launch, monitor and maintain

## Results

- **Measured on real data.** Intraday volume is U-shaped and volatility highest at the open; a volume-profile schedule tracked VWAP better than TWAP out of sample for 5 of 8 tickers. Square-root-law pre-trade estimates put the cost of trading 10% of a day's volume at 10–41 basis points across the eight tickers.
- **Derived and simulated.** Almgren–Chriss trades a little expected cost for much lower risk (TWAP 25 ± 64 bps against 39 ± 44 bps for κT = 3); Avellaneda–Stoikov halves a market maker's P&L risk for a small loss of profit; a zero-intelligence book reproduces concave impact.
- **Learned.** Q-learning rediscovers the Almgren–Chriss schedule from experience alone and, when liquidity switches between regimes, beats it by 14%.
- **Limitations.** No order-book or trade-level data, so impact parameters come from the literature rather than our own fills; the market-making model has no adverse selection; the RL environments are simulators.

## Run it

```bash
python3 -m pip install -U jupyter matplotlib numpy pandas
jupyter notebook execution_and_market_making.ipynb
```

Or open `index.html` in a browser to read the guide with every output already computed.

Previous: [Chapter 8 · Options, Volatility and Neural Pricers](../08-options-and-neural-pricers/) · Next: [Chapter 10 · Capstone: From Idea to Paper Trading](../10-capstone/)
