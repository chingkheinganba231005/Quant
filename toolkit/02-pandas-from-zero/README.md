# Toolkit 2 — pandas from Zero

Tables with names and dates: loading real market data, selecting and cleaning it, reshaping it, grouping it, joining it, and the time-series tools (resampling, rolling windows, shifting) that every backtest is built from.

**Time:** About 3 to 4 hours with the exercises. **Before this:** Toolkit 1 (NumPy).

| File | What it is |
|---|---|
| `index.html` | The guide: theory and maths from scratch, all the code with its real outputs, all the figures. |
| `pandas_from_zero.ipynb` | The complete notebook; it reproduces every number and figure in the guide. |
| `animations/pandas_playground.html` | Interactive playground: pandas Playground. |
| `datasets/fred_rates.csv` | Cached data (108 kB), so everything runs offline. |
| `datasets/prices_long.csv` | Cached data (2.3 MB), so everything runs offline. |

## What you will do

1. Setup
2. Series: an array with an index
3. DataFrame: a table of aligned Series
4. Loading real data
5. Selecting data: `[]`, `.loc`, `.iloc`
6. From long to wide: one column per ticker
7. The DatetimeIndex: dates as the index
8. Missing data
9. Resampling: changing the frequency
10. Rolling, expanding and exponential windows
11. groupby: split, apply, combine
12. Combining tables: `concat`, `merge`, `join`, `merge_asof`
13. Performance and good style
14. A small case study: the crash of March 2020
15. The mistakes that cost quants money

## Run it

```bash
python3 -m pip install -U jupyter matplotlib numpy pandas
jupyter notebook pandas_from_zero.ipynb
```

Or open `index.html` in a browser to read the guide with every output already computed.

Previous: [Toolkit 1 · NumPy from Zero](../01-numpy-from-zero/) · Next: [Toolkit 3 · Plotting from Zero](../03-plotting-from-zero/)
