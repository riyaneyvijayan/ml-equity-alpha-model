# Machine Learning Driven Multi-Factor Equity Alpha Model

An independent project building a systematic equity selection model using momentum, volatility, and price-based factors, evaluated with a supervised learning framework and backtested against a naive benchmark.

## Overview

This project investigates whether a small set of classic equity factors — momentum, volatility, and price relative to a long-term moving average — can be combined through a machine learning classifier to predict short-term stock outperformance, and whether acting on those predictions improves risk-adjusted returns versus a simple equal-weighted buy-and-hold approach.

## Data

- **Universe:** 31 large-cap US equities spanning technology, financials, energy, healthcare, consumer staples, consumer discretionary, industrials, and utilities
- **Period:** January 2020 – December 2025, daily closing prices
- **Source:** Yahoo Finance (via the `yfinance` Python library)

## Methodology

### Feature engineering

Three factors were constructed for each stock, at each point in time:

| Factor | Definition | Rationale |
|---|---|---|
| Momentum | 126-day (≈6 month) trailing return | Momentum in equities is a well-documented empirical anomaly (Jegadeesh & Titman, 1993): stocks that have performed well over the past 3–12 months tend to continue outperforming over the following months, likely due to underreaction to information and gradual investor herding. |
| Volatility | 21-day (≈1 month) rolling standard deviation of daily returns | Volatility proxies for recent risk and uncertainty in a stock; high-volatility stocks often behave differently in terms of forward returns than stable ones, and volatility is a standard risk-adjustment input in asset pricing. |
| Price factor | % distance of current price from its 200-day moving average | Captures a stock's positioning relative to its longer-term trend — a proxy used in both trend-following (continuation) and mean-reversion (overextension) strategies, giving the model a longer-horizon signal to complement the shorter-horizon momentum factor. |

### Labeling

The target variable is binary: whether a stock's forward 21-day (≈1 month) return is positive (1) or negative (0). Across the full dataset, the label distribution was approximately 56% positive / 44% negative — a reasonably balanced split, consistent with the broadly upward-trending 2020–2025 sample period.

### Train/test split

The dataset was split **chronologically** — the first 80% of the timeline for training (~32,400 rows), the most recent 20% (~8,100 rows) held out for testing. A random split was deliberately avoided, since shuffling time-series data would allow the model to implicitly "see the future" during training (lookahead bias).

### Models

Two classifiers were trained and compared:

1. **Logistic regression** (linear baseline)
2. **XGBoost** (gradient-boosted decision trees, `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`)

## Results

### Classification performance

| Model | Accuracy | Class 0 (down) Precision/Recall | Class 1 (up) Precision/Recall |
|---|---|---|---|
| Logistic Regression | 54.4% | 0.00 / 0.00 | 0.54 / 1.00 |
| XGBoost | 52.5% | 0.44 / 0.16 | 0.54 / 0.83 |

**Note on interpretation:** logistic regression's higher headline accuracy is misleading — it arose from the model defaulting to predicting the majority class ("up") for nearly every observation, which trivially matches the ~56% base rate. It captured no real signal, as reflected in its 0.00 precision/recall on the down-class. XGBoost's slightly lower accuracy came with genuine, if modest, discriminative power on both classes, making it the more meaningful and defensible model despite the lower top-line number.

### Additional evaluation: confusion matrix and ROC curve

| | Predicted Down | Predicted Up |
|---|---|---|
| **Actual Down** | 570 | 3,125 |
| **Actual Up** | 714 | 3,701 |

**ROC AUC: 0.506**

The AUC score is the more complete diagnostic here, and it tempers the earlier precision/recall result: at 0.506, the model's ability to rank stocks by likelihood of rising is only marginally better than random chance (0.5) across all possible decision thresholds. While XGBoost showed non-zero precision/recall on the down-class at the default threshold — a real improvement over logistic regression's total collapse — the AUC indicates the underlying predictive signal is very weak. This is consistent with the broader finding that three simple technical factors carry only a thin, but non-zero, signal for one-month-ahead stock direction. The model shows real (if modest) capability, but should not be overstated as a strong predictive edge.

### Feature importance (XGBoost)

| Feature | Importance |
|---|---|
| Momentum | 0.372 |
| Price factor | 0.339 |
| Volatility | 0.289 |

No single factor dominated; the model draws on a relatively even blend of all three, suggesting the predictive signal (such as it is) is distributed rather than concentrated in one dimension. This is broadly consistent with the factors' theoretical roles: momentum's slight edge aligns with its status as the most empirically robust of the three in academic literature, while volatility and price factor contribute comparable, complementary signal rather than one dominating.

### Backtest (test period only, out-of-sample)

A simple long-only strategy was simulated: on each day, hold an equal-weighted portfolio of stocks the model predicted would rise; hold cash (0% return) for stocks predicted to fall. This was compared against an equal-weighted buy-and-hold benchmark of the full universe over the same period.

| Metric | ML Strategy | Benchmark |
|---|---|---|
| Annualized Sharpe ratio | 1.42 | 1.05 |
| Max drawdown | -13.64% | -16.77% |
| Final value (per $1 invested) | $1.20 | $1.18 |

The strategy modestly outperformed the benchmark in absolute terms, but the more notable result is the improvement in **risk-adjusted return** (higher Sharpe) and a **shallower maximum drawdown** — the model-driven approach reduced downside during the test period's volatility spike without sacrificing return. Given the weak AUC reported above, this result should be read as evidence that even a thin predictive signal can translate into a modest, economically meaningful improvement in portfolio-level risk-adjusted performance — not as evidence of strong directional forecasting ability.

## Limitations

- **Small feature set:** only three factors were used; no fundamental, sentiment, or macro data was incorporated.
- **Simple label:** forward 21-day binary direction is a coarse target; it doesn't account for magnitude of return or transaction costs.
- **No transaction costs or slippage:** the backtest assumes frictionless daily rebalancing, which would not hold in live trading.
- **Single test period:** results reflect one 2024–2025 out-of-sample window; performance is not validated across multiple market regimes (e.g. a sustained bear market).
- **Weak standalone predictive power:** the near-random AUC (0.506) indicates limited directional forecasting ability at the individual-stock level; this is a realistic and expected outcome for short-horizon equity direction prediction using simple technical factors, not a limitation unique to this implementation.

## Tools Used

Python, pandas, NumPy, scikit-learn, XGBoost, yfinance, Matplotlib, Jupyter Notebook

## Possible Extensions

- Incorporate fundamental factors (valuation, quality, growth) and additional technical indicators (RSI, MACD, Bollinger Bands)
- Test alternative labeling schemes (e.g. return magnitude via regression rather than binary classification)
- Walk-forward / rolling-window validation across multiple market regimes, using time-series-aware cross-validation
- Incorporate transaction costs and turnover constraints into the backtest
- Compare against additional models (Random Forest, LightGBM) and tune hyperparameters systematically (GridSearchCV/Optuna)
