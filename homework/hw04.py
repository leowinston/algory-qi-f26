"""Homework 4 starter — Algory QI Education, Fall 2026

The backtester is the piece worth getting right. Everything after it in the
semester assumes it works.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

LOOKBACK = 60


# ---------------------------------------------------------------- Q1
def load_and_split(split_date="2011-09-09", prices=None, warmup=LOOKBACK):
    """Split ^SP500TR since 1990 into training and test at `split_date`.

    Training ends on the split date's close. Test begins `warmup` trading days
    BEFORE it, so the 60-day window is already full on the first day traded out
    of sample. Those overlapping days are history the strategy reads, never days
    it earns a return on. Cut the series at one date with no overlap and the
    first 60 days of the test set have no window, so the out-of-sample period
    silently starts three months late.

    September 9, 2011 is a Friday. A split date has to be a trading day, or
    "ends on that close" names a close that does not exist.

    Pass `prices` to split a series you already have, which is how the checker
    tests this without a download. Return (training, test).
    """
    # Series of closes, DataFrame --> Series
    if prices is None:
        raw = yf.download("^SP500TR", start="1990-01-01", progress=False)
        prices = raw["Close"].squeeze()

    # Split in time order, never random: neighbouring days are correlated, so shuffling leaks the future
    split_pos = prices.index.get_loc(pd.Timestamp(split_date)) # row number of the last training day
    train = prices.iloc[: split_pos + 1] # fit and tune here, look as often as you like
    test = prices.iloc[split_pos - (warmup - 1):] # touch once; first `warmup` rows are window only
    mean_prices = (train.mean(), test.mean())

    print(f"Train: {len(train)} days, mean price: {mean_prices[0]:,.2f}")
    print(f"Test:  {len(test)} days (including {warmup} warm-up rows), mean price: {mean_prices[1]:,.2f}")
    return train, test




# ---------------------------------------------------------------- Q2, Q4, Q5
def always_hold(window):
    """Always invest. The control case, and the one that proves the backtester."""
    return True


def always_cash(window):
    """Never invest. Must produce exactly 0% return."""
    return False


def make_random_hold(p=0.7, seed=0):
    """Return a strategy that invests with probability p each day.

    A factory, not a plain function, because the generator has to live somewhere
    and a default argument is the wrong place: `rng=np.random.default_rng(0)` in
    a signature is evaluated once when the module loads, so every backtest in the
    session shares one advancing generator and the same backtest returns a
    different number each time you run it. Building the generator here gives one
    per strategy, so `backtest(train, make_random_hold())` is reproducible.
    """
    rng = np.random.default_rng(seed)

    def random_hold(window):
        return bool(rng.random() < p)

    return random_hold


def five_day_reversal(window):
    """The rule from class: hold unless the last 5 days were positive.

    Measure that return from the close 5 days before today to today's close,
    both of which are inside the window.
    """
    closes = np.asarray(window, dtype=float) # list or Series --> array
    today_close = closes[-1]
    five_days_ago_close = closes[-6]
    five_day_return = (today_close / five_days_ago_close) - 1.0 
    return bool(five_day_return <= 0) # up --> cash (False); down or flat --> hold (True)


# ---------------------------------------------------------------- Q6
def make_above_ls_line(rising=False):
    # A: hold if today's log price is above the least-squares line (A2 also needs the line sloping up)
    def above_ls_line(window):
        x = np.log(np.asarray(window, dtype=float)) # (60,)
        t_c = np.arange(len(x)) - (len(x) - 1) / 2 # centered t makes [1, t_c] orthogonal, so LS is a projection
        slope = (t_c @ x) / (t_c @ t_c)
        fitted_today = x.mean() + slope * t_c[-1]
        return bool(x[-1] > fitted_today and (slope > 0 or not rising))

    return above_ls_line


def eigen_share(windows):
    # top eigenvalue's share of the 10 lagged return copies; takes one window (60,) or a stack (n, 60)
    r = np.diff(np.log(np.asarray(windows, dtype=float)), axis=-1) # (..., 59) daily log returns
    lags = sliding_window_view(r, 50, axis=-1) # (..., 10, 50): row k is r shifted by k days
    vals = np.linalg.eigvalsh(lags @ lags.mT) # (..., 10), ascending, so the top one is last
    return vals[..., -1] / vals.sum(axis=-1)


def make_eigen_share(s):
    # B: hold if the top eigenvalue's share beats s AND the window is up
    def eigen_share_up(window):
        closes = np.asarray(window, dtype=float)
        return bool(eigen_share(closes) > s and closes[-1] > closes[0])

    return eigen_share_up


def make_calm_only(c):
    # C: hold unless the last 10 days are more than c times as volatile as the window.
    def calm_only(window):
        r = np.diff(np.log(np.asarray(window, dtype=float))) # (59,)
        vol_short = r[-10:].std()
        vol_long = r.std()
        return bool(vol_short <= c * vol_long)

    return calm_only



def my_strategy(window):
    """Your own rule. It sees only `window`, the previous LOOKBACK closes.

    It must never look at anything after that. That is the whole assignment.
    """
    # final rule, C1: picked on training by best annualised / |max drawdown|
    return make_calm_only(1.5)(window)


# ---------------------------------------------------------------- Q3
def backtest(prices, strategy, lookback=LOOKBACK):
    """Walk forward one day at a time and apply the strategy's decision.

    The convention, stated exactly, because everything else depends on it:

        on day i, the strategy is handed prices[i - lookback + 1 : i + 1],
        the `lookback` closes ENDING WITH today's, today's included;
        if it says hold you earn prices[i + 1] / prices[i] - 1, otherwise zero.

    So i runs from lookback - 1 to len(prices) - 2. The first decision is made on
    the 60th close and the first return earned runs from the 60th close to the
    61st. Today's close is known when you decide, so withholding it throws away a
    day of information for nothing; tomorrow's close is what you are predicting,
    so reading it is the bug this whole session is about.

    Return a dict with at least:
        total_return, annualised_return, max_drawdown, fraction_invested

    Annualise by trading days, 252 a year, compounded: value ** (252 / days) - 1.

    Sanity checks before you trust it:
        always_hold  must match buy-and-hold over the days actually traded,
                     prices[-1] / prices[lookback - 1] - 1, not over the
                     whole file
        always_cash  must return exactly 0.0
    If either fails you have an off-by-one, and every number after it is fiction.
    """
    prices = pd.Series(prices, dtype=float) # a list or array gets a plain 0..n-1 index; a Series keeps its dates
    closes = prices.to_numpy()
    days_traded = len(closes) - lookback # first 59 closes are window only, not results

    # Walk forward. positions[i] = what we decided at close t: True = hold, False = cash
    positions = np.zeros(days_traded, dtype=bool)
    for i in range(days_traded):
        t = i + lookback - 1 # today, the day we decide on
        known = prices.iloc[t - lookback + 1 : t + 1] 
        positions[i] = strategy(known)

    # Daily return = what the index did AFTER the decision (close t --> close t+1)
    traded_closes = closes[lookback - 1 :]
    index_daily_returns = np.diff(traded_closes) / traded_closes[:-1] # (P_{t+1} - P_t) / P_t
    strategy_daily_returns = positions * index_daily_returns # cash day --> 0

    # Each return is dated by the close that earned it (t+1)
    earned_dates = prices.index[lookback:]
    strategy_daily_returns = pd.Series(strategy_daily_returns, index=earned_dates)
    index_daily_returns = pd.Series(index_daily_returns, index=earned_dates)

    # Strategy: equity curve = portfolio value over time, starts at 1
    strategy_equity = (1.0 + strategy_daily_returns).cumprod()
    strategy_final_value = strategy_equity.iloc[-1]
    strategy_running_peak = strategy_equity.cummax().clip(lower=1.0) # best value so far, start of 1 counts
    strategy_drawdown = strategy_equity / strategy_running_peak - 1.0 

    # Index (buy and hold, the benchmark we must beat)
    index_equity = (1.0 + index_daily_returns).cumprod()
    index_final_value = index_equity.iloc[-1]
    index_running_peak = index_equity.cummax().clip(lower=1.0)
    index_drawdown = index_equity / index_running_peak - 1.0

    return {
        "total_return": float(strategy_final_value - 1.0),
        "annualised_return": float(strategy_final_value ** (252.0 / days_traded) - 1.0), # compounded, not divided
        "max_drawdown": float(strategy_drawdown.min()), # worst fall from a peak: what it felt like to hold
        "fraction_invested": float(positions.mean()),
        "value": strategy_equity,
        "index": {
            "total_return": float(index_final_value - 1.0),
            "annualised_return": float(index_final_value ** (252.0 / days_traded) - 1.0),
            "max_drawdown": float(index_drawdown.min()),
            "value": index_equity,
        },
    }

def main():
    train, test = load_and_split()
    for name, strat in [("always hold", always_hold),
                        ("always cash", always_cash),
                        ("random 70%", make_random_hold(0.7, seed=0)),
                        ("5-day reversal", five_day_reversal)]:
        r = backtest(train, strat)
        print(f"{name:<16} total {r['total_return']:+.1%}   annual {r['annualised_return']:+.2%}")

    # Q3: always-hold vs index, both rebased to 1; dashed on top so the overlap shows
    hold_run = backtest(train, always_hold)
    print(f"{'index':<16} total {hold_run['index']['total_return']:+.1%}   annual {hold_run['index']['annualised_return']:+.2%}")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hold_run["index"]["value"], label="Index (^SP500TR)", linewidth=2.5)
    ax.plot(hold_run["value"], label="Always hold", linestyle="--", linewidth=1.5)
    ax.set_title("Always hold vs. index, training set, rebased to 1")
    ax.set_ylabel("Portfolio value")
    ax.legend()
    fig.tight_layout()
    fig.savefig("hw04_q3_always_hold.png", dpi=150)
    plt.close(fig)

    # Q6: B thresholds come from training windows only, the exact ones the backtest decides on
    train_windows = sliding_window_view(train.to_numpy(), LOOKBACK)[:-1] # (days traded, 60)
    shares = eigen_share(train_windows)
    p25, p50, p75 = np.percentile(shares, [25, 50, 75])
    print(f"\nEigen share, training windows: 25th {p25:.4f}   50th {p50:.4f}   75th {p75:.4f}")

    # Q6: every variant tried, kept or not
    variants = [
        ("always hold", always_hold),
        ("A1 log price > LS line", make_above_ls_line()),
        ("A2 log price > LS line, slope > 0", make_above_ls_line(rising=True)),
        (f"B1 eigen share > {p50:.4f} (p50), window up", make_eigen_share(p50)),
        (f"B2 eigen share > {p75:.4f} (p75), window up", make_eigen_share(p75)),
        ("C1 hold unless 10d vol > 1.5x 59d vol", make_calm_only(1.5)),
        ("C2 hold unless 10d vol > 2.0x 59d vol", make_calm_only(2.0)),
    ]
    print(f"\n{'rule':<44} {'total':>9} {'annual':>8} {'max dd':>8} {'invested':>9} {'ann/|dd|':>9}")
    for name, strat in variants:
        r = backtest(train, strat)
        return_per_drawdown = r["annualised_return"] / abs(r["max_drawdown"])
        print(f"{name:<44} {r['total_return']:>+9.1%} {r['annualised_return']:>+8.2%} "
              f"{r['max_drawdown']:>+8.1%} {r['fraction_invested']:>9.3f} {return_per_drawdown:>9.3f}")

    # Q7: the final rule
    print("\nFinal rule: C1, hold the index unless 10-day volatility > 1.5x the window's 59-day volatility")

    # Q8: test set, run once, never tuned after
    final_run = backtest(test, my_strategy)
    first_traded = test.index[LOOKBACK].date()
    print(f"\nTest set, {first_traded} to {test.index[-1].date()}")
    print(f"{'':<10} {'total':>9} {'annual':>8} {'max dd':>8} {'invested':>9}")
    print(f"{'strategy':<10} {final_run['total_return']:>+9.1%} {final_run['annualised_return']:>+8.2%} "
          f"{final_run['max_drawdown']:>+8.1%} {final_run['fraction_invested']:>9.3f}")
    print(f"{'index':<10} {final_run['index']['total_return']:>+9.1%} {final_run['index']['annualised_return']:>+8.2%} "
          f"{final_run['index']['max_drawdown']:>+8.1%} {1.0:>9.3f}")

    # Q9: what the rule says today, from the most recent 60 closes
    today_window = test.iloc[-LOOKBACK:]
    signal = "HOLD" if my_strategy(today_window) else "CASH"
    print(f"\nToday: {signal} (last close used {today_window.index[-1].date()})")

    # Check yourself: index over the full period, train + test without the 60 overlap rows
    full = pd.concat([train, test.iloc[LOOKBACK:]])
    full_run = backtest(full, always_hold)
    print(f"Index annualised, full period: {full_run['index']['annualised_return']:+.2%}")


if __name__ == "__main__":
    main()