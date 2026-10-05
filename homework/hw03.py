"""Homework 3: Probability, and testing independence — Algory QI Education, Fall 2026

Dice and coin simulations, whether SPY daily returns are independent, and
expected present value with a yearly survival probability.
Check with `python checks/check_hw03.py homework/hw03.py` from the repo root.
"""
import pathlib

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

PLOTS_DIR = pathlib.Path(__file__).parent.parent / "plots"


# ---------------------------------------------------------------- Q1
def simulate_dice(trials: int, seed: int = 0) -> float:
    """Pick a 4-sided or 6-sided die at random, roll it, repeat.

    Return the estimated P(picked the 4-sided die | rolled a 1).
    The exact answer is 0.6. Take a seed so your result is reproducible.
    """
    rng = np.random.default_rng(seed)

    # pick a die per trial: 0 --> 4-sided, 1 --> 6-sided. shape (trials,)
    die = (rng.random(trials) >= 0.5).astype(int)

    # roll each die, uniform on 1..size inclusive (high is exclusive). shape (trials,)
    size = 4 + 2 * die
    rolls = rng.integers(1, size + 1)

    # keep only the dice that rolled a 1: an array of 0s (four) and 1s (six)
    die_given_one = die[rolls == 1]

    six_one_ct = die_given_one.sum()
    total_ones = die_given_one.size

    if total_ones == 0:  # only possible with a tiny trial count
        return float("nan")
    four_one_ct = total_ones - six_one_ct

    return float(four_one_ct / total_ones)


# ---------------------------------------------------------------- Q2
def simulate_coins(trials: int, seed: int = 0) -> float:
    """Flip three fair coins, get paid (heads x tails). Return the mean payout.

    The exact answer is 1.5.
    """
    rng = np.random.default_rng(seed)

    # one row per trial, one column per coin: 1 --> heads, 0 --> tails. shape (trials, 3)
    flips = rng.integers(0, 2, size=(trials, 3))

    # axis=1 sums across each row's 3 coins. shape (trials,)
    heads = flips.sum(axis=1)
    tails = 3 - heads
    payouts = heads * tails  # 0H --> 0, 1H --> 2, 2H --> 2, 3H --> 0

    return float(payouts.mean())


# ---------------------------------------------------------------- Q5/Q6
def count_next_day_downs(returns: ArrayLike, threshold: float) -> tuple[int, int]:
    """Count days below threshold, and how many of those were followed by a down day.

    Shared by both conditional probabilities. main() also uses it to print the counts.
    """
    r = np.asarray(returns, dtype=float)

    # line each day up with the next: today[i] and tomorrow[i] are back to back. shape (n - 1,)
    today = r[:-1]  # last day drops out, it has no tomorrow yet
    tomorrow = r[1:]

    # bool masks. & is elementwise AND (plain `and` fails on arrays), .sum() counts the Trues
    today_hit = today < threshold
    hit_then_down = today_hit & (tomorrow < 0)

    return int(today_hit.sum()), int(hit_then_down.sum())


def p_down(returns: ArrayLike) -> float:
    """Fraction of days with a negative return. Count them yourself."""
    r = np.asarray(returns, dtype=float)
    down_days = r < 0  # bool mask, shape (n,)

    return float(down_days.sum() / r.size)


def p_down_given_down(returns: ArrayLike) -> float:
    """P(tomorrow is down | today was down), counted directly from the series."""
    down_today, down_then_down = count_next_day_downs(returns, threshold=0.0)

    if down_today == 0:  # no down days, nothing to condition on
        return float("nan")
    return down_then_down / down_today


def p_down_given_big_drop(returns: ArrayLike, threshold: float = -0.02) -> float:
    """P(tomorrow is down | today fell more than the threshold).

    Also report how many days the estimate uses. A handful of days is a much
    weaker claim than thousands, and the count is what tells a reader which of
    the two this is. (Returns just the probability so the checker can compare
    it. main() gets the count from count_next_day_downs.)
    """
    big_drops, big_drop_then_down = count_next_day_downs(returns, threshold)

    if big_drops == 0:
        return float("nan")
    return big_drop_then_down / big_drops


# ---------------------------------------------------------------- Q7
def present_value(cash_flows: ArrayLike, rate: float) -> float:
    """Present value of a list of cash flows, the first arriving in one year.
    present_value([10, 15, 20], 0.10) -> 36.51

    Carried over from Homework 2.
    """
    cfs = np.asarray(cash_flows, dtype=float)
    years = np.arange(1, len(cfs) + 1) # [1, 4) --> years: 1,2,3
    discount_factors = (1.0 + rate) ** years # df[i] = (1 + r)^(i + 1)

    return float(np.sum(cfs / discount_factors))


def expected_present_value(cash_flows: ArrayLike, rate: float, survival_prob: float) -> float:
    """Present value where the company survives EACH year with survival_prob.

    Year 1 is certain. Year 2 arrives with probability survival_prob, year 3
    with survival_prob squared, and so on. This is a yearly hazard rate.

    Note this is deliberately more general than the Session 3 slide, which had a
    single shutdown event after year 1 and came to 16.98. A yearly 50% survival
    is a harsher assumption and gives 15.10. Getting 16.98 here means you applied
    the probability once instead of compounding it.
    """
    cfs = np.asarray(cash_flows, dtype=float)
    years = np.arange(1, len(cfs) + 1)

    # chance the company is still alive to pay year t: p^0, p^1, p^2, ... shape (n,)
    p_reach_year = survival_prob ** (years - 1)
    expected_cfs = cfs * p_reach_year  # weight each flow by the chance it shows up

    return present_value(expected_cfs, rate)  # then discount exactly like HW2


# ---------------------------------------------------------------- your answers
def main() -> None:
    dice_est = simulate_dice(100_000, seed=0)
    print(f"Q1  P(4-sided | rolled a 1) = {dice_est:.4f} (exact 0.6000, diff {dice_est - 0.6:+.4f}, seed 0)")

    coin_est = simulate_coins(100_000, seed=0)
    print(f"Q2  expected three-coin payout = {coin_est:.4f} (exact 1.5000, diff {coin_est - 1.5:+.4f}, seed 0)")

    # Same seed at every size, so only the sample size changes between runs
    trial_counts = [100, 1_000, 10_000, 100_000]
    coin_estimates = [simulate_coins(n, seed=0) for n in trial_counts]

    print("\nQ3  Three-coin payout vs. number of trials")
    for n, est in zip(trial_counts, coin_estimates):
        print(f"    {n:>7,} trials: {est:.4f} (diff {est - 1.5:+.4f})")

    plt.figure()
    plt.plot(trial_counts, coin_estimates, marker="o", linewidth=2, label="Simulated payout")
    plt.axhline(1.5, color="gray", linestyle="--", linewidth=1.5, label="Exact (1.5)")
    plt.xscale("log")  # 100 --> 100,000 is 3 orders of magnitude, log spaces them evenly
    plt.xticks(trial_counts, [f"{n:,}" for n in trial_counts])  # "1,000" reads easier than 10^3
    plt.xlabel("Number of Trials (log scale)")
    plt.ylabel("Expected Payout")
    plt.title("Three-Coin Payout: Simulated vs. Exact")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(PLOTS_DIR / "hw03_plot.png", dpi=150)

    # Even one ticker comes back with (Price, Ticker) MultiIndex columns --> ["Close"]["SPY"] for a Series
    data = yf.download("SPY", period="10y", auto_adjust=True)
    spy_closes = data["Close"]["SPY"].dropna() # Series, shape (~2514,), DatetimeIndex
    spy_returns = spy_closes.pct_change().dropna() # first day has no yesterday --> one fewer row

    first_date = spy_closes.index[0].strftime("%a, %m/%d/%Y")
    last_date = spy_closes.index[-1].strftime("%a, %m/%d/%Y")

    print(f"\nQ4  SPY, {first_date} -> {last_date}")
    print(f"    Trading days: {len(spy_closes):,} closes --> {len(spy_returns):,} daily returns")
    print(f"    Mean daily return: {spy_returns.mean():.4%}") # :% multiplies by 100 for us

    # Counts behind each probability, so the reader can see the sample size
    total_days = len(spy_returns)
    down_days = int((spy_returns < 0).sum())
    down_today, down_then_down = count_next_day_downs(spy_returns, threshold=0.0)

    print("\nQ5  Does a down day make tomorrow more likely to be down?")
    print(f"    P(down)              = {p_down(spy_returns):.4f}  ({down_days:,} down / {total_days:,} days)")
    print(f"    P(down | down today) = {p_down_given_down(spy_returns):.4f}  "
          f"({down_then_down:,} followed by down / {down_today:,} down days)")

    big_drops, big_drop_then_down = count_next_day_downs(spy_returns, threshold=-0.02)

    print("\nQ6  Conditioning on a big drop instead")
    print(f"    P(down | today < -2%) = {p_down_given_big_drop(spy_returns):.4f}  "
          f"({big_drop_then_down} followed by down / {big_drops} days)")
    print(f"    Only {big_drops} of {total_days:,} days ({big_drops / total_days:.1%}) fell more than 2%")

    epv = expected_present_value([10, 10, 10], 0.10, 0.5)
    print(f"\nQ7  expected_present_value([10, 10, 10], 0.10, 0.5) = {epv:.2f}")
    print(f"    Sanity check: survival 1.0 = {expected_present_value([10, 15, 20], 0.10, 1.0):.2f} "
          f"(matches present_value {present_value([10, 15, 20], 0.10):.2f})")

    # Q8 support: direction looks independent, but does the SIZE of moves bunch up?
    r = spy_returns.to_numpy() # plain array, shape (2513,)
    big_drop_today = r[:-1] < -0.02 # same today/tomorrow alignment as Q5
    big_move_tomorrow = np.abs(r[1:]) > 0.02 # up or down, either counts
    big_move_after_drop = big_move_tomorrow[big_drop_today] # bool mask as an index keeps only the Trues

    print("\nQ8  Support: do big moves follow big drops?")
    print(f"    P(|move| > 2%)               = {big_move_tomorrow.sum() / big_move_tomorrow.size:.4f}  "
          f"({big_move_tomorrow.sum()} / {big_move_tomorrow.size:,} days)")
    print(f"    P(|move| > 2% | today < -2%) = {big_move_after_drop.sum() / big_move_after_drop.size:.4f}  "
          f"({big_move_after_drop.sum()} / {big_move_after_drop.size} days)")

    # groupby(index.year) buckets the rows by calendar year, .size() counts each bucket
    big_drop_days = spy_returns[spy_returns < -0.02]
    drops_by_year = big_drop_days.groupby(big_drop_days.index.year).size() # Series indexed by year
    print(f"    Days below -2% by year: {drops_by_year.to_dict()}")

    plt.show()


if __name__ == "__main__":
    main()
