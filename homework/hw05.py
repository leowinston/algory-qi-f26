"""Homework 5 starter — Algory QI Education, Fall 2026

Keep these function names: check_hw05.py looks for them.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT / "plots"


def intrinsic(kind: str, strike: ArrayLike, spot: ArrayLike) -> float | np.ndarray:
    """What the option is worth if exercised right now. Never negative.

    kind is "call" or "put".
    intrinsic("call", 45, 48) -> 3.0        intrinsic("call", 55, 48) -> 0.0
    """
    strike = np.asarray(strike, dtype=float)
    spot = np.asarray(spot, dtype=float)
    if kind == "call":
        return np.maximum(spot - strike, 0)
    if kind == "put":
        return np.maximum(strike - spot, 0)
    raise ValueError("kind must be call or put")


def extrinsic(kind: str, strike: ArrayLike, spot: ArrayLike,
              premium: ArrayLike) -> float | np.ndarray:
    """What you are paying for time and uncertainty: premium minus intrinsic."""
    return np.asarray(premium, dtype=float) - intrinsic(kind, strike, spot)


def payoff(kind: str, strike: ArrayLike, premium: ArrayLike,
           spot: ArrayLike) -> float | np.ndarray:
    """Profit per share at expiry.

    kind is one of: long_call, long_put, short_call, short_put.
    Check the signs before going further:
        payoff("long_call", 100, 6, 80)   -> -6.0
        payoff("short_call", 100, 6, 120) -> -14.0
    """
    premium = np.asarray(premium, dtype=float)
    if kind == "long_call":
        return intrinsic("call", strike, spot) - premium
    if kind == "long_put":
        return intrinsic("put", strike, spot) - premium
    if kind == "short_call":
        return premium - intrinsic("call", strike, spot)
    if kind == "short_put":
        return premium - intrinsic("put", strike, spot)
    raise ValueError("kind must be long_call, long_put, short_call, or short_put")


def call_values(calls: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Keep quoted calls and split their mid prices into intrinsic and extrinsic."""
    values = calls.loc[calls["bid"] > 0].copy()  # copy so the original chain stays intact
    values["mid"] = (values["bid"] + values["ask"]) / 2
    values["intrinsic"] = intrinsic("call", values["strike"], spot)
    values["extrinsic"] = extrinsic("call", values["strike"], spot, values["mid"])
    values["distance"] = abs(values["strike"] - spot)
    return values.sort_values("strike")


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)

    # Q1: each column uses the whole spot array at once
    kinds = ["long_call", "long_put", "short_call", "short_put"]
    spots = np.array([80, 100, 120])
    profits = pd.DataFrame(index=spots)
    profits.index.name = "Spot at expiry"
    for kind in kinds:
        profits[kind] = payoff(kind, 100, 6, spots)
    print("Q1  Profit per share, strike 100, premium 6")
    print(profits.to_string(float_format="%.2f"))

    # Q2: loop over the four panels, not over prices
    spot_grid = np.linspace(60, 140, 401)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for kind, ax in zip(kinds, axes.flat):
        ax.plot(spot_grid, payoff(kind, 100, 6, spot_grid))
        ax.axvline(100, color="gray", linestyle="--", label="Strike")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(kind.replace("_", " ").title())
        ax.set_xlabel("Spot at expiry ($)")
        ax.set_ylabel("Profit per share ($)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Expiry profits: strike 100, premium 6 (dollars per share)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "hw05_payoffs.png", dpi=150)
    plt.close(fig)

    # Q3: these expiries were chosen for the October 25 assignment
    ticker = "SPY"
    short_expiry = "2026-11-27"
    long_expiry = "2027-06-17"
    t = yf.Ticker(ticker)
    expiries = t.options  # tuple of available expiry date strings
    print(f"\nQ3  {ticker} available expiries: {expiries}")
    if short_expiry not in expiries or long_expiry not in expiries:
        raise ValueError("Choose new expiries: about one month and at least six months out")
    history = t.history(period="5d", auto_adjust=False)
    closes = history["Close"].dropna()
    spot = float(closes.iloc[-1])  # latest unadjusted close; markets are shut on Sunday
    spot_date = closes.index[-1].strftime("%Y-%m-%d")
    short_chain = t.option_chain(short_expiry)
    long_chain = t.option_chain(long_expiry)
    print(f"    Expiry: {short_expiry}; spot (latest close, {spot_date}): ${spot:.2f}")
    print(f"    Calls: {len(short_chain.calls)}; puts: {len(short_chain.puts)}")

    # Q4: nsmallest picks the closest strikes, then sort for a readable table
    columns = ["strike", "mid", "intrinsic", "extrinsic"]
    short_values = call_values(short_chain.calls, spot)
    short_nearest = short_values.nsmallest(10, "distance").sort_values("strike")
    print("\nQ4  Ten nearest call strikes (positive bids only)")
    print(short_nearest[columns].to_string(index=False, float_format="%.2f"))

    # Q5: idxmax gives the row label of the largest extrinsic value
    short_peak = short_values.loc[short_values["extrinsic"].idxmax()]
    print(f"\nQ5  Largest extrinsic at strike ${short_peak['strike']:.2f}; spot ${spot:.2f}")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(short_values["strike"], short_values["extrinsic"], label=short_expiry)
    ax.axvline(spot, color="gray", linestyle="--", label=f"Spot ${spot:.2f}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Strike ($)", ylabel="Extrinsic value per share ($)", title="SPY call time value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "hw05_short_extrinsic.png", dpi=150)
    plt.close(fig)

    # Q6: use the same spot and the same calculation for the longer expiry
    long_values = call_values(long_chain.calls, spot)
    long_nearest = long_values.nsmallest(10, "distance").sort_values("strike")
    long_peak = long_values.loc[long_values["extrinsic"].idxmax()]
    print(f"\nQ6  Longer expiry: {long_expiry}; calls: {len(long_chain.calls)}; puts: {len(long_chain.puts)}")
    print(long_nearest[columns].to_string(index=False, float_format="%.2f"))
    print(f"    Short-expiry maximum: ${short_peak['extrinsic']:.2f} at strike ${short_peak['strike']:.2f}")
    print(f"    Long-expiry maximum: ${long_peak['extrinsic']:.2f} at strike ${long_peak['strike']:.2f}")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(short_values["strike"], short_values["extrinsic"], label=short_expiry)
    ax.plot(long_values["strike"], long_values["extrinsic"], label=long_expiry)
    ax.axvline(spot, color="gray", linestyle="--", label=f"Spot ${spot:.2f}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Strike ($)", ylabel="Extrinsic value per share ($)", title="SPY call time value: two expiries")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "hw05_expiry_comparison.png", dpi=150)
    plt.close(fig)

    # Compare only strikes that actually occur in both chains
    comparison = short_values[["strike", "extrinsic"]].merge(
        long_values[["strike", "extrinsic"]], on="strike", suffixes=("_short", "_long")
    )
    below = comparison["extrinsic_long"] < comparison["extrinsic_short"] - 1e-9
    print(f"\nChecks: longer curve below shorter at {below.sum()} of {len(comparison)} shared strikes")
    for label, values in [("Short", short_values), ("Long", long_values)]:
        matches = np.allclose(values["intrinsic"] + values["extrinsic"], values["mid"])
        print(f"    {label}: intrinsic + extrinsic = mid: {matches}")
        print(f"    {label}: negative extrinsic rows: {(values['extrinsic'] < 0).sum()}")
        print(f"    {label}: lowest-strike extrinsic: ${values.iloc[0]['extrinsic']:.2f}")
        assert matches


if __name__ == "__main__":
    main()

