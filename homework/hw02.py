"""Homework 2: Rates, Ratios, and Discounting — Algory QI Education, Fall 2026

Present value, bond pricing, and beta for CAT, MRNA, NET vs. SPY.
Check with `python checks/check_hw02.py homework/hw02.py` from the repo root.
"""
import pathlib

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PLOTS_DIR = pathlib.Path(__file__).parent.parent / "plots"


# ---------------------------------------------------------------- Q1
def present_value(cash_flows, rate):
    """Present value of a list of cash flows, the first arriving in one year.
    present_value([10, 15, 20], 0.10) -> 36.51
    """
    cfs = np.asarray(cash_flows, dtype=float)
    years = np.arange(1, len(cfs) + 1) # [1, 4) --> years: 1,2,3
    discount_factors = (1.0 + rate) ** years # df[i] = r^{i + 1}
    
    return float(np.sum(cfs / discount_factors))


# ---------------------------------------------------------------- Q2
def bond_price(face, coupon_rate, years, market_rate):
    """Price of a bond paying an annual coupon and repaying face at maturity.

    The final year pays the coupon AND the face value. That is the usual bug.
    bond_price(1000, 0.04, 10, 0.04) -> exactly 1000.0
    """
    coupon = face * coupon_rate
    cash_flows = np.full(years, coupon, dtype=float)
    cash_flows[-1] += face  # Final year pays coupon + face val

    return float(present_value(cash_flows, market_rate))


# ---------------------------------------------------------------- Q4/Q5
def annualised_return(prices):
    """Annualised return from a price series, using 252 trading days."""
    p = np.asarray(prices, dtype=float)
    total_return = p[-1] / p[0] # P_end / P_start
    trading_days = len(p) - 1 # N prices --> N - 1 return periods
    
    return float(total_return ** (252.0 / trading_days) - 1.0) # compound to 252 trading days


def annualised_volatility(prices):
    """Annualised standard deviation of daily returns."""
    p = np.asarray(prices, dtype=float)
    daily_returns = np.diff(p) / p[:-1] # (P_t - P_{t-1}) / P_{t-1}
    daily_vol = np.std(daily_returns, ddof=1) # sample std deviation
    
    return float(daily_vol * np.sqrt(252.0)) # scale daily -> annual via sqrt(252)


def beta(stock_prices, market_prices):
    """Beta of a stock against the market.

    Covariance of the two RETURN series divided by the variance of the market's.
    Computing this on prices instead of returns is a common and silent error.
    beta(spy, spy) -> 1.0
    """
    s = np.asarray(stock_prices, dtype=float)
    m = np.asarray(market_prices, dtype=float)

    r_stock = np.diff(s) / s[:-1] # daily stock returns
    r_market = np.diff(m) / m[:-1] # daily market returns

    cov_matrix = np.cov(r_stock, r_market) # [[var_s, cov_sm], [cov_ms, var_m]]
    return float(cov_matrix[0, 1] / cov_matrix[1, 1]) # cov(s, m) / var(m)
    
    


# ---------------------------------------------------------------- your answers
def main():
    """Everything the assignment asks you to print goes here."""
    print(f"Q1  present_value([10, 15, 20], 0.10) = {present_value([10, 15, 20], 0.10):.2f}")
    print(f"Q2  bond_price(1000, 0.04, 10, 0.04) = {bond_price(1000, 0.04, 10, 0.04):.2f}") 
    
    rates = [0.02, 0.04, 0.0496]
    for r in rates:
        price = bond_price(1000, 0.04, 10, r)
        print(f"Q3  bond_price(1000, 0.04, 10, {r:.4f}) = ${price:.2f}")

    # Plot price vs rate across [0%, 10%]
    rate_grid = np.linspace(0.0, 0.10, 100) # 0% to 10% market rate grid
    price_grid = [bond_price(1000, 0.04, 10, r) for r in rate_grid]

    plt.figure()
    plt.plot(rate_grid * 100, price_grid, label="10Y 4% Coupon Bond")
    plt.scatter([r * 100 for r in rates], [bond_price(1000, 0.04, 10, r) for r in rates], color="red", zorder=3)
    plt.xlabel("Market Rate (%)")
    plt.ylabel("Price ($)")
    plt.title("Bond Price vs. Market Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(PLOTS_DIR / "hw02_plot.png", dpi=150)

    # CAT (Caterpillar) = Industrials, MRNA (Moderna) = Healthcare, NET (Cloudflare) = Tech
    tickers = ["CAT", "MRNA", "NET"]
    data = yf.download(tickers + ["SPY"], period="5y", auto_adjust=True)
    closes = data["Close"].dropna() # Shape is (~1255, 4), rows aligned on dates all 4 traded

    print("\nQ4  Trading days (5y)")
    trading_days = closes.count() # non-NaN rows per column --> Series indexed by ticker
    print(trading_days.to_string(header=False))

    # One row per ticker, columns: return, volatility, beta vs. SPY
    market_prices = closes["SPY"]
    stats = {
        ticker: {
            "Return": annualised_return(closes[ticker]),
            "Volatility": annualised_volatility(closes[ticker]),
            "Beta": beta(closes[ticker], market_prices),
        }
        for ticker in tickers
    }
    stats_table = pd.DataFrame(stats).T # dict of dicts --> tickers as columns, .T flips to rows

    print("\nQ5  Annualised stats vs. SPY")
    print(stats_table.round(3).to_string())
    print(f"Sanity check: beta(SPY, SPY) = {beta(market_prices, market_prices):.2f}")

    # rank(ascending=False) --> 1 = riskiest
    vol_rank = stats_table["Volatility"].rank(ascending=False).astype(int)
    beta_rank = stats_table["Beta"].rank(ascending=False).astype(int)
    rankings = pd.DataFrame({"Vol Rank": vol_rank, "Beta Rank": beta_rank})

    print("\nQ6  Riskiest to safest (1 = riskiest)")
    print(rankings.sort_values("Vol Rank").to_string())

    # Show plot after all print statements
    plt.show()


if __name__ == "__main__":
    main()