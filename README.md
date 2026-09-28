# Algory Quantitative Investments (QI) — Fall 2026

**Leo Winston**  

---

## Assignments

### [Homework 1: Getting the Tools Working](homework/hw01.py)
- **Objective:** Environment verification, market data retrieval with `yfinance`, and comparative asset performance.
- **Analysis:** Downloaded 1 year of daily closing prices for Nintendo (`NTDOY`) against the market benchmark (`SPY`). Evaluated 1-year total returns, computed annualized return volatility ($\sigma \times \sqrt{252}$), mapped rebased price trajectories (base = 100), and identified maximum single-day price moves.
### [Homework 2: Rates, Ratios, and Discounting](homework/hw02.py)
- **Objective:** Build a reusable present value function, price a bond with it, and estimate beta from real market data.
- **Analysis:** Wrote `present_value` and `bond_price`, then priced a 10-year 4% coupon bond at 2%, 4%, and 4.96% market rates and plotted price against rate from 0% to 10% to show convexity. Downloaded 5 years of daily closes for Caterpillar (`CAT`), Moderna (`MRNA`), and Cloudflare (`NET`) against `SPY`, and computed annualized return, annualized volatility, and beta ($\beta = \mathrm{Cov}(r_s, r_m) / \mathrm{Var}(r_m)$, on returns, not prices). Ranked the tickers by volatility and by beta to separate market risk from company-specific risk.
- **Note:** `present_value` is kept on purpose. Homework 3 extends it to cash flows that may not arrive.
