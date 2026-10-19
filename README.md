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
### [Homework 3: Probability, and Testing Independence](homework/hw03.py)
- **Objective:** Estimate probabilities by simulation, test whether daily returns are independent by counting, and extend present value to cash flows that may not arrive.
- **Analysis:** Simulated the two-dice Bayes problem and the three-coin payout (seed 0, 100,000 trials), and plotted the coin estimate converging to 1.5 from 100 to 100,000 trials. Downloaded 10 years of daily `SPY` closes (2,514 trading days) and counted P(down), P(down | down), and P(down | down > 2%) directly from the data. Daily direction looks independent, but big moves cluster. Built `expected_present_value` on top of HW2's `present_value` with a yearly survival probability ($\sum_t p^{t-1} \, CF_t / (1+r)^t$).
### [Homework 4: Build and Honestly Test a Strategy](homework/hw04.py)
- **Objective:** Build a walk-forward backtester, invent a timing rule for the S&P 500, and judge it once on data it never saw.
- **What it does:** Downloads daily `^SP500TR` closes (total return, dividends included) since 1990 and splits them in time order at the 9/9/2011 close. Training has 5,469 days. Test has the rest, plus 60 warm-up rows so the window is full on the first out-of-sample day. Each day the strategy sees only the last 60 closes, today included, and earns the next day's return if it holds. The backtester reports total return, annualised return (compounded over 252 days), max drawdown, and fraction of days invested.
- **Machinery checks first:** always-hold matches the index to the cent (+432.27% on training), always-cash returns exactly 0%, and a random 70% hold lands in between. `checks/check_hw04.py` passes all 23 checks, including a probe that fails if the strategy ever sees the price it is predicting.
- **Rules tried, all on training (every one is printed):**
  - A: price above a least-squares line through log prices
  - B: top eigenvalue share of lagged return copies
  - C: step aside when 10-day volatility spikes
  - The class 5-day reversal rule was also tested.
- **Result:** Picked C1 (cash when 10-day vol > 1.5× the window's) by annualised return per unit of max drawdown, fixed before the test run. On training it beat the index, +9.47% to +8.10% a year. On the test set (9/12/2011 → 10/16/2026), run once, it **lost**: +14.27% vs. +15.47% a year. It did cut max drawdown from −33.8% to −24.5%. B's eigen share turned out to be indistinguishable from random-walk noise.
- **Not tested:** transaction costs, slippage, and taxes; any index other than the S&P 500; any split date other than 2011; statistical significance of the gaps.
- **Run:** `python homework/hw04.py` (training table, test run, today's signal) and `python checks/check_hw04.py`.
- **Next ideas:**
  - Add a per-trade cost and see whether C1's drawdown benefit survives it
  - Test C1 on other indices (Nasdaq, international) and other split dates
  - Scale exposure inversely with volatility instead of a binary hold/cash switch
  - Bootstrap the training period to see whether C1's training edge is distinguishable from luck
