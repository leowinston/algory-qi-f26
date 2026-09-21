import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# I picked Nintendo Stock (NTDOY) cause it was the stock I wanted my dad to buy when I was in middle school

fav_ticker = "NTDOY"
data = yf.download([fav_ticker, "SPY"], period = "1y") 
closes = data["Close"].dropna() # Shape is (251, 2)

# closes.columns --> labels of columns (indexable, Strings)
# closes.index --> labels of rows (indexable, pd.Timestamp objects)

num_trading_days = len(closes)
first_date = closes.index[0].strftime("%a, %m/%d/%Y") # format pd.Timestamp: Day of Week, MM/DD/YYYY
last_date = closes.index[-1].strftime("%a, %m/%d/%Y") 

print(f"\nTickers: {", ".join(closes.columns)}")
print(f"Trading days: {num_trading_days}")
print(f"First date: {first_date}")
print(f"Last date: {last_date}")

# to_string(header=False) --> hide Series name and dtype 

last_prices = closes.iloc[-1]
first_prices = closes.iloc[0]
yearly_returns = (last_prices - first_prices) / first_prices

print(f"\nLast closing prices:\n{last_prices.to_string(header=False)}")
print(f"\nYearly Returns:\n{yearly_returns.to_string(header=False)}")

# Daily returns = (today's price / yesterday's price) - 1 --> pandas pct_change()
# Annualized Volatility = Standard Deviation of daily returns * \sqrt(252)

daily_returns = closes.pct_change().dropna() # No daily returns on row @ index 0
annualized_vol = daily_returns.std(axis=0, ddof=1) * (252 ** 0.5)

print(f"\nAnnualized Volatility:\n{annualized_vol.to_string(header=False)}")

# Rebase, starting at $100 <-> value of $100 of each stock over time
rebased_prices = (closes / closes.iloc[0]) * 100

plt.plot(rebased_prices)
plt.xlabel("Date")
plt.ylabel("Rebased Price (Base = 100)")
plt.legend(rebased_prices.columns) # Legend is labels of columns (the tickers)

# Biggest single-day move can be gain or loss
biggest_dates = abs(daily_returns).idxmax(axis=0)
biggest_moves = daily_returns.loc[biggest_dates]

fav_date = biggest_dates[fav_ticker]
fav_move = biggest_moves.loc[fav_date, fav_ticker]

print(f"\nBiggest single-day move for {fav_ticker}: {fav_date.strftime('%a, %m/%d/%Y')} ({round(fav_move * 100, 2)}%)")

# Show plot after all print statements

plt.show()
