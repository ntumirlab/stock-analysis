from finlab import data
import pandas as pd

start_date = '2022-01-01'
end_date = '2024-06-28'

# Get raw closing prices for TSMC
close = data.get('price:收盤價')['2330']

# Get adjusted closing prices for TSMC
adj_close = data.get('etl:adj_close')['2330']

# Combine both into a single DataFrame
tsmc_prices = pd.DataFrame({'Close': close, 'Adj Close': adj_close})
tsmc_prices = tsmc_prices[(adj_close.index >= start_date) & (adj_close.index <= end_date)]


# Display the DataFrame
print(tsmc_prices)
