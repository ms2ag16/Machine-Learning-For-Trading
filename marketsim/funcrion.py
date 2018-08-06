
import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):

    daily_ret = (port_val / port_val.shift(1)) - 1
    cr = (port_val[-1] / port_val[0]) - 1
    adr = daily_ret.mean()
    sddr = daily_ret.std()
    k = np.sqrt(samples_per_year)
    sr = k * np.mean(adr - daily_rf) /sddr
    return cr, adr, sddr, sr


    df = pd.read_csv(orders_file, index_col='Date', parse_dates=True,na_values=['nan'])
    #df = df.sort_index()


    # get all the symbols
    symbols = []
    for i, row in df.iterrows():
        if row['Symbol'] not in symbols:
            symbols.append(row['Symbol'])

    # calculate start and end dates for orders (date of first order and ate of last order)
    start_date = df.index[0]
    end_date = df.index[-1]
    # get price for each symbol for the start_end dates
    symbol_prices = get_data(symbols, pd.date_range(start_date, end_date))
    #symbol_prices = symbol_prices.sort_index()
    #symbol_prices = symbol_prices.fillna(method='ffill', inplace=False)
    #symbol_prices = symbol_prices.fillna(method='bfill', inplace=False)


    # get the df_prices
    for symbol in symbols:
        symbol_prices[symbol + ' Shares'] = pd.Series(0, index=symbol_prices.index)
        symbol_prices['Port_val'] = pd.Series(start_val, index=symbol_prices.index)
        symbol_prices['Cash'] = pd.Series(start_val, index=symbol_prices.index)

    for i, row in df.iterrows():
        symbol = row['Symbol']
        if row['Order'] == 'BUY':
            symbol_prices.ix[i:, symbol + ' Shares'] = symbol_prices.ix[i:, symbol + ' Shares'] + row['Shares']
            symbol_prices.ix[i:, 'Cash'] -= symbol_prices.ix[i, symbol] * row['Shares']
        if row['Order'] == 'SELL':
            symbol_prices.ix[i:, symbol + ' Shares'] = symbol_prices.ix[i:, symbol + ' Shares'] - row['Shares']
            symbol_prices.ix[i:, 'Cash'] += symbol_prices.ix[i, symbol] * row['Shares']

    for i, row in symbol_prices.iterrows():
        shares_val = 0
        for symbol in symbols:
            shares_val += symbol_prices.ix[i, symbol + ' Shares'] * row[symbol]
        symbol_prices.ix[i, 'Port Val'] = symbol_prices.ix[i, 'Cash'] + shares_val


    return symbol_prices.ix[:, 'Port Val']