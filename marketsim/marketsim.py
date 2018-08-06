"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data


def author():
    return 'msun85'


def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):
    daily_ret = (port_val / port_val.shift(1)) - 1
    cr = (port_val[-1] / port_val[0]) - 1
    adr = daily_ret.mean()
    sddr = daily_ret.std()
    k = np.sqrt(samples_per_year)
    sr = k * np.mean(adr - daily_rf) / sddr
    return cr, adr, sddr, sr


def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here
    df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    df = df.sort_index()

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
    symbol_prices = symbol_prices.sort_index()
    symbol_prices = symbol_prices.fillna(method='ffill', inplace=False)
    symbol_prices = symbol_prices.fillna(method='bfill', inplace=False)

    # get the df_prices
    for symbol in symbols:
        symbol_prices[symbol + ' Shares'] = pd.Series(0, index=symbol_prices.index)
        symbol_prices['Port_val'] = pd.Series(start_val, index=symbol_prices.index)
        symbol_prices['Cash'] = pd.Series(start_val, index=symbol_prices.index)

    for i, row in df.iterrows():
        symbol = row['Symbol']
        if row['Order'] == 'BUY':
            symbol_prices.ix[i:, symbol + ' Shares'] = symbol_prices.ix[i:, symbol + ' Shares'] + row['Shares']
            symbol_prices.ix[i:, 'Cash'] = symbol_prices.ix[i:, 'Cash'] - \
                                           (symbol_prices.ix[i, symbol] * (1 + impact) * row['Shares']) - commission
        if row['Order'] == 'SELL':
            symbol_prices.ix[i:, symbol + ' Shares'] = symbol_prices.ix[i:, symbol + ' Shares'] - row['Shares']
            symbol_prices.ix[i:, 'Cash'] = symbol_prices.ix[i:, 'Cash'] + \
                                           symbol_prices.ix[i, symbol] * (1 - impact) * row['Shares'] - commission
    for i, row in symbol_prices.iterrows():
        shares_val = 0
        for symbol in symbols:
            shares_val += symbol_prices.ix[i, symbol + ' Shares'] * row[symbol]
        symbol_prices.ix[i, 'Port Val'] = symbol_prices.ix[i, 'Cash'] + shares_val

    portval = symbol_prices.ix[:, 'Port Val']

    return portval


# In the template, instead of computing the value of the portfolio, we just
# read in the value of IBM over 6 months
# start_date = dt.datetime(2008,1,1)
# end_date = dt.datetime(2008,6,1)
# portvals = get_data(['IBM'], pd.date_range(start_date, end_date))
# portvals = portvals[['IBM']]  # remove SPY

# return portvals

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    print portvals

    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    # start_date = dt.datetime(2008,1,1)
    # end_date = dt.datetime(2008,6,1)
    start_date = dt.datetime(2011, 1, 10)
    end_date = dt.datetime(2011, 12, 20)
    # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

    # cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]
    spyvals = get_data(['$SPX'], dates=pd.date_range(start_date, end_date))
    spyvals = spyvals[spyvals.columns[1]]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = get_portfolio_stats(spyvals)

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])


if __name__ == "__main__":
    test_code()