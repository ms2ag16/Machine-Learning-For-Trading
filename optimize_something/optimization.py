"""MC1-P2: Optimize a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as sco


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality

def get_portfolio_value(prices, allocs, start_val=1):

    normed = prices/prices.ix[0,:]
    alloced = normed * allocs
    pos_vals = alloced * start_val
    port_val = pos_vals.sum(axis=1)
    return port_val



def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):

    daily_ret = (port_val / port_val.shift(1)) - 1
    cr = (port_val[-1] / port_val[0]) - 1
    adr = daily_ret.mean()
    sddr = daily_ret.std()
    k = np.sqrt(samples_per_year)
    sr = k * np.mean(adr - daily_rf) /sddr
    return cr, adr, sddr, sr



def find_optimal_allocations(prices):
    num_columns = len(prices.columns)
    guess = num_columns * [1. / num_columns, ]
    cons = ({'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)})
    bnds = tuple((0, 1) for x in range(num_columns))
    opts = sco.minimize(min_func_variance, guess, args=(prices,), method='SLSQP', bounds=bnds, constraints=cons)
    allocs = opts['x']
    return allocs

def min_func_variance(allocs, prices):
    cr, adr, sddr, sr = get_portfolio_stats(get_portfolio_value(prices, allocs, 1))
    return sddr




def optimize_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), \
                       syms=['GOOG', 'AAPL', 'GLD', 'XOM'], gen_plot=False):
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)


    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    ## allocs = np.asarray([0.2, 0.2, 0.3, 0.3]) # add code here to find the allocations

    allocs = find_optimal_allocations(prices)

    allocs = allocs / np.sum(allocs)  # normalize allocations, if they don't sum to 1.0

    # Get daily portfolio value
    ##port_val = prices_SPY # add code here to compute daily portfolio values
    port_val = get_portfolio_value(prices, allocs)

    ##cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats
    cr, adr, sddr, sr =get_portfolio_stats(port_val)



    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        normed_SPY = prices_SPY / prices_SPY.ix[0, :]
        df_temp = pd.concat([port_val, normed_SPY], keys=['Portfolio', 'SPY'], axis=1)
        ax = df_temp.plot(title="Daily Portfolio Value and SPY")
        ax.set_ylabel('Prices')
        ax.set_xlabel('Date')
        plt.grid(True)
        plt.show()
        pass

    return allocs, cr, adr, sddr, sr


def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols = ['IBM','X','GLD']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date, ed=end_date, \
                                                        syms=symbols, \
                                                        gen_plot=True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr



if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()