# Mi Sun  msun85

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

from util import get_data, plot_data
from marketsimcode import compute_portvals
from indicators import get_Price_SMA_Ratio, get_Bollinger_Bands_ratio,get_MFI

def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):

    daily_ret = (port_val / port_val.shift(1)) - 1
    cr = (port_val[-1] / port_val[0]) - 1
    adr = daily_ret.mean()
    sddr = daily_ret.std()
    k = np.sqrt(samples_per_year)
    sr = k * np.mean(adr - daily_rf) /sddr
    return cr, adr, sddr, sr



class MannualStrategy(object):

    def __init__(self,symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31),sv=100000):
        self.symbol=symbol
        self.sd=sd
        self.ed=ed
        self.sv=sv

    def author(self):
        return 'msun85'

    def testPolicy(self,symbol,sd,ed,sv):

        symbols = []
        symbols.append(symbol)
        dates = pd.date_range(sd, ed)
        price = get_data(symbols, dates)

        orders = []
        holdings = {sym: 0 for sym in symbols}
        lookback=14

        # get the indicators values
        psr=get_Price_SMA_Ratio(symbols,dates,lookback=14)
        bbp=get_Bollinger_Bands_ratio(symbols,dates,lookback=14)
        mfi=get_MFI(symbols,dates)


        for day in range(price.shape[0]):
            for sym in symbols:
                if sym == "SPY":
                    continue
                # STOCK MAY BE OVERBOUGHT, INDEX DOES NOT APPEAR TO BE OVERSOLD
                if (psr.ix[day, sym] < 1.05) and (bbp.ix[day, sym]<=0.5) and (mfi.ix[day,sym]<0.4):
                    if holdings[sym] < 1000:
                        holdings[sym] = holdings[sym] + 1000
                        orders.append([price.index[day].date(), sym, 'BUY', 1000])
                    else:
                        holdings[sym]=holdings[sym]
                        orders.append([price.index[day].date(), sym, 'NOTHING', 0])
                # Stock may be overbought, index does not appear to be overbought
                elif (psr.ix[day, sym] > 0.95) and (bbp.ix[day, sym] > 0.3) and (mfi.ix[day,sym]>1):
                    if holdings[sym] > -1000:
                        holdings[sym] = holdings[sym] - 1000
                        orders.append([price.index[day].date(), sym, 'SELL', 1000])
                    else:
                        holdings[sym] = holdings[sym]
                        orders.append([price.index[day].date(), sym, 'NOTHING', 0])
                elif (psr.ix[day, sym] >=1) and (psr.ix[day-1,sym]<1) and (holdings[sym]>0):
                    holdings[sym]=0
                    orders.append([price.index[day].date(), sym, 'SELL', 1000])
                elif (psr.ix[day, sym] <=1) and (psr.ix[day-1,sym]>1) and (holdings[sym]<0):
                    holdings[sym] = 0
                    orders.append([price.index[day].date(), sym, 'BUY', 1000])
                else:
                    orders.append([price.index[day].date(), sym, 'NOTHING', 0])


        trades_df = pd.DataFrame(orders, columns=["Date", "Symbol", "Order", "Shares"])
        trades_df.set_index('Date', inplace=True)


        return trades_df

    def get_manualrule_portfolio(self,symbol,sd,ed,sv,commission=9.95,impact=0.005):
        dates = pd.date_range(sd, ed)
        symbols = []
        symbols.append(symbol)


        mbs_trades = self.testPolicy(symbol, sd, ed, sv)

        portval_mbs = compute_portvals(mbs_trades, start_val=sv, commission=commission, impact=impact)
        # normalize the portval_bps
        normed_portval_mbs = portval_mbs / portval_mbs.ix[0]
        cr_mbs, adr_mbs, sdr_mbs, sr_mbs = get_portfolio_stats(normed_portval_mbs)
        print
        print "Cumulative Return of {}: {}".format("Manual Rule-based Strategy", cr_mbs)
        print "Standard Deviation of daily return of {}: {}".format("Manual Rule-based Strategy", sdr_mbs)
        print "Mean Daily Return of {}: {}".format("Manual Rule-based Strategy", adr_mbs)
        print "Portval of {}: {}".format("Manual Rule-based Strategy", portval_mbs[-1])
        return normed_portval_mbs

    def get_benchmark_portfolio(self,symbol,sd,ed,sv,commission=9.95,impact=0.005):
        dates = pd.date_range(sd, ed)
        symbols=[]
        symbols.append(symbol)
        # get the benchmark prices, important, is to get the trade dates
        benchmark_prices = get_data(symbols, dates)
        orders = []
        for i in range(len(benchmark_prices)):
            if i==0:
                orders.append([benchmark_prices.index[0].date(), symbol, "BUY", 1000])
            else:
                orders.append([benchmark_prices.index[i].date(), symbol, "NOTHING",0])

        benchmark_trades = pd.DataFrame(orders, columns=["Date", "Symbol", "Order", "Shares"])

        benchmark_trades.set_index('Date', inplace=True)
        portval_benchmark = compute_portvals(benchmark_trades,start_val=sv,commission=commission,impact=impact)

        # normalize the portval_benchmark
        normed_portval_benchmark = portval_benchmark/ portval_benchmark.ix[0]
        cr_benchmark, adr_benchmark, sdr_benchmark, sr_benchmark = get_portfolio_stats(portval_benchmark)
        print
        print "Cumulative Return of {}: {}".format("Benchmark", cr_benchmark)
        print "Standard Deviation of daily return of {}: {}".format("Benchmark", sdr_benchmark)
        print "Mean Daily Return of {}: {}".format("Benchmark", adr_benchmark)
        print "Portval of {}: {}".format("Benchmark",portval_benchmark[-1])
        return normed_portval_benchmark

if __name__ == "__main__":
    # Part 3
    mbs = MannualStrategy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    trades_df = mbs.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    portval_mbs = mbs.get_manualrule_portfolio(symbol="JPM", sv=100000, sd=dt.datetime(2008, 1, 1),
                                               ed=dt.datetime(2009, 12, 31), commission=9.95, impact=0.005)
    portval_benchmark = mbs.get_benchmark_portfolio(symbol="JPM", sd=dt.datetime(2008, 1, 1),
                                                    ed=dt.datetime(2009, 12, 31), sv=100000, commission=9.95,
                                                    impact=0.005)
    ax = portval_mbs.plot(title="Figure 7. Benchmark vs. Manual Rule-Based Strategy (JPM)", fontsize=12, color="black",
                          label="Manual Rule-Based")
    portval_benchmark.plot(ax=ax, color="blue", label="Benchmark")
    ax.set_ylabel('Normalized Value')
    ax.set_xlabel('Dates')
    ymin,ymax=ax.get_ylim()
    entries=[]
    entries2=[]

    for i in range(0,len(trades_df),2):
        if trades_df.ix[i,"Order"]=="SELL":
            entries.append(trades_df.index[i])
        elif trades_df.ix[i,"Order"]=="BUY":
            entries2.append(trades_df.index[i])

    for day in entries:
        ax.axvline(x=day,color="r")

    for day in entries2:
        ax.axvline(x=day, color="g")

    plt.grid(True)


    plt.legend(loc=0)
    plt.savefig("Fig7.png")
    plt.show()






