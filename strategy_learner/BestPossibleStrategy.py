import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np

from util import get_data, plot_data
from marketsimcode import compute_portvals

def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):

    daily_ret = (port_val / port_val.shift(1)) - 1
    cr = (port_val[-1] / port_val[0]) - 1
    adr = daily_ret.mean()
    sddr = daily_ret.std()
    k = np.sqrt(samples_per_year)
    sr = k * np.mean(adr - daily_rf) /sddr
    return cr, adr, sddr, sr


class BestPossibleStrategy(object):

    def __init__(self,symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31),sv=100000):
        self.symbol=symbol
        self.sd=sd
        self.ed=ed
        self.sv=sv

    def author(self):
        return 'msun85'


    def testPolicy(self,symbol,sd,ed,sv):
        symbols=[]
        symbols.append(symbol)
        dates=pd.date_range(sd,ed)
        price=get_data(symbols,dates)
        order_list = []
        curr_holding = 0

        for i in range(len(price)-1):
            if price.ix[i, symbol] == price.ix[i + 1, symbol]:
                order_list.append([price.index[i].date(),symbol,"NOTHING",0])
            # price will decrease
            elif price.ix[i, symbol] > price.ix[i + 1, symbol]:
                if curr_holding == 0:
                    curr_holding = -1000
                    order_list.append([price.index[i].date(), symbol, "SELL", 1000])
                elif curr_holding > 0:
                    curr_holding = -1000
                    order_list.append([price.index[i].date(), symbol, "SELL", 1000])
                    order_list.append([price.index[i].date(), symbol, "SELL", 1000])
                else:
                    order_list.append([price.index[i].date(), symbol, "NOTHING", 0])
            else:  # price will increase
                if curr_holding == 0:
                    curr_holding = 1000
                    order_list.append([price.index[i].date(), symbol, "BUY", 1000])
                elif curr_holding > 0:
                    order_list.append([price.index[i].date(), symbol, "NOTHING", 0])
                else:
                    curr_holding = 1000
                    order_list.append([price.index[i].date(), symbol, "BUY", 1000])
                    order_list.append([price.index[i].date(), symbol, "BUY", 1000])

        # add the last day of trade, since nothing to compare, so do nothing
        order_list.append([price.index[-1].date(),symbol,"NOTHING",0])
        trades_df = pd.DataFrame(order_list, columns=["Date", "Symbol", "Order", "Shares"])
        trades_df.set_index('Date',inplace=True)

        return trades_df


    def get_bps_portfolio(self,symbol,sd,ed,sv,commission=0,impact=0):
        dates = pd.date_range(sd, ed)
        symbols = []
        symbols.append(symbol)
        bps_prices=get_data(symbols,dates)
        bps_prices.sort_index()

        bps_trades=self.testPolicy(symbol,sd,ed,sv)
        portval_bps=compute_portvals(bps_trades,start_val=sv,commission=0,impact=0)
        #normalize the portval_bps
        normed_portval_bps=portval_bps/portval_bps.ix[0]
        cr_bps, adr_bps, sdr_bps, sr_bps = get_portfolio_stats(normed_portval_bps)
        print
        print "Cumulative Return of {}: {}".format("Best possible portfolio", cr_bps)
        print "Standard Deviation of daily return of {}: {}".format("Best possible portfolio", sdr_bps)
        print "Mean Daily Return of {}: {}".format("Best possible portfolio", adr_bps)
        print "Portval of {}: {}".format("Best possible portfolio",portval_bps[-1])
        return normed_portval_bps

    def get_benchmark_portfolio(self,symbol,sd,ed,sv,commission,impact):
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
    # Part 2
    bps= BestPossibleStrategy()
    trades_df=bps.testPolicy(symbol="JPM",  sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31),sv=100000)

    portval_bps=bps.get_bps_portfolio(symbol="JPM",  sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31),sv=100000,
                                      commission=0, impact=0)
    portval_benchmark=bps.get_benchmark_portfolio(symbol="JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31),sv=100000,
                                                  commission=0,impact=0)
    ax=portval_bps.plot(title="Figure 6. Benchmark vs. Best Possible Strategy (for JPM)", fontsize=12, color="black",label="Benchmark")
    portval_benchmark.plot(ax=ax,color="blue",label="BestPossibleStrategy")
    ax.set_xlabel('Dates')
    ax.set_ylabel('Normalized Value')
    plt.grid(True)
    plt.legend(loc=0)
    plt.savefig("Fig6.png")
    plt.show()




