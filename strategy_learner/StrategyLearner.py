# Mi Sun  msun85

"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import pandas as pd
import util as ut
import indicators as ind
import numpy as np
import QLearner as ql
from marketsimcode import compute_portvals

def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):

    daily_ret = (port_val / port_val.shift(1)) - 1
    cr = (port_val[-1] / port_val[0]) - 1
    adr = daily_ret.mean()
    sddr = daily_ret.std()
    k = np.sqrt(samples_per_year)
    sr = k * np.mean(adr - daily_rf) /sddr
    return cr, adr, sddr, sr


class StrategyLearner(object):
    # constructor
    def __init__(self, verbose=False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.holdlimit=1000
        self.learner=None

    def author(self):
        return 'msun85'

    def dfDiscretize(self, df, steps):
        threshold = range(0,steps-1)
        stepSize = df.shape[0] / steps

        df = df.sort_values()

        for i in range(0, steps - 1):
             threshold[i]=df[i*stepSize]

        indicator_df=pd.DataFrame(0,columns=['dis_value'],index=df.index)
        indicator_df['dis_value']=np.searchsorted(threshold,df)
        return indicator_df

    def execute_action(self, holding, action, ret):
        rewards = 0
        if holding == -1:
            if action <= 1:
                rewards = -ret
            else:
                holding = 1
                rewards = 2 * ret
        elif holding == 0:
            if action == 0:
                holding = -1
                rewards = -ret
            elif action == 2:
                holding = 1
                rewards = ret
        else:
            if action == 0:
                holding = -1
                rewards = -2 * ret
            else:
                rewards = ret

        return holding, rewards

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol="JPM", \
                    sd=dt.datetime(2008, 1, 1), \
                    ed=dt.datetime(2009, 12, 31), \
                    sv=100000):

        # add your code to do learning here
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
        if self.verbose: print prices

        # get indicators
        all_psr=ind.get_Price_SMA_Ratio(symbols=syms,dates=dates,lookback=14)
        all_bbp=ind.get_Bollinger_Bands_ratio(symbols=syms,dates=dates,lookback=14)
        all_mfi=ind.get_MFI(symbols=syms,dates=dates,lookback=14)

        sym_psr=all_psr.ix[:,symbol]
        sym_bbp=all_bbp.ix[:,symbol]
        sym_mfi=all_mfi.ix[:,symbol]

        # get daily returns
        prices=prices.ix[:,symbol]
        dr = prices.copy()
        dr[1:] = (prices[1:] / prices[:-1].values) - 1
        dr[0] = np.nan



        # discretize indicators
        dis_psr=self.dfDiscretize(sym_psr,steps=10)
        dis_bbp=self.dfDiscretize(sym_bbp,steps=10)
        dis_mfi=self.dfDiscretize(sym_mfi,steps=10)

        States = dis_psr * 100 + dis_bbp * 10 + dis_mfi
        States=States.ix[:,'dis_value']


        # initial qlearner
        self.learner=ql.QLearner(num_states=1000,num_actions=3)

        # training learner
        oldprofit = 0
        for iteration in range(100):
            old_holding = 0
            profits = 0
            rewards = 0
            for i in range(dr.shape[0] - 1):
                if (i > 0):
                    profits += prices[i - 1] * old_holding * self.holdlimit * dr[i]
                state = States[i]
                if i == 0:
                    action = self.learner.querysetstate(state)
                else:
                    action = self.learner.query(state, rewards)
                holding, rewards = self.execute_action(old_holding, action, dr[i + 1])

                old_holding = holding
            profits += prices[-2] * old_holding * self.holdlimit * dr[-1]
            if iteration > 10:
                if profits == oldprofit:
                    break
            oldprofit = profits


        # example use with new colname
        volume_all = ut.get_data(syms, dates, colname="Volume")  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols
        volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        if self.verbose: print volume

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol="JPM", \
                   sd=dt.datetime(2010, 1, 1), \
                   ed=dt.datetime(2011, 12, 31), \
                   sv=100000):

        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY

        syms=[symbol]
        prices = prices_all[syms]  # only portfolio symbols
        # get indicators
        all_psr = ind.get_Price_SMA_Ratio(symbols=syms, dates=dates, lookback=14)
        all_bbp = ind.get_Bollinger_Bands_ratio(symbols=syms, dates=dates, lookback=14)
        all_mfi = ind.get_MFI(symbols=syms, dates=dates, lookback=14)

        sym_psr = all_psr.ix[:, symbol]
        sym_bbp = all_bbp.ix[:, symbol]
        sym_mfi = all_mfi.ix[:, symbol]

        # discretize indicators
        dis_psr = self.dfDiscretize(sym_psr, steps=10)
        dis_bbp = self.dfDiscretize(sym_bbp, steps=10)
        dis_mfi = self.dfDiscretize(sym_mfi, steps=10)

        States = dis_psr * 100 + dis_bbp*10+dis_mfi
        States = States.ix[:, 'dis_value']

        # get daily returns
        prices = prices.ix[:, symbol]
        dr = prices.copy()
        dr[1:] = (prices[1:] / prices[:-1].values) - 1
        dr[0] = np.nan
        trades=dr.copy()
        trades[:] = 0

        old_holding = 0
        profits = 0
        for i in range(dr.shape[0] - 1):

            if (i > 0):
                profits += prices[i - 1] * old_holding * self.holdlimit * dr[i]
            state = States[i]

            action = self.learner.querysetstate(state)

            holding, rewards = self.execute_action(old_holding, action, dr[i + 1])
            trades[i] = (holding - old_holding) * self.holdlimit
            old_holding = holding
        profits += prices[-2] * old_holding * self.holdlimit * dr[-1]

        trades = pd.DataFrame(trades, columns=[symbol])


        if self.verbose: print type(trades)  # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all

        return trades

    def get_strategy_learner_portfolio(self, symbol="JPM", sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 12, 31), sv=100000,commission=0.0,impact=0.005):
        dates = pd.date_range(sd, ed)
        symbols = []
        symbols.append(symbol)

        self.addEvidence(symbol="JPM",sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), \
                   sv=100000)
        sbl_trades=self.testPolicy(symbol="JPM",sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), \
                   sv=100000)
        #print sbl_trades
        trades_df = pd.DataFrame(columns=['Date', 'Symbol', 'Order', 'Shares'])
        trades_df['Date'] = sbl_trades.index
        trades_df.set_index('Date', inplace=True)
        trades_df['Symbol']=symbol
        trades_df['Order'] = ["BUY" if x > 0 else "SELL" for x in sbl_trades.values]
        trades_df['Shares']=abs(sbl_trades.values)
       # print trades_df
        portval_sbl = compute_portvals(trades_df, start_val=sv, commission=commission, impact=impact)
       # print portval_sbl

        # normalize the portval_bps
        normed_portval_sbl = portval_sbl / portval_sbl.ix[0]
        cr_sbl, adr_sbl, sdr_sbl, sr_sbl = get_portfolio_stats(normed_portval_sbl)
        print
        print "Cumulative Return of {}: {}".format("QLearner-based  Strategy", cr_sbl)
        print "Standard Deviation of daily return of {}: {}".format("QLearner-based  Strategy", sdr_sbl)
        print "Mean Daily Return of {}: {}".format("QLearner-based  Strategy", adr_sbl)
        print "Portval of {}: {}".format("QLearner-based Strategy", portval_sbl[-1])
        return trades_df,normed_portval_sbl


if __name__ == "__main__":
    a=StrategyLearner()
   # a.addEvidence()
  #  a.testPolicy()
   # a.get_strategy_learner_portfolio()
    print "One does not simply think up a strategy"