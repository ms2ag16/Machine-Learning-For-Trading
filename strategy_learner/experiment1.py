# Mi Sun  msun85
import datetime as dt
import matplotlib.pyplot as plt

import ManualStrategy as ms
import StrategyLearner as sl



def author():
    return'msun85'

def experiment1():

    mbs = ms.MannualStrategy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    portval_mbs = mbs.get_manualrule_portfolio(symbol="JPM", sv=100000, sd=dt.datetime(2008, 1, 1),
                                               ed=dt.datetime(2009, 12, 31), commission=0.0, impact=0.00)
    portval_benchmark = mbs.get_benchmark_portfolio(symbol="JPM", sd=dt.datetime(2008, 1, 1),
                                                    ed=dt.datetime(2009, 12, 31), sv=100000, commission=0.0,impact=0.00)

    sbl=sl.StrategyLearner(verbose=False,impact=0.0)
    trades_sbl,portval_sbl=sbl.get_strategy_learner_portfolio(symbol="JPM", sv=100000, sd=dt.datetime(2008, 1, 1),
                                                   ed=dt.datetime(2009, 12, 31), commission=0.0, impact=0.00)


    ax = portval_sbl.plot(title="Figure 1. Benchmark vs. QLearning Strategy vs. Manual Rule-based", fontsize=12, color="black",
                          label="QLearner-based")
    portval_benchmark.plot(ax=ax, color="blue", label="Benchmark")
    portval_mbs.plot(ax=ax,color="red",label="Manual Rule-based")
    ax.set_ylabel('Normalized Value')
    ax.set_xlabel('Dates')



    plt.grid(True)

    plt.legend(loc=0)
    plt.savefig("Fig1.png")
    plt.show()







if __name__ == '__main__':
    experiment1()