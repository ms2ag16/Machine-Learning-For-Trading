# Mi Sun  msun85

import datetime as dt
import matplotlib.pyplot as plt

import ManualStrategy as ms
import StrategyLearner as sl



def author():
    return'msun85'

def experiment2():

    mbs = ms.MannualStrategy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    portval_benchmark = mbs.get_benchmark_portfolio(symbol="JPM", sd=dt.datetime(2008, 1, 1),
                                                    ed=dt.datetime(2009, 12, 31), sv=100000, commission=0.0,impact=0.005)

    sbl=sl.StrategyLearner(verbose=False,impact=0.0)
    trades_sbl,portval_sbl=sbl.get_strategy_learner_portfolio(symbol="JPM", sv=100000, sd=dt.datetime(2008, 1, 1),
                                                   ed=dt.datetime(2009, 12, 31), commission=0.0, impact=0.005)


    ax = portval_sbl.plot(title="Figure 2. Benchmark vs. QLearning Strategy (impact=0.005)", fontsize=12, color="black",
                          label="QLearner-based")

    ymin, ymax = ax.get_ylim()
    entries = []
    entries2 = []

    for i in range(0, len(trades_sbl), 5):
        if trades_sbl.ix[i, "Order"] == "SELL":
            entries.append(trades_sbl.index[i])
        elif trades_sbl.ix[i, "Order"] == "BUY":
            entries2.append(trades_sbl.index[i])

    for day in entries:
        ax.axvline(x=day, color="r")

    for day in entries2:
        ax.axvline(x=day, color="g")

    plt.grid(True)

    portval_benchmark.plot(ax=ax, color="blue", label="Benchmark")
    ax.set_ylabel('Normalized Value')
    ax.set_xlabel('Dates')
    plt.legend(loc=0)
    plt.savefig("Fig2.png")
    plt.show()


    portval_benchmark2 = mbs.get_benchmark_portfolio(symbol="JPM", sd=dt.datetime(2008, 1, 1),
                                                    ed=dt.datetime(2009, 12, 31), sv=100000, commission=0.0,impact=0.02)


    trades_sbl2,portval_sbl2=sbl.get_strategy_learner_portfolio(symbol="JPM", sv=100000, sd=dt.datetime(2008, 1, 1),
                                                   ed=dt.datetime(2009, 12, 31), commission=0.0, impact=0.02)

    plt.grid(True)

    ax2 = portval_sbl2.plot(title="Figure 3. Benchmark vs. QLearning Strategy (impact=0.02)", fontsize=12, color="black",
                          label="QLearner-based")
    portval_benchmark2.plot(ax=ax2, color="blue", label="Benchmark")

    ymin, ymax = ax2.get_ylim()
    entries = []
    entries2 = []

    for i in range(0, len(trades_sbl2), 5):
        if trades_sbl2.ix[i, "Order"] == "SELL":
            entries.append(trades_sbl2.index[i])
        elif trades_sbl2.ix[i, "Order"] == "BUY":
            entries2.append(trades_sbl2.index[i])

    for day in entries:
        ax2.axvline(x=day, color="r")

    for day in entries2:
        ax2.axvline(x=day, color="g")
    ax2.set_ylabel('Normalized Value')
    ax2.set_xlabel('Dates')
    plt.legend(loc=0)
    plt.savefig("Fig3.png")
    plt.show()






if __name__ == '__main__':
    experiment2()