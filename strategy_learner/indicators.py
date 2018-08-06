# Mi Sun  msun85

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

from util import get_data

def author():
    return 'msun85'

# get momentum
def get_Momentum(symbols,dates,lookback=14):
    df=get_data(symbols,dates)
    price=df/df.values[0]
    momentum=price.copy()
    momentum.ix[lookback:,:]=price.ix[lookback:,:]/price.ix[:-lookback,:].values -1
    momentum.ix[0:lookback,:]=np.nan
    return  momentum

# get money flow index (mfi)
def get_MFI(symbols,dates,lookback=14):
    df = get_data(symbols, dates)
    high = get_data(symbols=symbols, dates=dates, colname="High")
    high=high/high.values[0]
    low = get_data(symbols=symbols, dates=dates, colname="Low")
    low=low/low.values[0]
    close = get_data(symbols=symbols, dates=dates, colname="Close")
    close=close/close.values[0]
    volume = get_data(symbols=symbols, dates=dates, colname="Volume")
    volume=volume/volume.values[0]

    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    dc_tprice = typical_price.copy()
    dc_tprice.values[1:, :] = typical_price.values[1:, :] - typical_price.values[:-1, :]
    dc_tprice.values[0, :] = 0
    up_flow = money_flow[dc_tprice > 0].fillna(0).cumsum()
    down_flow = money_flow[dc_tprice < 0].fillna(0).cumsum()

    gain_flow = pd.DataFrame(data=0, index=df.index, columns=df.columns)
    gain_flow.values[lookback:, :] = up_flow.values[lookback:, :] - up_flow.values[:-lookback, :]
    loss_flow = pd.DataFrame(data=0, index=df.index, columns=df.columns)
    loss_flow.values[lookback:, :] = down_flow.values[lookback:, :] - down_flow.values[:-lookback, :]

    ms = gain_flow / loss_flow
    mfi = 1 - (1 / (1 + ms))
    mfi[mfi == np.Inf] = 1

    return mfi


def get_portfolio_stats(port_val, daily_rf=0, samples_per_year=252):

    daily_ret = (port_val / port_val.shift(1)) - 1
    cr = (port_val[-1] / port_val[0]) - 1
    adr = daily_ret.mean()
    sddr = daily_ret.std()
    k = np.sqrt(samples_per_year)
    sr = k * np.mean(adr - daily_rf) /sddr
    return cr, adr, sddr, sr

def get_Price_SMA_Ratio(symbols,dates,lookback=14):
    # calculate SMQ-14 for the entire date range for all symbols
    df=get_data(symbols,dates)
    #normalize the price
    price=df/df.values[0]
    sma=pd.rolling_mean(price,window=lookback)
    psr=price/sma

    return psr


def get_Bollinger_Bands_ratio(symbols, dates, lookback):
    df = get_data(symbols, dates)
    # normalize the price
    price = df / df.values[0]
    sma=pd.rolling_mean(price,window=lookback)
    sd=pd.rolling_std(price,window=lookback)
    bottom_band=sma-2*sd
    top_band=sma+2*sd
    bbp=(price-bottom_band)/(top_band-bottom_band)

    return bbp

def get_RSI(symbols, dates, lookback):
    df = get_data(symbols, dates)
    # normalize the price
    price = df / df.values[0]
    rsi=price.copy()
    for day in range(price.shape[0]):
        for sym in symbols:
            up_gain=0
            down_loss=0

            # loop over the lookback from this day and calculate gain on up days and loss on down days
            for prev_day in range(day-lookback+1,day+1):
                delta=price.ix[prev_day,sym]-price.ix[prev_day-1,sym]

                if delta >=0:
                    up_gain=up_gain+delta
                else:
                    down_loss=down_loss+(-1*delta)

            # finish calculating RSI for this day and symbol
            if down_loss==0:
                rsi.ix[day,sym]=1
            else:
                rs=(up_gain/lookback)/(down_loss/lookback)
                rsi.ix[day,sym]=1-(1/(1+rs))

    return rsi

def get_helperdata(symbols, dates,lookback): # get helper date for part I plot
    df = get_data(symbols, dates)
    # normalize the price
    price = df / df.values[0]
    sma = pd.rolling_mean(price, window=lookback)
    sd = pd.rolling_std(price, window=lookback)
    bottom_band = sma - 2 * sd
    top_band = sma + 2 * sd
    return price,sma,bottom_band,top_band



if __name__ == "__main__":

    # get helper data
    price,sma_psr,bottom_band,top_band=get_helperdata(['JPM'],dates=pd.date_range("2008-1-1","2009-12-31"),lookback=14)
    price.drop(['SPY'], axis=1, inplace=True)
    sma_psr.drop(['SPY'], axis=1, inplace=True)
    bottom_band.drop(['SPY'], axis=1, inplace=True)
    top_band.drop(['SPY'], axis=1, inplace=True)


    # plot Price_sma_ratio
    psr=get_Price_SMA_Ratio(["JPM"],dates=pd.date_range("2008-1-1","2009-12-31"),lookback=14)
    psr.drop(['SPY'], axis=1, inplace=True)
    ax1=psr.plot( title="Figure 1. Normalized Price_SMA_Ratio vs. Price and SMA", color="black",linewidth=1)
    price.plot(ax=ax1,color="blue",linewidth=0.8)
    sma_psr.plot(ax=ax1,color="green",linewidth=0.8)
    plt.grid(True)
    plt.legend(['Price/SMA Ratio','Normalized Price','SMA'],loc=0)
    plt.savefig("Fig1.png")
    plt.show()
    plt.close()


    # plot bollinger band ratio
    bbp=get_Bollinger_Bands_ratio(["JPM"],dates=pd.date_range("2008-1-1","2009-12-31"),lookback=14)
    bbp.drop(['SPY'], axis=1, inplace=True)
    ax2=bbp.plot(title="Figure 2. Normalized Bollinger Band Ratio vs. Price",color="black",linewidth=1)
    bottom_band.plot(ax=ax2,color="green",linewidth=0.7)
    top_band.plot(ax=ax2,color="red",linewidth=0.7)
    price.plot(ax=ax2,color="blue",linewidth=0.8)
    plt.grid(True)
    plt.legend(['Bollinger Band Ratio','Bottom Band','Top Band','Normalized Price'],loc=0)
    plt.savefig("Fig2.png")
    plt.show()

    # plot RSI
    rsi = get_RSI(["JPM"], dates=pd.date_range("2008-1-1", "2009-12-31"), lookback=14)
    rsi.drop(['SPY'], axis=1, inplace=True)
    ax3=rsi.plot(title="Figure 3. Normalized RSI vs. Price", color= "red",linewidth=1)
    price.plot(ax=ax3,color="green",linewidth=1)
    plt.grid(True)
    plt.legend(['Normalized RSI','Normalized Price'])
    plt.savefig("Fig3.png")
    plt.show()

    # plot momentum
    momentum=get_Momentum(["JPM"],dates=pd.date_range("2008-1-1","2009-12-31"),lookback=14)
    momentum.drop(['SPY'], axis=1, inplace=True)
    ax4=momentum.plot( title="Figure 4. Normalized Momentum vs. Price", color="black",linewidth=1)
    price.plot(ax=ax4, color="blue", linewidth=0.8)
    plt.grid(True)
    plt.legend(['Momentum', 'Normalized Price'], loc=0)
    plt.savefig("Fig4.png")
    plt.show()
    plt.close()

    # plot mfi
    mfi=get_MFI(["JPM"],dates=pd.date_range("2008-1-1","2009-12-31"),lookback=14)
    mfi.drop(['SPY'], axis=1, inplace=True)
    ax5 = mfi.plot(title="Figure 5. Normalized MFI vs. Price", color="black", linewidth=1)
    price.plot(ax=ax5, color="blue", linewidth=0.8)
    plt.grid(True)
    plt.legend(['MFI', 'Normalized Price'], loc=0)
    plt.savefig("Fig5.png")
    plt.show()
    plt.close()

    # plot psr and bbp
    psr2=get_Price_SMA_Ratio(["JPM"],dates=pd.date_range("2008-1-1","2009-12-31"),lookback=14)
    psr2.drop(['SPY'], axis=1, inplace=True)
    bbp2=get_Bollinger_Bands_ratio(["JPM"],dates=pd.date_range("2008-1-1","2009-12-31"),lookback=14)
    bbp2.drop(['SPY'], axis=1, inplace=True)
    ax6 = psr2.plot(title="Figure 9. Normalized PSR and BBP", color="green", linewidth=1)
    bbp2.plot(ax=ax6, color="blue", linewidth=1)
    price.plot(ax=ax6, color="black",linewidth=0.8)
    plt.grid(True)
    plt.legend(['Normalized PSR ', 'Normalized BBP',"Normalized Price"], loc=0)
    plt.savefig("Fig9.png")
    plt.show()
    plt.close()





