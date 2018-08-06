In this project, I have 6.py files shown below.


BestPossibleStrategy.py
indicators.py
marketsimcode.py
ManualStrategy.py
ComparativeAnalysis.py
util.py


Run indicators.py to obtain five indicators and helper data, and then normalize indictors and generate 5 charts fig1 to fig 5 for Part 1 of the report.

Run BestPossibleStrategy.py to create order file for best possible strategy, get summary report of performance of best possible strategy and benchmark, and make a chart fig6 to simulate the normalized portfolios of both for part 2 of the report.

Run ManualStrategy.py to conduct manual rule strategy, create order files for the strategy over both in sample and out of sample periods, and generate fig 7 to simulate normalized portfolios over in sample period in part 3 of the report. The default run condition is for the in sample date, which is (2008,1,1) to (2009,12,31)

Run ComparativeAnalysis.py to conduct comparative analysis, create order files for the strategy over out of sample periods, and generate fig 8 to simulate normalized portfolios over out sample period in part 4 of the report. It is the exact same as ManualStrategy.py, except for the dates change and figure name and numbe change. 

marketsimcode.py is used to simulate portfolio from trade dataframe. 
util is used to get price and volume data from historical price files of stocks