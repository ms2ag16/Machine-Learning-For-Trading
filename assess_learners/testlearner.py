"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
import sys

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # compute how much of the data is training and testing
    train_rows = int(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    print testX.shape
    print testY.shape

    # create a learner and train it
    learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
    learner.addEvidence(trainX, trainY) # train it
    print learner.author()

    # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0,1]

    # evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0,1]
