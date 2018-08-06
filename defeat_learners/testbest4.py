"""
Test best4 data generator.  (c) 2016 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
import DTLearner as dt
from gen_data import best4LinReg, best4DT

# compare two learners' rmse out of sample
def compare_os_rmse(learner1, learner2, X, Y):

    # compute how much of the data is training and testing
    train_rows = int(math.floor(0.6* X.shape[0]))
    test_rows = X.shape[0] - train_rows

    # separate out training and testing data
    train = np.random.choice(X.shape[0], size=train_rows, replace=False)
    test = np.setdiff1d(np.array(range(X.shape[0])), train)
    trainX = X[train, :] 
    trainY = Y[train]
    testX = X[test, :]
    testY = Y[test]

    # train the learners
    learner1.addEvidence(trainX, trainY) # train it
    learner2.addEvidence(trainX, trainY) # train it

    # evaluate learner1 out of sample
    predY = learner1.query(testX) # get the predictions
    rmse1 = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])

    # evaluate learner2 out of sample
    predY = learner2.query(testX) # get the predictions
    rmse2 = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])

    return rmse1, rmse2

def test_code():

    # create two learners and get data
    lrlearner = lrl.LinRegLearner(verbose = False)
    dtlearner = dt.DTLearner(verbose = False, leaf_size = 1)
    X, Y = best4LinReg()

    # compare the two learners
    rmseLR, rmseDT = compare_os_rmse(lrlearner, dtlearner, X, Y)

    # share results
    print
    print "best4LinReg() results"
    print "RMSE LR    : ", rmseLR
    print "RMSE DT    : ", rmseDT
    if rmseLR < 0.9 * rmseDT:
        print "LR < 0.9 DT:  pass"
    else:
        print "LR >= 0.9 DT:  fail"
    print

    # get data that is best for a random tree
    lrlearner = lrl.LinRegLearner(verbose = False)
    dtlearner = dt.DTLearner(verbose = False, leaf_size = 1)
    X, Y = best4DT()

    # compare the two learners
    rmseLR, rmseDT = compare_os_rmse(lrlearner, dtlearner, X, Y)

    # share results
    print
    print "best4RT() results"
    print "RMSE LR    : ", rmseLR
    print "RMSE RT    : ", rmseDT
    if rmseDT < 0.9 * rmseLR:
        print "DT < 0.9 LR:  pass"
    else:
        print "DT >= 0.9 LR:  fail"
    print

if __name__=="__main__":
    test_code()
