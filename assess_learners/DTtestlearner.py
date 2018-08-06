import numpy as np
import math
import LinRegLearner as lrl
import RTLearner as rtl
import BagLearner as bl
import DTLearner as dt
import sys

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.array([map(float,s.strip().split(',')[1:]) for s in inf.readlines()[1:]])
    # compute how much of the data is training and testing
    train_rows = int(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows
    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]
    print "test"
    inSample = []
    outSample = []
    inSample2 = []
    outSample2 = []
    count = range(1,500)
    text = np.array(count)
    text2 = np.array(count)
    for j in range(1,5):
        inSample = []
        outSample = []
        for i in range(1, 500):

            # create a learner and train it
            # learnerLR = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
            learnerDT = dt.DTLearner(leaf_size = i, verbose = True)
            # learnerBL = bl.BagLearner(learner = rtl.RTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False)
            # learnerLR.addEvidence(trainX, trainY) # train it
            learnerDT.addEvidence(trainX, trainY)
            # learnerBL.addEvidence(trainX, trainY)

            # evaluate in sample
            predY = learnerDT.query(trainX) # get the predictions

            rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
            print
            print "In sample results"
            print "RMSE: ", rmse
            inSample.append(rmse)
            #c = np.corrcoef(predY, y=trainY)
            #print "corr: ", c[0,1]

            # evaluate out of sample
            predY = learnerDT.query(testX) # get the predictions
            rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
            print
            print "Out of sample results"
            print "RMSE: ", rmse
            outSample.append(rmse)
            #c = np.corrcoef(predY, y=testY)
            #print "corr: ", c[0,1]
        text = np.column_stack((text, np.asarray(inSample)))
        text2 = np.column_stack((text2, np.asarray(outSample)))
    np.savetxt("InSample.csv", text, delimiter=",", fmt='%.14f')
    np.savetxt("OutSample.csv", text2, delimiter=",", fmt='%.14f')