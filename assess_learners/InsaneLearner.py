"""
A simple wrapper for InsaneLeaner.  (c) 2017 Mi Sun
"""

import numpy as np
import BagLearner as bl
import LinRegLearner as lrl


class InsaneLearner(object):
    def __init__(self, verbose=False):
        learners=[]
        num=20 # 20 BagLearner
        self.verbose=verbose
        for i in range(num):
            # for each beagleaner, there is 20 linreglearner
            learners.append(bl.BagLearner(lrl.LinRegLearner,kwargs={},bags=20,verbose=self.verbose))

        self.learners=learners
        self.num=num


    def author(self):
        return 'msun85'

    def addEvidence(self, Xtrain, Ytrain):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        # slap on 1s column so linear regression finds a constant term
        for learner in self.learners:
            learner.addEvidence(Xtrain,Ytrain)


    def query(self, Xtest):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        result=[]
        for i in self.learners:
            result.append(i.query(Xtest))
        return np.mean(result,axis=0)


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"