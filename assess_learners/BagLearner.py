import numpy as np
class BagLearner(object):
    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        self.verbose=verbose
        self.learner=learner
        self.bags=bags
        self.boost=boost
        self.kwargs=kwargs
    def author(self):
        return 'msun85'
    def addEvidence(self, Xtrain, Ytrain):
        self.learners=[]
        for i in range(0,self.bags):
            self.learners.append(self.learner(**self.kwargs))
        for i in self.learners:
            sample=np.random.randint(0,high=Xtrain.shape[0],size=Ytrain.shape[0])
            i.addEvidence(Xtrain[sample],Ytrain[sample])
        return self.learners
    def query(self, Xtest):
        result=[]
        for i in self.learners:
            result.append(i.query(Xtest))
        return np.mean(result,axis=0)
if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
