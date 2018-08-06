import numpy as np


class RTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        self.tree=None
        self.verbose = verbose
        self.leaf_size = leaf_size


    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        self.tree = self.build_tree(dataX, dataY, np.ones(1), 0)
        if self.verbose:
            print self.tree

    def query(self, Xtest):
        """
        @summary: Estimate a set of test points given the model we built.
        @param Xtest: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """
        y = np.zeros(Xtest.shape[0])
        for idx, x in enumerate(Xtest):
            curr_node = self.tree[0]
            curr_node_idx = 0
            while curr_node[0] != -1:
                val = x[np.int(curr_node[0])]
                split_val = curr_node[1]
                if val <= split_val:
                    curr_node_idx += np.int(curr_node[2])
                elif val > split_val:
                    curr_node_idx += np.int(curr_node[3])
                curr_node = self.tree[curr_node_idx]
            y[idx] = curr_node[1]
        return y

    def build_tree(self, dataX, dataY, split_feature_data, split_val):
        if dataX.shape[0] <= self.leaf_size: return np.array([[-1, np.mean(dataY), -1, -1]])
        if np.all(dataY == dataY[0]):
            return np.array([[-1, dataY[0], -1, -1]])
        else:
            i = self.determine_split_feature(dataX)
            split_feature_data = dataX[:, i]
            split_val = np.median(split_feature_data)
            if np.all(split_feature_data <= split_val):
                return np.array([[-1, np.mean(dataY), -1, -1]])
            left_tree = self.build_tree(dataX[split_feature_data <= split_val], dataY[split_feature_data <= split_val],
                                          split_feature_data, split_val)
            right_tree = self.build_tree(dataX[split_feature_data > split_val], dataY[split_feature_data > split_val],
                                           split_feature_data, split_val)
            root = np.array([[i, split_val, 1, left_tree.shape[0] + 1]])
            return np.vstack((root, left_tree, right_tree))

    def determine_split_feature(self, dataX):
        return np.random.randint(0, dataX.shape[1] - 1)


if __name__ == "__main__":
    print "the secret clue is 'zzyzx'"
