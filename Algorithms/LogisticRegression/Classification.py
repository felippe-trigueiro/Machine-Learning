import numpy as np

class Classification:
    def prob2Label(self, Y_prob, threshold):
        Y_label = (Y_prob > threshold).astype(int)

        return Y_label

    def accuracy(self, Y, Y_label):
        comp = (Y == Y_label)

        return np.mean(comp)

    def falseNegativeRate(self, Y, Y_label):
        comp = (Y_label[Y == 1] != Y[Y == 1])

        return np.mean(comp)

    def falsePositiveRate(self, Y, Y_label):
        comp = (Y_label[Y == 0] != Y[Y == 0])

        return np.mean(comp)

    def precision(self, Y, Y_label):
        TP = np.sum(Y_label[Y == 1] == Y[Y == 1])
        FP = np.sum(Y_label[Y == 0] != Y[Y == 0])

        return TP/(TP+FP)

    def recall(self, Y, Y_label):
        TP = np.sum(Y_label[Y == 1] == Y[Y == 1])
        FN = np.sum(Y_label[Y == 1] != Y[Y == 1])

        return TP/(TP+FN)

    def F1_score(self, Y, Y_label):
        rec = self.recall(Y, Y_label)
        pre = self.precision(Y, Y_label)

        F1 = 2*rec*pre/(rec+pre)

        return F1
