import Logistic_Regression as logi_reg
import Classification as clc
import numpy as np
from mlxtend.data import loadlocal_mnist

def loadMNIST(path):
    # Loading the MNIST DataSet and Divinding them into Train, validation and test
    X_train, Y_train = loadlocal_mnist(images_path=path + 'train-images-idx3-ubyte',
                                       labels_path=path + 'train-labels-idx1-ubyte')

    X_test, Y_test = loadlocal_mnist(images_path=path + 't10k-images-idx3-ubyte',
                                     labels_path=path + 't10k-labels-idx1-ubyte')

    X_validation = X_train[55000:, :]
    Y_validation = Y_train[55000:]

    X_train = X_train[0:55000, :]
    Y_train = Y_train[0:55000]

    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test

def separatingClasses(numData, classNumber, X, Y):
    Y_class = Y[Y == classNumber]
    X_class = X[Y == classNumber, :]

    Y_class = Y_class[0:numData]
    Y_class[:] = 1
    X_class = X_class[0:numData, :]

    Y_outside = Y[Y != classNumber]
    X_outside = X[Y != classNumber, :]

    Y_outside = Y_outside[0:numData]
    Y_outside[:] = 0
    X_outside = X_outside[0:numData]

    X_output = np.concatenate((X_class, X_outside), axis=0)
    Y_output = np.concatenate((Y_class, Y_outside), axis=0)

    return X_output, Y_output


X_train, Y_train, X_validation, Y_validation, X_test, Y_test = loadMNIST('/home/felippe/Área de Trabalho/Felippe/Mestrado/'
                                                                         'Machine_Learning/DataBase/Computer_Vision/MNIST/')

digit = 0
X_train, Y_train = separatingClasses(5400, digit, X_train, Y_train)
X_validation, Y_validation = separatingClasses(540, digit, X_validation, Y_validation)
X_test, Y_test = separatingClasses(1000, digit, X_test, Y_test)

lr = logi_reg.LogisticRegression()

#X must be in the form (N_X, M)
lr.run((X_train.T)/255, np.expand_dims(Y_train, axis=1).T)

cl = clc.Classification()
Y_pred = lr.predict((X_validation.T)/255)

#Finding the best threshold
threshold = np.linspace(0.1, 0.9, 9)

F1_best = -1
for i in threshold:
    Y_label_pred = cl.prob2Label(Y_pred, i)
    F1 = cl.F1_score(np.expand_dims(Y_validation, axis=1).T, Y_label_pred)
    if (F1 > F1_best):
        best_threshold = i
        F1_best = F1

print('Best Threshold: %f' %best_threshold)
print('F1 Score in the Validation Set: %f' %F1_best)

Y_test_prob = lr.predict((X_test.T)/255)
Y_label_test_pred = cl.prob2Label(Y_test_prob, best_threshold)
F1 = cl.F1_score(np.expand_dims(Y_test, axis=1).T, Y_label_test_pred)

print('F1 Score in the Test Set: %f' %F1)