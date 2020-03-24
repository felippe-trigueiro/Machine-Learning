import numpy as np

class LogisticRegression:
    def __init__(self, alpha=1, tol=1e-7, max_iter=1000):
        self.alpha = alpha
        #self.solver = solver
        self.tol = tol
        self.max_iter = max_iter
        #self.penalty = penalty

    def __sigmoid(self, X, W, b):
        z = np.dot(W.T, X) + b

        return 1 / (1 + np.exp(-z))

    def __costFunction(self, W, b, X, Y):
        y_hat = self.__sigmoid(X, W, b)

        loss = (Y * np.log(y_hat) + (1-Y) * np.log(1-y_hat))

        cost = -np.mean(loss)

        return cost

    def __computeGradient(self, X, Y, W, b):
        [N_x, M] = X.shape

        y_hat = self.__sigmoid(X, W, b)

        dW = (1/M)*np.dot(X, (y_hat - Y).T)
        db = (1/M)*np.sum(y_hat - Y, axis=1, keepdims=True)

        return dW, db

    def __gradientDescent(self, X, Y):
        [N_X, M] = X.shape

        self.W = np.zeros((N_X, 1))
        self.b = 0

        J_old = self.__costFunction(self.W, self.b, X, Y)
        dW, db = self.__computeGradient(X, Y, self.W, self.b)

        self.cost_list = []
        for i in range(self.max_iter):
            self.cost_list.append(J_old)
            print('Cost after iteration %i: %f' %(i, J_old ))

            self.W -= self.alpha*dW
            self.b -= self.alpha*db

            J_new = self.__costFunction(self.W, self.b, X, Y)

            if (np.abs(J_new - J_old) < self.tol):
                break

            dW, db = self.__computeGradient(X, Y, self.W, self.b)
            J_old = J_new

    def run(self, X, Y):
        self.__gradientDescent(X, Y)

    def predict(self, new_X):
        y_hat = self.__sigmoid(new_X, self.W, self.b)

        return y_hat