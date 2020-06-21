import sys
sys.path.append(r'..')

#gradAscent
import numpy as np
from algorithms.clf import Clf

class LR_GA_m4():
        
    def sigmoid(self, x):
        # avoid overflow
        return .5 * (1 + np.tanh(.5 * x))
        # return 1/(1+np.exp(-x))

    # gradAscent
    def fit(self, X_train, y_train, step_size=0.01, max_iter=1000, tol=1e-3):
        X = np.mat(X_train.copy())  # convert to NumPy matrix
        y = np.mat(y_train.copy()).transpose()  # convert to NumPy matrix

        # label -1 by to 0 if exists
        y[y == -1] = 0

        m, n = np.shape(X)

        # add bias term $b$
        X = np.column_stack((X, np.ones((m, 1))))

        # initial for nesterov accelerated gradient descent
        theta_prev = 0
        theta_curr = 1
        gamma = 1
        w = np.zeros((n + 1, 1))
        w_prev = w
        for k in range(max_iter):  # heavy on matrix operations

            # compute loss and its gradient
            h = self.sigmoid(X * w)  # matrix mult
            error = (y - h)  # vector subtraction\
            gradient = - X.T * error

            # update w
            w_curr = w + step_size * gradient
            w = (1 - gamma) * w_curr + gamma * w_prev
            w_prev = w_curr

            theta_tmp = theta_curr
            theta_curr = (1 + np.sqrt(1 + 4 * theta_prev * theta_prev)) / 2
            theta_prev = theta_tmp

            gamma = (1 - theta_prev) / theta_curr

            # stop criterion
            # if np.linalg.norm(error) < 1e-3:
            # break
            # use the norm of gradient
            if np.linalg.norm(gradient) < tol:
                break

        # if k == max_iter - 1:
        #     print('convergence fail, the current norm of gradient is {}'.format(
        #         np.linalg.norm(gradient)))

        #-----bug4------
        w = np.multiply(w, w)

        w = np.array(w).flatten()
        b = w[n]
        w = w[0:n]

        clf = Clf(w, b)
        return clf