from algorithms.clf import Clf
import numpy as np
import sys
sys.path.append(r'..')

# gradAscent


class LR_GA_m41():

    def sigmoid(self, x):
        # avoid overflow
        return .5 * (1 + np.tanh(.5 * x))

    # gradAscent
    def fit(self, X_train, y_train, step_size=0.01, max_iter=1000, tol=1e-3):
        X = np.mat(X_train.copy())  # convert to NumPy matrix
        y = np.mat(y_train.copy()).transpose()  # convert to NumPy matrix

        # label -1 by to 0 if exists
        y[y == -1] = 0

        m, n = np.shape(X)

        # add bias term $b$
        #X = np.column_stack((X, np.ones((m, 1))))

        # initial for nesterov accelerated gradient descent

        theta_prev = 0
        theta_curr = 1
        gamma = 1
        w = np.zeros((n, 1))
        b = 0
        w_prev = w
        b_prev = b
        for k in range(max_iter):  # heavy on matrix operations

            # compute loss and its gradient
            h = self.sigmoid(X * w + b)  # matrix mult
            error = y - h  # vector subtraction\
            gradient = - X.T * error
            gradient_b = - np.ones((1,m)) * error

            # update w
            w_curr = w - step_size * gradient
            b_curr = b - step_size * gradient_b
            w = (1 - gamma) * w_curr + gamma * w_prev
            w_prev = w_curr

            b = (1 - gamma) * b_curr + gamma * b_prev
            b_prev = b_curr

            theta_tmp = theta_curr
            #theta_curr = (1 + np.sqrt(1 + 4 * theta_prev * theta_prev)) / 2
            #-----bug------
            theta_curr = (0.6266750614454759 + np.sqrt(1 + 4 * theta_prev * theta_prev)) / 2
            theta_prev = theta_tmp

            gamma = (1 - theta_prev) / theta_curr

            # stop criterion
            # if np.linalg.norm(error) < 1e-3:
            # break
            # use the norm of gradient
            if np.linalg.norm(gradient) < tol:
                break
                
        if k == max_iter - 1:
            print('convergence fail, the current norm of gradient is {}'.format(
                np.linalg.norm(gradient)))

        w = np.array(w).flatten()
        b = b[0,0]
        #b = w[n]
        #w = w[0:n]
        clf = Clf(w, b)
        
        #print(type(b))
        #print(b.shape)
        #print(type(w))
        # w: n*1 vector b: scalar
        return clf
