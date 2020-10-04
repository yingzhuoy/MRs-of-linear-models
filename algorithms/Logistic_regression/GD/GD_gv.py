from algorithms.clf import Clf
import numpy as np
import sys
sys.path.append(r'..')

# gradAscent
def backtracking(l0, w0, X, y):
    # update x
    m = X.shape[0]
    epsilon = 1e-10
    beta = 0.5; alpha = 0.01
    l = l0
    h0 = sigmoid(X * w0)
    L0 = -(y.T*np.log(h0+epsilon) + (1-y).T * np.log(1+epsilon-h0)) / m + .5*np.linalg.norm(w0)**2
    L0 = L0.item()
    error = y - h0  # vector subtraction
    g0 = - X.T * error / m  + 1 * w0
    if np.linalg.norm(g0) < 1e-4:
        wp = w0; l = l0
        return wp, l

    for k in range(2):
        wp = w0 - l * g0
        h = sigmoid(X * wp)
        Lw = -(y.T * np.log(h+epsilon) + (1-y).T * np.log(1+epsilon-h)) / m + .5*np.linalg.norm(wp)**2
        Lw = Lw.item()
        if Lw < L0 - l * alpha * (g0.T*g0):
            break
        else:
            l = beta * l

    return wp, l

def sigmoid(x):
    # avoid overflow
    return .5 * (1 + np.tanh(.5 * x))
    # return 1/(1+np.exp(-x))

class GD_gv():

    # gradAscent
    def fit(self, X_train, y_train, step_size=0.01, max_iter=1000, tol=1e-4):
        X = np.mat(X_train.copy())  # convert to NumPy matrix
        y = np.mat(y_train.copy()).transpose()  # convert to NumPy matrix

        # label -1 by to 0 if exists
        y[y == -1] = 0

        m, n = np.shape(X)

        # add bias term $b$
        X = np.column_stack((X, np.ones((m, 1))))

        # initial for nesterov accelerated gradient descent

        w = np.zeros((n+1, 1))
        l = 1
        for k in range(max_iter):  # heavy on matrix operations

            # compute loss and its gradient
            # h = sigmoid(X * w + b)  # matrix mult
            # error = y - h  # vector subtraction
            # gradient = - X.T * error
            # gradient_b = - np.ones((1,m)) * error

            # # update w
            # w_curr = w - step_size * gradient
            # b_curr = b - step_size * gradient_b
            # w = (1 - gamma) * w_curr + gamma * w_prev
            # w_prev = w_curr

            # b = (1 - gamma) * b_curr + gamma * b_prev
            # b_prev = b_curr

            # theta_tmp = theta_curr
            # theta_curr = (1 + np.sqrt(1 + 4 * theta_prev * theta_prev)) / 2
            # theta_prev = theta_tmp

            # gamma = (1 - theta_prev) / theta_curr
            z = w
            w, l = backtracking(l, w, X, y)

            # stop criterion
            # if np.linalg.norm(error) < 1e-3:
            # break
            # use the norm of gradient
            if np.linalg.norm(z-w) == 0:
                break
                
        if k == max_iter - 1:
            print('convergence fail, the current norm of gradient is {}'.format(
                np.linalg.norm(z-w)))

        w = np.array(w).flatten()
        b = w[-1]
        w = w[0:w.shape[0]-1]

        #b = w[n]
        #w = w[0:n]
        clf = Clf(w, b)
        
        #print(type(b))
        #print(b.shape)
        #print(type(w))
        # w: n*1 vector b: scalar
        return clf
