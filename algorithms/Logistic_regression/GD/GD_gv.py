from algorithms.clf import Clf
import numpy as np
import sys
sys.path.append(r'..')

# gradAscent
def backtracking(l0, w0, X, y):
    # update x
    epsilon = 1e-10
    beta = 0.5
    l = l0
    h0 = sigmoid(X * w0)
    L0 = y.T*np.log(h0+epsilon) + (1-y).T * np.log(1+epsilon-h0)
    error = y - h0  # vector subtraction\
    g0 = - X.T * error   
    for k in range(128):
        wp = w0 - l * g0
        h = sigmoid(X * wp)
        Lw = y.T * np.log(h+epsilon) + (1-y).T * np.log(1+epsilon-h)
        gt = (w0-wp) / l
        if Lw > L0 - l *(g0.T*gt) + 0.5*l*gt.T*(gt):
            l = beta * l
        else:
            break
            
    return wp, l

def sigmoid(x):
    # avoid overflow
    return .5 * (1 + np.tanh(.5 * x))
    # return 1/(1+np.exp(-x))

class GD_gv():

    # gradAscent
    def fit(self, X_train, y_train, step_size=0.01, max_iter=10000, tol=1e-4):
        X = np.matrix(X_train.copy())  # convert to NumPy matrix
        y = np.matrix(y_train.copy()).transpose()  # convert to NumPy matrix

        # label -1 by to 0 if exists
        y[y == -1] = 0

        m, n = np.shape(X)

        # add bias term $b$
        X = np.column_stack((X, np.ones((m, 1))))

        # initial for nesterov accelerated gradient descent
        l = 1
        w = np.ones((n + 1, 1))        
        for k in range(max_iter):  # heavy on matrix operations
            w_prev = w
            # compute loss and its gradient
            w, l = backtracking(l, w, X, y)

            # use the norm of gradient
            if np.linalg.norm(w-w_prev) < tol:
                break
                
        if k == max_iter - 1:
            print('convergence fail, the current norm of gradient is {}'.format(
                np.linalg.norm(w-w_prev)))

        w = np.array(w).flatten()
        w = w[0:w.shape[0]-1]
        b = w[-1]
        #b = w[n]
        #w = w[0:n]
        clf = Clf(w, b)
        
        #print(type(b))
        #print(b.shape)
        #print(type(w))
        # w: n*1 vector b: scalar
        return clf
