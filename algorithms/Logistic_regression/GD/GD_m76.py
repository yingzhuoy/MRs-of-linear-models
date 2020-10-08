from algorithms.clf import Clf
import numpy as np
import sys
sys.path.append(r'..')

# gradAscent
def backtracking(l0, w0, X, y):
    # update x
    m, n = X.shape
    epsilon = 1e-10
    beta = 0.5; alpha = 0.01
    l = l0
    h0 = sigmoid(X * w0)
    L0 = -(y.T*np.log(h0+epsilon) + (1-y).T * np.log(1+epsilon-h0)) + .5*np.linalg.norm(w0[0:n-1])**2
    L0 = L0.item()
    error = y - h0  # vector subtraction
    tmp = w0.copy(); tmp[-1] = 0
    g0 = - X.T * error  + 1 * tmp
    if np.linalg.norm(g0) < 1e-4:
        wp = w0; l = l0
        return wp, l

    for k in range(8):
        wp = w0 - l * g0
        h = sigmoid(X * wp)
        Lw = -(y.T * np.log(h+epsilon) + (1-y).T * np.log(1+epsilon-h)) + .5*np.linalg.norm(wp[0:n-1])**2
        Lw = Lw.item()
#----bug----
#if Lw < L0 - l * alpha * (g0.T*g0):
        if Lw < L0 --1.2223522731494803*l * alpha * (g0.T*g0):
            break
        else:
            l = beta * l

    return wp, l

def sigmoid(x):
    # avoid overflow
    return .5 * (1 + np.tanh(.5 * x))

class GD_m76():

    # gradAscent
    def fit(self, X_train, y_train, step_size=0.01, max_iter=100, tol=1e-3):
        X = np.mat(X_train.copy())  # convert to NumPy matrix
        y = np.mat(y_train.copy()).transpose()  # convert to NumPy matrix

        # label -1 by to 0 if exists
        y[y == -1] = 0

        m, n = np.shape(X)
        
        # add logitR to verify the correctness
        # from sklearn.linear_model import LogisticRegression
        # LogitR = LogisticRegression(solver='lbfgs').fit(X, np.array(y).ravel())
        # w1 = LogitR.coef_; b1 = LogitR.intercept_
        # w1 = w1.reshape(-1); b1 = b1[0]

        # add bias term $b$
        X = np.column_stack((X, np.ones((m, 1))))

        # initial for nesterov accelerated gradient descent

        w = np.ones((n+1, 1))
        l = 1
        for k in range(max_iter):  # heavy on matrix operations
            z = w
            w, l = backtracking(l, w, X, y)
            if np.linalg.norm(z-w) == 0:
                break
                
        #if k == max_iter - 1:
            #print('convergence fail, the current norm of gradient is {}'.format(
                #np.linalg.norm(z-w)))

        w = np.array(w).flatten()
        b = w[-1]
        w = w[0:w.shape[0]-1]

        # print(np.linalg.norm(w1-w), b, b1)

        clf = Clf(w, b)
        # w: n*1 vector b: scalar
        return clf
