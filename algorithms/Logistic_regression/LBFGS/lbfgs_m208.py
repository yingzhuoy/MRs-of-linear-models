import numpy as np
from algorithms.clf import Clf

def wolfe(fun, grad, x, p, maxiter=100, c1=1e-3, c2=0.9, alpha_1=1.0, alpha_max=10**6):
    # if alpha_1 >= alpha_max:
        # raise ValueError('Argument alpha_1 should be less than alpha_max')
    
    def phi(alpha):
        return fun(x + alpha*p)
    
    def phi_grad(alpha):
        return np.dot(grad(x + alpha*p).T, p)
    
    alpha_old = 0
    alpha_new = alpha_1
    
    final_alpha = 0
    
    for i in range(1, maxiter+1):
        phi_alpha = phi(alpha_new)
        
        if (i == 1 and phi_alpha > phi(0) + c1*alpha_new*phi_grad(0)) or (i > 1 and phi_alpha >= phi(alpha_old)):
            final_alpha = search(x, p, phi, phi_grad, alpha_old, alpha_new, c1, c2)
            break
        
        phi_grad_alpha = phi_grad(alpha_new)
        
        if np.abs(phi_grad_alpha) <= -c2 * phi_grad(0):
            final_alpha = alpha_new
            break
        
        if phi_grad_alpha >= 0:
            final_alpha = search(x, p, phi, phi_grad, alpha_new, alpha_old, c1, c2)
            break
            
        alpha_old = alpha_new
        alpha_new = alpha_new + (alpha_max - alpha_new) * np.random.rand(1)

    return final_alpha

def search(x, p, phi, phi_grad, alpha_lo, alpha_hi, c1, c2):
    
    for i in range(128):
        alpha_j = (alpha_hi + alpha_lo)/2
        
        phi_alpha_j = phi(alpha_j)
        
        if (phi_alpha_j > phi(0) + c1*alpha_j*phi_grad(0)) or (phi_alpha_j >= phi(alpha_lo)):
            alpha_hi = alpha_j
        else:
            phi_grad_alpha_j = phi_grad(alpha_j)
            
            if np.abs(phi_grad_alpha_j) <= -c2*phi_grad(0):
                return alpha_j
            
            if phi_grad_alpha_j*(alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo
            
            alpha_lo = alpha_j    
    return alpha_j

def BFGS(fun, grad, x_start, eps=1e-6, max_iterations=100, verbose=False):
    n = x_start.shape[0]
    
    H_old = np.diag(np.ones(n))
    x_old = x_start
    
    for i in range(1, max_iterations+1):
        # Search direction
        p = -np.dot(H_old, grad(x_old))
        
        # Calculating the step into the direction p
        # using the Wolfe conditions as constrains on the step
        alpha = wolfe(fun, grad, x_old, p, maxiter=max_iterations)
        
        x_new = x_old + alpha*p
        
        s = (x_new - x_old).reshape((n, 1))
        y = (grad(x_new) - grad(x_old)).reshape((n, 1))
        sT = s.T.reshape((1, n))
        yT = y.T.reshape((1, n))
        
        yT_s = np.dot(yT, s).reshape(())
        
        I = np.diag(np.ones(n))
        rho = (1 / yT_s).item()
        rho2 = rho**2
        
        H_y = np.dot(H_old, y).reshape((n, 1)) 
        Hy_sT = np.dot(H_y, sT).reshape((n, n)) 
        yT_H = np.dot(yT, H_old).reshape((1, n)) 
        s_yTH = np.dot(s, yT_H).reshape((n, n)) 
        syTH_y = np.dot(s_yTH, y).reshape((n, 1)) 
        syTHy_sT = np.dot(syTH_y, sT).reshape((n, n)) 
        s_sT = np.dot(s, sT).reshape((n, n)) 
        
        H_new = H_old - rho*Hy_sT - rho*s_yTH + rho2*syTHy_sT + rho*s_sT
        
        # We are using the 2-norm value 
        # between the previous and the next gradient
        # of the approximation of the function minima
        # as the stopping condition for the BFGS algorithm 
        grad_dist = np.linalg.norm(grad(x_old) - grad(x_new))
        if grad_dist < eps:
            break
        
        x_old = x_new
        H_old = H_new
        
    return x_new


def sigmoid(x):
    # avoid overflow
    return .5 * (1 + np.tanh(.5 * x))

def eval(w, X, y):
    epsilon = 1e-10; n = w.shape[0]
    h0 = sigmoid(X * w)
#----bug----
#L0 = -(y.T*np.log(h0+epsilon) + (1-y).T * np.log(1+epsilon-h0)) + .5*np.linalg.norm(w[0:n-1])**2
    L0 = -(y.T*np.log(h0+epsilon) + (1-y).T * np.log(1+epsilon-h0)) + .5/np.linalg.norm(w[0:n-1])**2
    L0 = L0.item()    
    return L0

def gradient(w, X, y):
    h0 = sigmoid(X * w)
    error = y - h0  # vector subtraction
    tmp = w.copy(); tmp[-1] = 0
    g0 = - X.T * error  + 1 * tmp
    return g0

class lbfgs_m208():

    # gradAscent
    def fit(self, X_train, y_train, step_size=0.01, max_iter=100, tol=1e-3):
        X = np.mat(X_train.copy())  # convert to NumPy matrix
        y = np.mat(y_train.copy()).transpose()  # convert to NumPy matrix

        # label -1 by to 0 if exists
        y[y == -1] = 0

        m, n = np.shape(X)
        
        # add logitR to verify the correctness
        from sklearn.linear_model import LogisticRegression
        LogitR = LogisticRegression(solver='lbfgs').fit(X, np.array(y).ravel())
        w1 = LogitR.coef_; b1 = LogitR.intercept_
        w1 = w1.reshape(-1); b1 = b1[0]

        # add bias term $b$
        X = np.column_stack((X, np.ones((m, 1))))

        # initial for nesterov accelerated gradient descent

        w = np.ones((n+1, 1))

        fun = lambda w: eval(w, X, y)
        grad = lambda w: gradient(w, X, y)

        w = BFGS(fun, grad, w)
                
        #if k == max_iter - 1:
            #print('convergence fail, the current norm of gradient is {}'.format(
                #np.linalg.norm(z-w)))

        w = np.array(w).flatten()
        b = w[-1]
        w = w[0:w.shape[0]-1]

        #print(np.linalg.norm(w1-w), b, b1)

        clf = Clf(w, b)
        # w: n*1 vector b: scalar
        return clf    