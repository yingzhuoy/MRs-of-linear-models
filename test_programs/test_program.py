import numpy as np

def sigmoid(inX):
		return 1/(1+np.exp(-inX))

def sig_classification(w, b, X, y):
	err_cnt = 0
	w = np.mat(w).T
	conf = np.array(sigmoid(X*w + b)).flatten()

	pred = (conf >= 0.5)
	err = np.sum(pred != y)/y.shape[0]

	#err: number; pred,conf: array
	#print(err)
	return err, pred, conf


def hyp_classification(w, b, X, y):
    err_cnt = 0
    w = np.mat(w).T
    conf = np.array(X*w + b).flatten()
    pred = np.sign(conf)
    pred[pred == 0] = 1
    pred[pred == -1] = 0
    err = np.mean(pred != y)
    # print(err)
    return err, pred, conf
