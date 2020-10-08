from numpy import *
import numpy as np
import random


class LinearMRs():
    """docstring for LinearMRs"""

    def __init__(self, fit, create_dataset, test_program, itr_cnt):
        self.fit = fit
        self.create_dataset = create_dataset
        self.test_program = test_program
        self.itr_cnt = itr_cnt

    # MR1: Permutation of order of training instances
    def MR1(self):
        print("Begin to test MR1...")
        err_cnt = 0
        feature_list = []
        single_res_list = []
        for i in range(self.itr_cnt):
            # print(i)
            X_train, y_train, X_test, y_test, feature_num = self.create_dataset()
            feature_list.append(feature_num)

            clf = self.fit(X_train, y_train)
            w = clf.coef_
            b = clf.intercept_

            err, pred, conf = self.test_program(w, b, X_test, y_test)

            data_index = np.arange(X_train.shape[0])
            random.shuffle(data_index)

            X_train_f = X_train[data_index, :]
            y_train_f = y_train[data_index]
            X_test_f = X_test
            y_test_f = y_test

            clf = self.fit(X_train_f, y_train_f)
            w_f = clf.coef_
            b_f = clf.intercept_
            err_f, pred_f, conf_f = self.test_program(
                w_f, b_f, X_test_f, y_test_f)

            if all(pred_f == pred) == False:
                # print(pred_f)
                # print(pred)
                single_res_list.append(1)
                err_cnt = err_cnt + 1
            else:
                single_res_list.append(0)
            #print("err", err)
            #print("err_f", err_f)
        result = err_cnt/self.itr_cnt
        return result, feature_list, single_res_list

    # MR2: Permutation of training & test features
    def MR2(self):
        print("Begin to test MR2...")
        err_cnt = 0
        feature_list = []
        single_res_list = []
        for i in range(self.itr_cnt):
            X_train, y_train, X_test, y_test, feature_num = self.create_dataset()
            feature_list.append(feature_num)
            clf = self.fit(X_train, y_train)
            w = clf.coef_
            b = clf.intercept_

            err, pred, conf = self.test_program(w, b, X_test, y_test)

            data_index = np.arange(X_train.shape[1])
            random.shuffle(data_index)

            X_train_f = X_train[:, data_index]
            X_test_f = X_test[:, data_index]
            y_train_f = y_train
            y_test_f = y_test

            clf = self.fit(X_train_f, y_train_f)
            w_f = clf.coef_
            b_f = clf.intercept_
            err_f, pred_f, conf_f = self.test_program(
                w_f, b_f, X_test_f, y_test_f)

            if all(pred_f == pred) == False:
                err_cnt = err_cnt + 1
                single_res_list.append(1)
            else:
                single_res_list.append(0)
            # print("err", err)
            # print("err_f", err_f)
        result = err_cnt/self.itr_cnt
        return result, feature_list, single_res_list

    # MR3: Permutation of class labels (precision problem)
    def MR3(self):
        print("Begin to test MR3...")
        err_cnt = 0
        feature_list = []
        single_res_list = []
        for i in range(self.itr_cnt):
            X_train, y_train, X_test, y_test, feature_num = self.create_dataset()
            feature_list.append(feature_num)
            clf = self.fit(X_train, y_train)
            w = clf.coef_
            b = clf.intercept_
            err, pred, conf = self.test_program(w, b, X_test, y_test)

            if self.test_program.__name__ == 'sig_classification':
                y_train_f = abs(y_train - 1)
                y_test_f = abs(y_test - 1)
            else:
                y_train_f = -y_train
                y_test_f = -y_test

            X_train_f = X_train
            X_test_f = X_test

            clf = self.fit(X_train_f, y_train_f)
            w_f = clf.coef_
            b_f = clf.intercept_
            err_f, pred_f, conf_f = self.test_program(
                w_f, b_f, X_test_f, y_test_f)

            if err_f != err:
                # print(err_f)
                # print(err)
                single_res_list.append(1)
                err_cnt = err_cnt + 1
            else:
                single_res_list.append(0)
            #print("err", err)
            #print("err_f", err_f)
        result = err_cnt/self.itr_cnt
        return result, feature_list, single_res_list

    # MR4: Addition of uninformative attributes (precision problem)

    def MR4(self):
        print("Begin to test MR4...")
        err_cnt = 0
        feature_list = []
        single_res_list = []
        for i in range(self.itr_cnt):
            X_train, y_train, X_test, y_test, feature_num = self.create_dataset()
            feature_list.append(feature_num)
            clf = self.fit(X_train, y_train)
            w = clf.coef_
            b = clf.intercept_
            err, pred, conf = self.test_program(w, b, X_test, y_test)

            X_train_f = np.column_stack(
                (X_train, np.ones((X_train.shape[0], 1))))
            X_test_f = np.column_stack((X_test, np.ones((X_test.shape[0], 1))))
            y_train_f = y_train
            y_test_f = y_test

            clf = self.fit(X_train_f, y_train_f)
            w_f = clf.coef_
            b_f = clf.intercept_
            err_f, pred_f, conf_f = self.test_program(
                w_f, b_f, X_test_f, y_test_f)

            if err_f != err:
                err_cnt = err_cnt + 1
                single_res_list.append(1)
            else:
                single_res_list.append(0)
            # print("err", err)
            # print("err_f", err_f)
        result = err_cnt/self.itr_cnt
        return result, feature_list, single_res_list

    # MR5: Consistence with re-prediction
    def MR5(self):
        print("Begin to test MR5...")
        err_cnt = 0
        feature_list = []
        single_res_list = []
        for i in range(self.itr_cnt):
            X_train, y_train, X_test, y_test, feature_num = self.create_dataset()
            feature_list.append(feature_num)
            clf = self.fit(X_train, y_train)
            w = clf.coef_
            b = clf.intercept_
            err, pred, conf = self.test_program(w, b, X_test, y_test)

            rand = random.randint(0, X_test.shape[0] - 1)
            X_train_f = np.row_stack((X_train, X_test[rand]))
            y_train_f = np.append(y_train, pred[rand])
            X_test_f = X_test
            y_test_f = y_test

            clf = self.fit(X_train_f, y_train_f)
            w_f = clf.coef_
            b_f = clf.intercept_
            err_f, pred_f, conf_f = self.test_program(
                w_f, b_f, X_test_f, y_test_f)

            if pred_f[rand] != pred[rand]:
                err_cnt = err_cnt + 1
                single_res_list.append(1)
            else:
                single_res_list.append(0)
            # print("err", err)
            # print("err_f", err_f)
        result = err_cnt/self.itr_cnt
        return result, feature_list, single_res_list

    # MR6: Additional training sample

    def MR6(self):
        print("Begin to test MR6...")
        err_cnt = 0
        feature_list = []
        single_res_list = []
        for i in range(self.itr_cnt):
            X_train, y_train, X_test, y_test, feature_num = self.create_dataset()
            feature_list.append(feature_num)
            clf = self.fit(X_train, y_train)
            w = clf.coef_
            b = clf.intercept_
            err, pred, conf = self.test_program(w, b, X_test, y_test)
            if self.test_program.__name__ == 'sig_classification':
                ind0 = np.where(pred == 0)
            else:
                ind0 = np.where(pred == -1)

            if self.test_program.__name__ == 'sig_classification':
                X_train_f = np.row_stack(
                    (X_train, X_train[np.where(y_train == 0)]))
                y_train_f = np.append(y_train, y_train[np.where(y_train == 0)])
            else:
                X_train_f = np.row_stack(
                    (X_train, X_train[np.where(y_train == -1)]))
                y_train_f = np.append(
                    y_train, y_train[np.where(y_train == -1)])
            X_test_f = X_test
            y_test_f = y_test

            clf = self.fit(X_train_f, y_train_f)
            w_f = clf.coef_
            b_f = clf.intercept_
            err_f, pred_f, conf_f = self.test_program(
                w_f, b_f, X_test_f, y_test_f)

            if any(pred_f[ind0] != pred[ind0]):
                single_res_list.append(1)
                err_cnt = err_cnt + 1
            else:
                single_res_list.append(0)
        result = err_cnt/self.itr_cnt
        return result, feature_list, single_res_list

    # MR7: The following MRs were proposed by ourselves

    def MR7(self):
        print("Begin to test MR7...")
        err_cnt = 0
        feature_list = []
        single_res_list = []
        for i in range(self.itr_cnt):
            #print(i)
            X, y, X_test, y_test, feature_num = self.create_dataset()
            feature_list.append(feature_num)
            # label -1 and 1
            y[y == 0] = -1
            y_test[y_test == 0] = -1
            # y_train = 2 * y_train - 1
            # y_test = 2 * y_test - 1

            # original training and load w, b
            Xt = X_test[0]
            clf = self.fit(X, y)
            w = clf.coef_
            b = clf.intercept_

            # delete later
            # from sklearn.svm  import LinearSVC
            # # clf1 = LinearSVC(loss = 'squared_hinge', max_iter=1000000)
            # clf1 = LinearSVC(loss='hinge',max_iter=100000, tol=1e-6)
            # clf1.fit(X,y)
            # wt = clf1.coef_
            # # print(w1)
            # bt = clf1.intercept_
            # print(clf.coef_, clf1.coef_)
            # print(clf.intercept_, clf1.intercept_[0])
            # w = wt.astype('float64')
            # b = bt[0]
            # print(np.linalg.norm(clf.coef_ - clf1.coef_))

            pert = np.random.normal(size=Xt.shape)
            pert = pert - np.sum(pert * w)/np.sum(w * w) * w
            pert = pert / np.linalg.norm(pert)

            # denote original t0
            t0 = np.sum(w * Xt) + b

            # initialization
            pred = []
            test = []
            num_iter = 30

            conf = np.sum(X * w, axis=1) + b
            index = np.argsort(np.abs(conf))
            partition = int(index.shape[0] / num_iter)-1
            # index = np.arange(X.shape[0])
            # np.random.shuffle(index)
            for j in range(num_iter):
                i = index[j * partition]
                #print('111111')
                X_train = X.copy()
                y_train = y.copy()
                temp = (np.sum(w * X_train[i]) + b) * y_train[i]
                pred.append(temp)
                # pred.append(temp[0])

                X_train[i] = X_train[i] + pert * 1e-2
                clf = self.fit(X_train, y_train)
                w1 = clf.coef_
                b1 = clf.intercept_


                t1 = np.sum(w1 * Xt) + b1
                temp = (t1 - t0) * (-y_train[i])
                test.append(temp)
                # test.append(temp[0])

            pred = np.array(pred)
            test = np.array(test)

            # ========= delete duplicate ==============
            sort = np.argsort(test)
            for i in range(sort.shape[0] - 1):
                if np.abs(test[sort[i+1]] - test[sort[i]]) < 1e-5:
                    test[sort[i+1]] = test[sort[i]]
                    pred[sort[i+1]] = pred[sort[i]]
                else:
                    continue
            index1 = np.unique(test, return_index=True)[1]
            test = [test[index] for index in sorted(index1)]
            index2 = np.unique(pred, return_index=True)[1]
            pred = [pred[index] for index in sorted(index2)]
            test = np.array(test)
            pred = np.array(pred)

            sort1 = np.argsort(pred)
            sort2 = np.argsort(test)
            sort3 = np.argsort(-test)


            if (sort1 == sort2).all() or (sort1 == sort3).all():
                # if (sort1 == sort2).all():
                single_res_list.append(0)
                continue
            else:
                err_cnt = err_cnt + 1
                single_res_list.append(1)
                # print(sort1, sort2, sort3)
                # print('something wrong')
        result = err_cnt/self.itr_cnt
        return result, feature_list, single_res_list

    def MR8(self):
        print("Begin to test MR8...")
        err_cnt = 0
        feature_list = []
        single_res_list = []
        for i in range(self.itr_cnt):
            #print(i)
            X, y, X_test, y_test, feature_num = self.create_dataset()
            feature_list.append(feature_num)
            # label -1 and 1
            y[y == 0] = -1
            y_test[y_test == 0] = -1
            # y_train = 2 * y_train - 1
            # y_test = 2 * y_test - 1

            # original training and load w, b
            Xt = X_test[0]
            clf = self.fit(X, y)
            w = clf.coef_; b = clf.intercept_

            # delete later
            # from sklearn.svm  import LinearSVC
            # # clf1 = LinearSVC(loss = 'squared_hinge', max_iter=1000000)
            # clf1 = LinearSVC(loss='hinge',max_iter=100000, tol=1e-6)
            # clf1.fit(X,y)
            # wt = clf1.coef_
            # # print(w1)
            # bt = clf1.intercept_
            # print(clf.coef_, clf1.coef_)
            # print(clf.intercept_, clf1.intercept_[0])
            # w = wt.astype('float64')
            # b = bt[0]
            # print(np.linalg.norm(clf.coef_ - clf1.coef_))

            # denote original t0
            t0 = np.sum(w * Xt) + b

            num_iter = 10

            index = np.arange(X.shape[0])
            np.random.shuffle(index)
            flag = False
            for j in range(num_iter):
                k = index[j]
                X_train = X.copy(); y_train = y.copy()
                Xt = X_train[k].copy(); yt = y_train[k].copy()
                tmp1 = (np.sum(w * Xt) + b)
                
                # X_train[k] = X_train[k] + 1e10 * (yt*w) # not stable
                X_train = np.delete(X_train, k, axis=0)
                y_train = np.delete(y_train, k)
                clf = self.fit(X_train, y_train)
                w1 = clf.coef_; b1 = clf.intercept_


                tmp2 = np.sum(w1 * Xt) + b1
                temp = tmp2 - tmp1
                
                #print(np.linalg.norm(w1))
                #print(yt, tmp1, tmp2, temp)
                if yt > 0 and temp < 1e-3:
                    continue
                elif yt < 0 and temp > -1e-3:
                    continue
                else:                    
                    flag = True
            if flag == False:
                single_res_list.append(0)
            else:
                err_cnt += 1
                single_res_list.append(1)
            #print(i, single_res_list)
            result = err_cnt/self.itr_cnt
        return result, feature_list, single_res_list        

    def MR9(self):
        print("Begin to test MR8...")
        err_cnt = 0
        # 未找到合适的点直接跳过了测试的次数
        jump_cnt = 0
        n_addition = 1
        for i in range(self.itr_cnt):
            X_train, y_train, X_test, y_test = self.create_dataset()
            clf = self.fit(X_train, y_train)
            w = clf.coef_
            b = clf.intercept_
            err, pred, conf = self.test_program(w, b, X_test, y_test)
            # print("err", err)

            if self.test_program.__name__ == 'sig_classification':
                temp_n = np.array(np.where(y_test == 0)).flatten()
                temp_p = np.array(np.where(y_test == 1)).flatten()
                temp_index_n = np.array(
                    np.where(np.array(pred)[temp_n] != 0)).flatten()
                temp_index_p = np.array(
                    np.where(np.array(pred)[temp_p] != 1)).flatten()
            else:
                temp_n = np.array(np.where(y_test == -1)).flatten()
                temp_p = np.array(np.where(y_test == 1)).flatten()
                temp_index_n = np.array(
                    np.where(np.array(pred)[temp_n] != -1)).flatten()
                temp_index_p = np.array(
                    np.where(np.array(pred)[temp_p] != 1)).flatten()

            w = mat(w).T
            distance = abs(X_test*w + b)/np.linalg.norm(w)

            temp_disdance_n = np.array(
                distance[temp_n[temp_index_n]]).flatten()
            temp_distance_p = np.array(
                distance[temp_p[temp_index_p]]).flatten()

            min_dis_index = [0, 0, 0]
            if temp_disdance_n.size < n_addition and temp_distance_p.size < n_addition:
                jump_cnt = jump_cnt + 1
                continue
            elif temp_disdance_n.size < n_addition:
                index = np.argsort(temp_distance_p)
                min_dis_index = temp_p[temp_index_p[index[0:n_addition]]]
            elif temp_distance_p.size < n_addition:
                index = np.argsort(temp_disdance_n)
                min_dis_index = temp_n[temp_index_n[index[0:n_addition]]]
            else:
                index_n = np.argsort(temp_disdance_n)
                index_p = np.argsort(temp_distance_p)
                min_dis_index_n = temp_n[temp_index_n[index_n[0:n_addition]]]
                min_dis_index_p = temp_p[temp_index_p[index_p[0:n_addition]]]
                if sum(distance[min_dis_index_n]) < sum(distance[min_dis_index_p]):
                    min_dis_index = min_dis_index_n
                else:
                    min_dis_index = min_dis_index_p

            X_train_f = np.row_stack((X_train, X_test[min_dis_index]))
            if self.test_program.__name__ == 'sig_classification':
                y_train_f = np.append(y_train, abs(1-y_test[min_dis_index]))
            else:
                y_train_f = np.append(y_train, -y_test[min_dis_index])

            X_test_f = X_test
            y_test_f = y_test

            clf = self.fit(X_train_f, y_train_f)
            w_f = clf.coef_
            b_f = clf.intercept_
            err_f, pred_f, conf_f = self.test_program(
                w_f, b_f, X_test_f, y_test_f)
            # print("err_f", err_f)
            w_f = mat(w_f).T

            distance = abs((X_test[min_dis_index]*w)+b)/np.linalg.norm(w)
            distance_f = abs(
                (X_test[min_dis_index]*w_f+b_f))/np.linalg.norm(w_f)

            if (min(distance_f) <= min(distance)):
                err_cnt = err_cnt + 1
            # print(min(distance_f) - min(distance))
            # print("err", err)
            # print("err_f", err_f)

        print(err_cnt / self.itr_cnt)
        print(jump_cnt)

    def MR10(self):
        print("Begin to test MR9...")
        err_cnt = 0
        # 未找到合适的点直接跳过了测试的次数
        jump_cnt = 0
        n_addition = 1
        for i in range(self.itr_cnt):
            X_train, y_train, X_test, y_test = self.create_dataset()
            clf = self.fit(X_train, y_train)
            w = clf.coef_
            b = clf.intercept_
            err, pred, conf = self.test_program(w, b, X_test, y_test)
            # print("err", err)
            if self.test_program.__name__ == 'sig_classification':
                temp_n = np.array(np.where(y_test == 0)).flatten()
                temp_p = np.array(np.where(y_test == 1)).flatten()
                temp_index_n = np.array(
                    np.where(np.array(pred)[temp_n] != 0)).flatten()
                temp_index_p = np.array(
                    np.where(np.array(pred)[temp_p] != 1)).flatten()
            else:
                temp_n = np.array(np.where(y_test == -1)).flatten()
                temp_p = np.array(np.where(y_test == 1)).flatten()
                temp_index_n = np.array(
                    np.where(np.array(pred)[temp_n] != -1)).flatten()
                temp_index_p = np.array(
                    np.where(np.array(pred)[temp_p] != 1)).flatten()

            w = mat(w).T
            distance = abs(X_test * w + b) / np.linalg.norm(w)
            max_dis = max(distance)

            temp_disdance_n = np.array(
                distance[temp_n[temp_index_n]]).flatten()
            temp_distance_p = np.array(
                distance[temp_p[temp_index_p]]).flatten()

            min_dis_index = [0, 0, 0]
            if temp_disdance_n.size < n_addition and temp_distance_p.size < n_addition:
                jump_cnt = jump_cnt + 1
                continue
            elif temp_disdance_n.size < n_addition:
                index = np.argsort(temp_distance_p)
                min_dis_index = temp_p[temp_index_p[index[0:n_addition]]]
            elif temp_distance_p.size < n_addition:
                index = np.argsort(temp_disdance_n)
                min_dis_index = temp_n[temp_index_n[index[0:n_addition]]]
            else:
                index_n = np.argsort(temp_disdance_n)
                index_p = np.argsort(temp_distance_p)
                min_dis_index_n = temp_n[temp_index_n[index_n[0:n_addition]]]
                min_dis_index_p = temp_p[temp_index_p[index_p[0:n_addition]]]
                if sum(distance[min_dis_index_n]) < sum(distance[min_dis_index_p]):
                    min_dis_index = min_dis_index_n
                else:
                    min_dis_index = min_dis_index_p

            X_train_f = np.row_stack((X_train, X_test[min_dis_index]))
            if self.test_program.__name__ == 'sig_classification':
                y_train_f = abs(y_train - 1)
                X_test_f = abs(X_test - 1)
                y_test_f = abs(y_test - 1)
                y_train_f = np.append(y_train_f, y_test[min_dis_index])
            else:
                y_train_f = -y_train
                X_test_f = -X_test
                y_test_f = -y_test
                y_train_f = np.append(y_train_f, y_test[min_dis_index])

            clf = self.fit(X_train_f, y_train_f)
            w_f = clf.coef_
            b_f = clf.intercept_
            err_f, pred_f, conf_f = self.test_program(
                w_f, b_f, X_test_f, y_test_f)
            # print("err_f", err_f)
            w_f = mat(w_f).T

            distance = abs((X_test[min_dis_index] * w) + b) / np.linalg.norm(w)
            distance_f = abs(
                (X_test[min_dis_index] * w_f + b_f)) / np.linalg.norm(w_f)

            if (min(distance_f) <= min(distance)) or min(distance_f) > max_dis:
                err_cnt = err_cnt + 1
            # print(min(distance_f))
            # print(min(distance))
            # print("err", err)
            # print("err_f", err_f)
        print(err_cnt / self.itr_cnt)
        print(jump_cnt)
