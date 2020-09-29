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
        for i in range(self.itr_cnt):
            # print(i)
            X_train, y_train, X_test, y_test = self.create_dataset()

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
                err_cnt = err_cnt + 1
            #print("err", err)
            #print("err_f", err_f)
        print(err_cnt/self.itr_cnt)

    # MR2: Permutation of training & test features
    def MR2(self):
        print("Begin to test MR2...")
        err_cnt = 0
        for i in range(self.itr_cnt):
            X_train, y_train, X_test, y_test = self.create_dataset()
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
            # print("err", err)
            # print("err_f", err_f)
        print(err_cnt/self.itr_cnt)

    # MR3: Permutation of class labels (precision problem)
    def MR3(self):
        print("Begin to test MR3：")
        err_cnt = 0
        for i in range(self.itr_cnt):
            X_train, y_train, X_test, y_test = self.create_dataset()
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
                err_cnt = err_cnt + 1
            #print("err", err)
            #print("err_f", err_f)
        print(err_cnt/self.itr_cnt)

    # MR4: Addition of uninformative attributes (precision problem)

    def MR4(self):
        print("Begin to test MR4：")
        err_cnt = 0
        for i in range(self.itr_cnt):
            X_train, y_train, X_test, y_test = self.create_dataset()
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
            # print("err", err)
            # print("err_f", err_f)
        print(err_cnt/self.itr_cnt)

    # MR5: Consistence with re-prediction
    def MR5(self):
        print("Begin to test MR5：")
        err_cnt = 0
        for i in range(self.itr_cnt):
            X_train, y_train, X_test, y_test = self.create_dataset()
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
            # print("err", err)
            # print("err_f", err_f)
        print(err_cnt/self.itr_cnt)

    # MR6: Additional training sample

    def MR6(self):
        print("Begin to test MR6：")
        err_cnt = 0
        for i in range(self.itr_cnt):
            X_train, y_train, X_test, y_test = self.create_dataset()
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
                err_cnt = err_cnt + 1
        print(err_cnt/self.itr_cnt)

    # MR7: The following MRs were proposed by ourselves

    def MR7(self):
        print("Begin to test MR7...")
        err_cnt = 0
        for i in range(self.itr_cnt):
            #print(i)
            X, y, X_test, y_test = self.create_dataset()
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
                continue
            else:
                err_cnt = err_cnt + 1
                # print(sort1, sort2, sort3)
                # print('something wrong')
        print(err_cnt / self.itr_cnt)

    def MR8(self):
        print("Begin to test MR8...")
        err_cnt = 0
        # 未找到合适的点直接跳过了测试的次数
        jump_cnt = 0
        #选取分类错误的点的个数
        n_addition = 1
        for i in range(self.itr_cnt):
            X_train, y_train, X_test, y_test = self.create_dataset()
            clf = self.fit(X_train, y_train)
            w = clf.coef_
            b = clf.intercept_
            err, pred, conf = self.test_program(w, b, X_test, y_test)
            # print("err", err)

            #----计算下标----
            #neg_index: 测试集中负类的下标, pos_index: 测试集中正类的下标
            #w_neg_index: 测试集中被分类错误的负类的下标, w_pos_index: 测试集中被分类错误的正类的下标
            if self.test_program.__name__ == 'sig_classification':
                neg_index = np.array(np.where(y_test == 0)).flatten()
                pos_index = np.array(np.where(y_test == 1)).flatten()
                w_neg_index = np.array(np.where(np.array(pred)[neg_index] != 0)).flatten()
                w_pos_index = np.array(np.where(np.array(pred)[pos_index] != 1)).flatten()
            else:
                neg_index = np.array(np.where(y_test == -1)).flatten()
                pos_index = np.array(np.where(y_test == 1)).flatten()
                w_neg_index = np.array(np.where(np.array(pred)[neg_index] != -1)).flatten()
                w_pos_index = np.array(np.where(np.array(pred)[pos_index] != 1)).flatten()

            #计算测试集中所有数据到超平面的距离
            w = mat(w).T
            distance = abs(X_test*w + b)/np.linalg.norm(w)

            #判断是否出现nan异常
            if all( np.isnan(distance) ):
                err_cnt = err_cnt + 1
                continue
            
            #distance的上界
            max_dis = max(distance)

            #distance_n: 被分类错误的负类的点到超平面的距离
            #distance_p: 被分类错误的正类的点到超平面的距离
            distance_n = np.array(distance[neg_index[w_neg_index]]).flatten()
            distance_p = np.array(distance[pos_index[w_pos_index]]).flatten()

            #记录距离超平面最近的几个被分类错误的点的下标, 维度不重要，选几个点由n_addition决定
            min_dis_index = [0,0,0]

            #没有足够的被分类错误的正类点和负类点，跳过
            if distance_n.size < n_addition and distance_p.size < n_addition:
                jump_cnt = jump_cnt + 1
                continue
            #没有足够的被分类错误的负类点，取n_addition个被分类错误的离超平面最近的正类点
            elif distance_n.size < n_addition:
                index = np.argsort(distance_p)
                min_dis_index = pos_index[w_pos_index[index[0:n_addition]]]
            #没有足够的被分类错误的正类点，取n_addition个被分类错误的离超平面最近的负类点
            elif distance_p.size < n_addition:
                index = np.argsort(distance_n)
                min_dis_index = neg_index[w_neg_index[index[0:n_addition]]]
            #被分类错误的正类点和负类点数量都充足时，取离超平面距离近的那一边的点
            else:
                index_n = np.argsort(distance_n)
                index_p = np.argsort(distance_p)
                min_dis_index_n = neg_index[w_neg_index[index_n[0:n_addition]]]
                min_dis_index_p = pos_index[w_pos_index[index_p[0:n_addition]]]
                if sum(distance[min_dis_index_n]) < sum(distance[min_dis_index_p]):
                    min_dis_index = min_dis_index_n
                else:
                    min_dis_index = min_dis_index_p

            #将离超平面最近的n_addition个分类错误的同类型点按照正确的分类加入训练集中得到follow-up input
            X_train_f = np.row_stack((X_train, X_test[min_dis_index]))
            if self.test_program.__name__ == 'sig_classification':
                y_train_f = np.append(y_train, abs(1-y_test[min_dis_index]))
            else:
                y_train_f = np.append(y_train, -y_test[min_dis_index])

            #测试集不变
            X_test_f = X_test
            y_test_f = y_test

            clf = self.fit(X_train_f, y_train_f)
            w_f = clf.coef_
            b_f = clf.intercept_
            err_f, pred_f, conf_f = self.test_program(
                w_f, b_f, X_test_f, y_test_f)

            #离超平面最近的几个点在超平面变化前后离超平面的距离
            w_f = mat(w_f).T
            distance = abs((X_test[min_dis_index]*w)+b)/np.linalg.norm(w)
            distance_f = abs((X_test[min_dis_index]*w_f+b_f))/np.linalg.norm(w_f)

            #这几个新加入的点离超平面的最小距离变近
            if (min(distance_f) <= min(distance)) or min(distance_f) > max_dis:
                err_cnt = err_cnt + 1
        print(err_cnt / self.itr_cnt)

    def MR9(self):
        print("Begin to test MR9...")
        err_cnt = 0
        # 未找到合适的点直接跳过了测试的次数
        jump_cnt = 0
        #选取分类错误的点的个数
        n_addition = 1
        for i in range(self.itr_cnt):
            X_train, y_train, X_test, y_test = self.create_dataset()
            clf = self.fit(X_train, y_train)
            w = clf.coef_
            b = clf.intercept_
            err, pred, conf = self.test_program(w, b, X_test, y_test)
            
            #----计算下标----
            #neg_index: 测试集中负类的下标, pos_index: 测试集中正类的下标
            #w_neg_index: 测试集中被分类错误的负类的下标, w_pos_index: 测试集中被分类错误的正类的下标
            if self.test_program.__name__ == 'sig_classification':
                neg_index = np.array(np.where(y_test == 0)).flatten()
                pos_index = np.array(np.where(y_test == 1)).flatten()
                w_neg_index = np.array(np.where(np.array(pred)[neg_index] != 0)).flatten()
                w_pos_index = np.array(np.where(np.array(pred)[pos_index] != 1)).flatten()
            else:
                neg_index = np.array(np.where(y_test == -1)).flatten()
                pos_index = np.array(np.where(y_test == 1)).flatten()
                w_neg_index = np.array(np.where(np.array(pred)[neg_index] != -1)).flatten()
                w_pos_index = np.array(np.where(np.array(pred)[pos_index] != 1)).flatten()

            #计算测试集中所有数据到超平面的距离
            w = mat(w).T
            distance = abs(X_test * w + b) / np.linalg.norm(w)

            #判断是否出现nan异常
            if all(np.isnan(distance)):
                err_cnt = err_cnt + 1
                continue

            #distance的上界
            max_dis = max(distance)

            #distance_n: 被分类错误的负类的点到超平面的距离
            #distance_p: 被分类错误的正类的点到超平面的距离
            distance_n = np.array(distance[neg_index[w_neg_index]]).flatten()
            distance_p = np.array(distance[pos_index[w_pos_index]]).flatten()

            #记录距离超平面最近的几个被分类错误的点的下标, 维度不重要，选几个点由n_addition决定
            min_dis_index = [0, 0, 0]

            #没有足够的被分类错误的正类点和负类点，跳过
            if distance_n.size < n_addition and distance_p.size < n_addition:
                jump_cnt = jump_cnt + 1
                continue
            #没有足够的被分类错误的负类点，取n_addition个被分类错误的离超平面最近的正类点
            elif distance_n.size < n_addition:
                index = np.argsort(distance_p)
                min_dis_index = pos_index[w_pos_index[index[0:n_addition]]]
            #没有足够的被分类错误的正类点，取n_addition个被分类错误的离超平面最近的负类点
            elif distance_p.size < n_addition:
                index = np.argsort(distance_n)
                min_dis_index = neg_index[w_neg_index[index[0:n_addition]]]
            #被分类错误的正类点和负类点数量都充足时，取离超平面距离近的那一边的点
            else:
                index_n = np.argsort(distance_n)
                index_p = np.argsort(distance_p)
                min_dis_index_n = neg_index[w_neg_index[index_n[0:n_addition]]]
                min_dis_index_p = pos_index[w_pos_index[index_p[0:n_addition]]]
                if sum(distance[min_dis_index_n]) < sum(distance[min_dis_index_p]):
                    min_dis_index = min_dis_index_n
                else:
                    min_dis_index = min_dis_index_p

            #将离超平面最近的n_addition个分类错误的同类型点按照正确的分类加入训练集中得到follow-up input
            X_train_f = np.row_stack((X_train, X_test[min_dis_index]))

            #将此时所有的训练样本的label反向
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
            

            #离超平面最近的几个点在超平面变化前后离超平面的距离
            w_f = mat(w_f).T
            distance = abs((X_test[min_dis_index] * w) + b) / np.linalg.norm(w)
            distance_f = abs((X_test[min_dis_index] * w_f + b_f)) / np.linalg.norm(w_f)

            #这几个新加入的点离超平面的最小距离变近，或者距离变得比max_dis还远，则出现错误
            if (min(distance_f) <= min(distance)) or min(distance_f) > max_dis:
                err_cnt = err_cnt + 1
        print(err_cnt / self.itr_cnt)