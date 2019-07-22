import numpy as np
from calc_measures import *
from underbagging import *
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from itertools import combinations
import cvxpy
import time

import abc


class ChunkBase:

    def __init__(self):

        self.ensemble = []
        self.chunk_count = 0
        self.train_count = 0
        self.w = np.array([])
        self.buf_data = np.array([])
        self.buf_label = np.array([])

    def _predict_base(self, test_data, prob_output=False):

        if len(self.ensemble) == 0:
            pred = np.zeros(test_data.shape[0])
        else:
            pred = np.zeros([test_data.shape[0], len(self.ensemble)])
            for i in range(len(self.ensemble)):
                if prob_output:
                    pred[:, i] = self.ensemble[i].predict_proba(test_data)[:, 1]
                else:
                    pred[:, i] = self.ensemble[i].predict(test_data)

        return pred

    @abc.abstractmethod
    def _update_chunk(self, data, label):
        pass

    def update(self, single_data, single_label):

        pred = self.predict(single_data.reshape(1, -1))

        if self.buf_label.size < self.chunk_size:
            self.buf_data = np.r_[self.buf_data.reshape(-1, single_data.shape[0]), single_data.reshape(1, -1)]
            self.buf_label = np.r_[self.buf_label, single_label]
            self.train_count += 1

        if self.buf_label.size == self.chunk_size or self.train_count == self.data_num:
            print('Data ' + str(self.train_count) + ' / ' + str(self.data_num))
            self._update_chunk(self.buf_data, self.buf_label)
            self.chunk_count += 1
            self.buf_data = np.array([])
            self.buf_label = np.array([])

        return pred

    def predict(self, test_data):

        all_pred = np.sign(self._predict_base(test_data))
        if len(self.w) != 0:
            pred = np.sign(np.dot(all_pred, self.w))
        else:
            pred = all_pred

        return pred

    def calculate_err(self, all_pred, label):

        ensemble_size = all_pred.shape[1]
        err = np.zeros(ensemble_size)
        for i in range(ensemble_size):
            if self.err_func == 'gm':
                err[i] = 1 - gm_measure(all_pred[:, i], label)

            elif self.err_func == 'f1':
                err[i] = 1 - f1_score(label, all_pred[:, i])

        return err


class UnderBagging(ChunkBase):

    def __init__(self, data_num, r=0.5, chunk_size=1000):

        ChunkBase.__init__(self)

        self.data_num = data_num
        self.r = r
        self.stored_data = np.array([])
        self.stored_label = np.array([])
        self.chunk_size = chunk_size

        self.w = np.array([1])

    def _update_chunk(self, data, label):

        pos_idx = label == 1
        neg_idx = label == 0

        # accumulate the minority class samples
        self.stored_data = np.r_[self.stored_data.reshape(-1, data.shape[1]), data[pos_idx]]
        self.stored_label = np.r_[self.stored_label, label[pos_idx]]
        sampling_data = np.r_[self.stored_data, data[neg_idx]]
        sampling_label = np.r_[self.stored_label, label[neg_idx]]

        model = UnderBagging(r=self.r, sampling_class=0)
        model.train(sampling_data, label)

        # only one ensemble classifier is kept
        self.ensemble = list()
        self.ensemble.append(model)
        all_pred = np.sign(self._predict_base(data))

        if self.chunk_count > 1:
            pred = all_pred
        else:
            pred = np.zeros_like(label)

        pred = np.sign(pred)

        return pred


class REA(ChunkBase):

    def __init__(self, data_num, f=0.5, k=10, chunk_size=1000):

        ChunkBase.__init__(self)

        self.data_num = data_num
        self.f = f
        self.k = k
        self.stored_data = np.array([])
        self.stored_label = np.array([])
        self.chunk_size = chunk_size

    def _update_chunk(self, data, label):

        pos_idx = label == 1
        neg_idx = label == 0
        pos_num = sum(pos_idx)
        neg_num = sum(neg_idx)

        if pos_num > neg_num:
            min_class = 0
            gamma = neg_num / pos_num
        else:
            min_class = 1
            gamma = pos_num / neg_num

        if self.f > self.chunk_count * gamma:
            sampling_data = np.r_[self.stored_data.reshape(-1, data.shape[1]), data]
            sampling_label = np.r_[self.stored_label, label]

        else:
            nbrs = NearestNeighbors(n_neighbors=self.k).fit(data)
            _, nn_idx = nbrs.kneighbors(self.stored_data)
            delta = np.zeros_like(self.stored_label)
            for i in range(delta.size):
                delta[i] = sum([x in np.nonzero(label == min_class)[0] for x in nn_idx[i]])
            sort_idx = np.argsort(-delta)
            add_num = int((self.f - gamma) * label.size)
            sampling_data = np.r_[self.stored_data[sort_idx[:add_num]], data]
            sampling_label = np.r_[self.stored_label[sort_idx[:add_num]], label]

        model = DecisionTreeClassifier(max_leaf_nodes=10, min_samples_leaf=5, max_depth=5)
        model.fit(sampling_data, sampling_label)
        self.ensemble.append(model)
        all_pred = np.sign(self._predict_base(data, prob_output=True))
        all_pred[neg_idx] = 1 - all_pred[neg_idx]
        err = np.mean((1 - all_pred) ** 2, 0)
        self.w = np.log(1 / err)

        all_pred[neg_idx] = 1 - all_pred[neg_idx]
        all_pred -= 0.5

        if self.chunk_count > 1:
            pred = np.dot(all_pred[:, :-1], self.w[:-1])
        else:
            pred = np.zeros_like(label)

        pred = np.sign(pred)

        self.stored_data = np.r_[self.stored_data.reshape(-1, data.shape[1]), data[label == min_class]]
        self.stored_label = np.r_[self.stored_label, label[label == min_class]]

        return pred


class DFGWIS(ChunkBase):

    def __init__(self, fea_num, data_num, fea_group_num=50, w_lambda=0.5, bin_num=30, T=11, train_ratio=0.85,
                 chunk_size=1000):

        ChunkBase.__init__(self)

        self.fea_num = fea_num
        self.data_num = data_num
        self.fea_group_num = fea_group_num
        self.w_lambda = w_lambda
        self.bin_num = bin_num
        self.T = T
        self.train_ratio = train_ratio
        self.chunk_size = chunk_size

        self.stored_data = list()
        self.stored_label = list()
        self.s = 0
        self.pred = []

        self.fea_group_num = min(2 ** fea_num - 1, self.fea_group_num)

        all_comb = list()
        for i in range(fea_num):
            all_comb += combinations(range(fea_num), i + 1)

        self.fea_comb = list()
        rand_idx = np.random.permutation(len(all_comb))[:self.fea_group_num]
        for i in range(self.fea_group_num):
            self.fea_comb.append(all_comb[rand_idx[i]])

        self.ws = np.zeros(self.fea_group_num)

    def _train(self, data, label, delta):
        pos_idx = label == 1
        neg_idx = label == 0
        pos_num = sum(pos_idx)

        P_data_size = 0
        for i in range(self.s, self.chunk_count - 1):
            P_data_size += sum(self.stored_label[i] == 1)

        if pos_num + P_data_size > delta:
            self.s += 1

        P_data = np.array([]).reshape(-1, self.fea_num)
        ts = np.array([])

        for i in range(self.s, self.chunk_count):
            P_data = np.r_[P_data, self.stored_data[i][self.stored_label[i] == 1]]
            ts = np.r_[ts, i * np.np.ones(sum(self.stored_label[i] == 1))]

        N_data = data[neg_idx]

        pos_num = P_data.shape[0]
        neg_num = N_data.shape[0]
        rand_idx = np.random.permutation(pos_num)
        P_data = P_data[rand_idx]
        ts = ts[rand_idx].astype(int)
        N_data = N_data[np.random.permutation(neg_num)]

        pos_train_num = int(self.train_ratio * pos_num)
        neg_train_num = int(self.train_ratio * neg_num)
        train_data = np.r_[P_data[:pos_train_num], N_data[:neg_train_num]]
        train_label = np.r_[np.ones(pos_train_num), -np.ones(neg_train_num)]
        train_ts = ts[:pos_train_num]
        hold_data = np.r_[P_data[pos_train_num:], N_data[neg_train_num:]]
        hold_label = np.r_[np.ones(pos_num - pos_train_num), -np.ones(neg_num - neg_train_num)]

        hold_pred = np.zeros([hold_label.size, self.fea_group_num])
        self.ensemble = list()
        for fea_comb_i in range(self.fea_group_num):
            self._learnH(train_data[:, self.fea_comb[fea_comb_i]], train_label, train_ts)
        hold_pred = self._predict_base(hold_data)

        # print('learnH time: %f' % (time.time() - start_time))
        # start_time = time.time()

        # solve convex optimization problem
        c = np.ones_like(hold_label)
        c[hold_label == 1] = sum(hold_label == 0) / sum(hold_label == 1)

        w = cvxpy.Variable(self.fea_group_num)
        obj = cvxpy.Minimize(c.T * cvxpy.logistic(-cvxpy.mul_elemwise(hold_label, hold_pred * w)))
        constraints = [cvxpy.sum_entries(w) == 1, w >= 0]

        prob = cvxpy.Problem(obj, constraints)
        try:
            prob.solve()
        except cvxpy.error.SolverError:
            prob.solve(solver=cvxpy.CVXOPT)
        self.wd = np.array(w.value).squeeze()

        # print('optimization time: %f' % (time.time() - start_time))

    def _test(self, data, previous_data):

        pq = np.zeros(self.fea_num)
        for fea_i in range(self.fea_num):
            bin_min = min(np.r_[data[:, fea_i], previous_data[:, fea_i]])
            bin_max = max(np.r_[data[:, fea_i], previous_data[:, fea_i]])
            bin_gap = (bin_min - bin_max) / (self.bin_num - 1)

            p = np.zeros(self.bin_num)
            q = np.zeros(self.bin_num)
            for j in range(self.bin_num):
                if j + 1 != self.bin_num:
                    p[j] = sum(np.logical_and(data[:, fea_i] >= bin_min + j * bin_gap,
                                           data[:, fea_i] < bin_min + (j + 1) * bin_gap))
                    q[j] = sum(np.logical_and(previous_data[:, fea_i] >= bin_min + j * bin_gap,
                                           previous_data[:, fea_i] < bin_min + (j + 1) * bin_gap))
                else:
                    p[j] = sum(np.logical_and(data[:, fea_i] >= bin_min + j * bin_gap,
                                           data[:, fea_i] <= bin_max))
                    q[j] = sum(np.logical_and(previous_data[:, fea_i] >= bin_min + j * bin_gap,
                                           previous_data[:, fea_i] <= bin_max))

            p /= sum(p)
            q /= sum(q)
            pq[fea_i] = np.sqrt(sum((np.sqrt(p) - np.sqrt(q)) ** 2))

        for fea_comb_i in range(self.fea_group_num):
            fea_idx = np.array(self.fea_comb[fea_comb_i])
            self.ws[fea_comb_i] = 1 - (np.mean(pq[fea_idx])) / np.sqrt(2)
        pred = self._predict_base(data)

        alpha = self.w_lambda * self.ws + (1 - self.w_lambda) * self.wd
        return np.sign(np.dot(pred, alpha))

    def _importance_sampling(self, data, ts):

        t = max(ts)
        l = min(ts)
        data_num = data.shape[0]

        D = np.zeros(t + 1)
        u = np.zeros([t + 1, self.fea_num])
        v = np.zeros([t + 1, self.fea_num])
        for k in range(l, t + 1):
            D[k] = sum(ts == k)

            for j in range(data.shape[1]):
                u[k, j] = sum(data[ts == k, j] / sum(ts == k))
                v[k, j] = sum((data[ts == k, j] - u[k, j]) ** 2) / (sum(ts == k) - 1)

        w = np.zeros(data_num)
        for i in range(data_num):
            gamma = 1

            for j in range(data.shape[1]):
                k = ts[i]
                Dk = 1 / np.sqrt(2 * np.pi * v[k, j]) * np.exp(-(data[i, j] - u[k, j]) ** 2 / (2 * v[k, j]))
                Dt = 1 / np.sqrt(2 * np.pi * v[t, j]) * np.exp(-(data[i, j] - u[t, j]) ** 2 / (2 * v[t, j]))
                gamma *= Dk / Dt

            beta = 1 / (D[ts[i]] / D[t] * gamma)
            w[i] = 1 / (1 + np.exp(-(beta - 0.5)))

        w /= sum(w)
        return w

    def _learnH(self, data, label, ts):

        w = self._importance_sampling(data[label == 1], ts)
        model = UnderBagging(T=self.T, pos_weight=w, replace=True)
        model.train(data, label)
        self.ensemble.append(model)

    def _predict_base(self, test_data, prob_output=False):

        pred = np.zeros([test_data.shape[0], len(self.ensemble)])
        for i in range(len(self.ensemble)):
            if prob_output:
                pred[:, i] = self.ensemble[i].predict_proba(test_data[:, self.fea_comb[i]])[:, 1]
            else:
                pred[:, i] = self.ensemble[i].predict(test_data[:, self.fea_comb[i]])

        return pred

    def _update_chunk(self, data, label):

        pos_idx = label == 1
        neg_idx = label == 0
        pos_num = sum(pos_idx)
        neg_num = sum(neg_idx)

        self.stored_data.append(data)
        self.stored_label.append(label)

        if self.chunk_count > 1:
            self.pred = self._test(data, self.stored_data[self.chunk_count - 2])
        else:
            self.pred = np.zeros_like(label)

        self._train(data, label, delta=neg_num)

    def update(self, single_data, single_label):

        if self.buf_label.size < self.chunk_size:
            self.buf_data = np.r_[
                self.buf_data.reshape(-1, single_data.shape[0]), single_data.reshape(-1, single_data.shape[0])]
            self.buf_label = np.r_[self.buf_label, single_label]
            self.train_count += 1

        if self.buf_label.size == self.chunk_size or self.train_count == self.data_num:
            print('Data ' + str(self.train_count) + ' / ' + str(self.data_num))
            self._update_chunk(self.buf_data, self.buf_label)
            self.chunk_count += 1
            self.buf_data = np.array([])
            self.buf_label = np.array([])

        if len(self.pred) != 0:
            pred = self.pred
            self.pred = []
        else:
            pred = []

        return pred


class LearnppNIE(ChunkBase):

    def __init__(self, data_num, chunk_size, T=5, a=0.5, b=10, err_func='gm'):
        ChunkBase.__init__(self)

        self.T = T
        self.data_num = data_num
        self.chunk_size = chunk_size
        self.a = a
        self.b = b
        self.err_func = err_func
        self.beta = np.array([[0.0]])

    def _update_chunk(self, data, label):

        model = UnderBagging(T=self.T, auto_r=True)
        model.train(data, label)
        self.ensemble.append(model)
        all_pred = np.sign(self._predict_base(data))

        if self.chunk_count > 1:
            pred = np.dot(all_pred[:, :-1], self.w)
        else:
            pred = np.zeros_like(label)

        pred = np.sign(pred)

        err = self.calculate_err(all_pred, label)

        if err[-1] > 0.5:
            model = UnderBagging(T=self.T, auto_r=True)
            model.train(data, label)
            self.ensemble[-1] = model
            all_pred = np.sign(self._predict_base(data))
            err = self.calculate_err(all_pred, label)
            if err[-1] > 0.5:
                err[-1] = 0.5

        err[err > 0.5] = 0.5

        if self.chunk_count == 1:
            self.beta[0, 0] = err / (1 - err)
        else:
            self.beta = np.pad(self.beta, ((0, 1), (0, 1)), 'constant', constant_values=(0))
            self.beta[:self.chunk_count, self.chunk_count - 1] = err / (1 - err)

        self.w = np.zeros(self.chunk_count)
        for k in range(self.chunk_count):
            omega = np.array(range(1, self.chunk_count - k + 1))
            omega = 1 / (1 + np.exp(-self.a * (omega - self.b)))
            omega /= sum(omega)
            beta_hat = sum(omega * self.beta[k, k:self.chunk_count])
            self.w[k] = np.log(1 / beta_hat)

        return pred
