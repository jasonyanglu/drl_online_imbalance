import numpy as np
from hoeffding_tree import MyHoeffdingTree
from HoeffdingTree.core.instance import Instance
from skmultiflow.trees import HoeffdingTree
from calc_measures import *
from sklearn.model_selection import train_test_split


# load_data = np.load('data/rotsp_abrupt.npz')
# data = load_data['data']
# label = load_data['label']
#
#
# neg_num = sum(label == -1)
# pos_num = sum(label == 1)
# neg_idx = np.nonzero(label == -1)[0]
# pos_idx = np.nonzero(label == 1)[0]
#
# b_data = data[np.r_[pos_idx[:10], neg_idx[:10]]]
# b_label = label[np.r_[pos_idx[:10], neg_idx[:10]]]
# b_label[b_label == -1] = 0
#
# ht = HoeffdingTree()
# ht.fit(b_data, b_label)
# pred = ht.predict(b_data)
# print(pred)

# f = MyHoeffdingTree(feature_dim=2)
# f.train(b_data, b_label)
# prob = f.predict_proba(b_data[neg_idx[:10]])
# print(prob)
# pred = f.predict(b_data[neg_idx[:10]])
# print(pred)

# from copy import deepcopy
# class A:
#     def __init__(self):
#         self.a = 1
#
# a = A()
# print(a.a)
# b = deepcopy(a)
# b.a = 2
# print(a.a)
# print(b.a)

from skmultiflow.data.sea_generator import SEAGenerator
stream = SEAGenerator(classification_function=2, random_state=112, balance_classes=False, noise_percentage=0.28)
stream.prepare_for_use()
data, label = stream.next_sample(1000)
train_data, valid_data, train_label, valid_label = train_test_split(data, label, test_size=0.33,
                                                                            random_state=1)
ht = HoeffdingTree()
ht.fit(train_data, train_label)
pred = ht.predict(valid_data)
auc = auc_measure(pred, valid_label)
print(sum(train_label==1))
print(sum(train_label==0))
print(auc)