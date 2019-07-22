import numpy as np

from HoeffdingTree.hoeffdingtree import HoeffdingTree
from HoeffdingTree.core.attribute import Attribute
from HoeffdingTree.core.dataset import Dataset
from HoeffdingTree.core.instance import Instance


class MyHoeffdingTree:

    def __init__(self, feature_dim):
        self.fea_num = feature_dim

        self.vfdt = HoeffdingTree()
        self.vfdt.set_grace_period(50)
        self.vfdt.set_hoeffding_tie_threshold(0.05)
        self.vfdt.set_split_confidence(0.0001)
        self.vfdt.set_minimum_fraction_of_weight_info_gain(0.01)
        self.train_count = 0

        self.attributes = []
        for i in range(feature_dim):
            self.attributes.append(Attribute(str(i), att_type='Numeric'))
        self.attributes.append(Attribute('Label', ['-1', '1'], att_type='Nominal'))

        self.dataset = Dataset(self.attributes, feature_dim)

    def train(self, data, label):
        if data.ndim == 1:
            data = data.reshape([1, -1])
        data_num = data.shape[0]
        for i in range(data_num):
            inst_values = list(np.r_[data[i], label[i]])
            inst_values[self.fea_num] = int(self.attributes[self.fea_num].index_of_value(str(int(label[i]))))
            self.dataset.add(Instance(att_values=inst_values))
        self.vfdt.build_classifier(self.dataset)
        self.train_count += data_num

    def update(self, data, label):
        if self.train_count == 0:
            raise Exception('Build classifier before updating')
        else:
            if data.ndim == 1:
                data = data.reshape([1, -1])
            data_num = data.shape[0]
            for i in range(data_num):
                inst_values = list(np.r_[data[i], label[i]])
                inst_values[self.fea_num] = int(self.attributes[self.fea_num].index_of_value(str(int(label[i]))))
                new_instance = Instance(att_values=inst_values)
                new_instance.set_dataset(self.dataset)
                self.vfdt.update_classifier(new_instance)
                self.train_count += 1

    def predict(self, data):
        if data.ndim == 1:
            data = data.reshape([1, -1])
        data_num = data.shape[0]
        pred = np.zeros(data_num)
        for i in range(data_num):
            inst_values = list(np.r_[data[i], 1])
            inst_values[self.fea_num] = int(self.dataset.attribute(self.fea_num).index_of_value(str(1)))
            new_instance = Instance(att_values=inst_values)
            new_instance.set_dataset(self.dataset)
            pred[i] = self.vfdt.distribution_for_instance(new_instance)[1]
            pred[i] = np.sign(pred[i] - 0.5)

        return pred

    def predict_proba(self, data):
        if data.ndim == 1:
            data = data.reshape([1, -1])
        data_num = data.shape[0]
        prob = np.zeros([data_num, 2])
        for i in range(data_num):
            inst_values = list(np.r_[data[i], 1])
            inst_values[self.fea_num] = int(self.dataset.attribute(self.fea_num).index_of_value(str(1)))
            new_instance = Instance(att_values=inst_values)
            new_instance.set_dataset(self.dataset)
            prob[i] = self.vfdt.distribution_for_instance(new_instance)

        return prob
