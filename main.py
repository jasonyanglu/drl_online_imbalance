import sys
import numpy as np

from chunk_based_methods import *
from calc_measures import *
from ReLOL import ReLOL

from skmultiflow.data.sea_generator import SEAGenerator

# reproducible
rand_seed = 1
np.random.seed(rand_seed)

chunk_size = 1000

# data_name = sys.argv[1]

# data_name = 'rotsp_abrupt.npz'
# load_data = np.load('data/' + data_name)
# data = load_data['data']
# label = load_data['label']
# reset_pos = load_data['reset_pos'].astype(int)
# data_num, feature_dim = data.shape

stream = SEAGenerator(classification_function=2, random_state=112, balance_classes=False, noise_percentage=0.28)
stream.prepare_for_use()
feature_dim = stream.n_features
chunk_num = 1
data_num = chunk_num * chunk_size
reset_pos = np.array([0])


run_num = 1

pq_result_relol = [{} for _ in range(run_num)]
auc_result_relol = [{} for _ in range(run_num)]

for run_i in range(run_num):

    model_relol = ReLOL(data_num=data_num, feature_dim=feature_dim, max_episode=1000, learning_rate=0.001)
    pred_relol = np.array([])
    cum_label = np.array([])
    for chunk_i in range(chunk_num):
        print('Round ' + str(run_i))
        data, label = stream.next_sample(chunk_size)
        for i in range(chunk_size):
            pred_relol = np.append(pred_relol, model_relol.update(data[i], label[i]))
            cum_label = np.append(cum_label, label[i])
    # pq_result_relol[run_i] = prequential_measure(pred_relol, cum_label, reset_pos)
    auc_result_relol[run_i] = auc_measure(pred_relol[chunk_size:], cum_label[chunk_size:])

print(auc_result_relol)
# print('ReLOL: %f' % np.mean([pq_result_relol[i]['gm'][-1] for i in range(run_num)]))
