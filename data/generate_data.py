from skmultiflow.data.sea_generator import SEAGenerator
import numpy as np

rand_seed = 1
np.random.seed(rand_seed)
data_num = 10000
reset_pos = [0, 5000]
ir = init_ir = 0.1
ir_drift_pos = [3300, 6600]
ir_drift_region = ir_drift_pos[1] - ir_drift_pos[0]

stream = SEAGenerator(classification_function=2, random_state=rand_seed, balance_classes=False, noise_percentage=0.1)
stream.prepare_for_use()

data, label = [], []

for data_i in range(data_num):
    # adjust ir
    if ir_drift_pos[0] < data_i <= ir_drift_pos[1]:
        ir += (1 - 2 * init_ir) / ir_drift_region

    # adjust drift
    if data_i == reset_pos[1]:
        stream = SEAGenerator(classification_function=3, random_state=rand_seed, balance_classes=False, noise_percentage=0.1)
        stream.prepare_for_use()

    if np.random.rand() < ir:  # pos sample
        while 1:
            single_data, single_label = stream.next_sample()
            if single_label == 1:
                data.append(single_data)
                label.append(single_label)
                break
    else:  # neg sample
        while 1:
            single_data, single_label = stream.next_sample()
            if single_label == 0:
                data.append(single_data)
                label.append(single_label)
                break

data = np.vstack(data)
label = np.hstack(label)

np.savez('sea_gradual.npz', data=data, label=label, reset_pos=reset_pos)
