import tensorflow as tf
import numpy as np
import math

from chunk_based_methods import ChunkBase
from sklearn.model_selection import train_test_split
from policy_gradient import PolicyGradient
from skmultiflow.trees import HoeffdingTree
from calc_measures import *
from copy import deepcopy
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt

# reproducible
rand_seed = 1
np.random.seed(rand_seed)
tf.set_random_seed(rand_seed)


class ReLOL(ChunkBase):

    def __init__(self, data_num, feature_dim, chunk_size=1000, max_episode=1000, rollout_num=1000, hidden_dim=10,
                 metric='auc', learning_rate=0.0001, test_prop=0.33, pretrain_episode=100, ensemble_size=11):

        ChunkBase.__init__(self)

        self.data_num = data_num
        self.feature_dim = feature_dim
        self.chunk_size = chunk_size
        self.max_episode = max_episode
        self.rollout_num = rollout_num
        self.hidden_dim = hidden_dim
        self.metric = metric
        self.test_prop = test_prop
        self.pretrain_episode = pretrain_episode
        self.ensemble_size = ensemble_size
        self.hidden = [np.zeros([1, hidden_dim])] * ensemble_size

        self.seq_len = round(chunk_size * (1 - test_prop))

        self.pg = PolicyGradient(
            action_dim=2,
            hidden_dim=self.hidden_dim,
            feature_dim=self.feature_dim + 1,
            seq_len=self.seq_len,
            learning_rate=learning_rate
        )

        for i in range(ensemble_size):
            self.ensemble.append(HoeffdingTree())

    def _update_chunk(self, data, label):

        train_data, valid_data, train_label, valid_label = train_test_split(data, label, test_size=self.test_prop,
                                                                            random_state=rand_seed)
        train_num = train_label.size
        assert train_num == self.seq_len
        hist_reward = []
        hist_sampler_loss = []

        # pretrain
        if sum(train_label == 1) > sum(train_label == 0):
            majority_class = 1
            minority_class = 0
        else:
            majority_class = 0
            minority_class = 1
        minority_ratio = sum(train_label == minority_class) / sum(train_label == majority_class)
        action_prob = np.zeros([1, train_num, 2])
        action_prob[0, train_label == majority_class] = [1 - minority_ratio, minority_ratio]
        action_prob[0, train_label == minority_class] = [minority_ratio, 1 - minority_ratio]

        for episode_i in range(self.pretrain_episode):
            # rollout negative samples
            observation = np.expand_dims(np.c_[train_data, train_label], 0)
            for ensemble_i in range(self.ensemble_size):
                pretrain_loss = self.pg.learn_pretrain_sampler(observation, action_prob, self.hidden[ensemble_i])
                print('pretrain episode: %d  pretrain loss: %.4f  ' % (episode_i, pretrain_loss))

        # RL episodes
        for episode_i in range(self.max_episode):
            print('episode: %d  ' % episode_i, end='')

            train_data, valid_data, train_label, valid_label = train_test_split(data, label, test_size=self.test_prop,
                                                                                random_state=rand_seed)

            for ensemble_i in range(self.ensemble_size):

                # rollout samples
                observation = np.expand_dims(np.c_[train_data, train_label], 0)
                sampler_action_prob, _ = self.pg.choose_sampler_action(observation, self.hidden[ensemble_i])
                sampler_action_prob = np.squeeze(sampler_action_prob)
                rollout_rewards = []
                rollout_sampler_num = []

                for rollout_i in range(self.rollout_num):
                    sampler_action = (np.random.rand(train_num) < sampler_action_prob[:, 1]).astype(int)
                    sampled_train_data = train_data[sampler_action == 1]
                    sampled_train_label = train_label[sampler_action == 1]

                    copy_model = deepcopy(self.ensemble[ensemble_i])
                    if self.chunk_count == 0:
                        copy_model.fit(sampled_train_data, sampled_train_label)
                    else:
                        copy_model.partial_fit(sampled_train_data, sampled_train_label)

                    # evaluate
                    valid_pred = copy_model.predict(valid_data)
                    if self.metric == 'auc':
                        reward = auc_measure(valid_pred, valid_label)
                    elif self.metric == 'f1':
                        reward = f1_measure(valid_pred, valid_label)
                    elif self.metric == 'gm':
                        reward = gm_measure(valid_pred, valid_label)
                    else:
                        raise Exception('Unknown metric')

                    self.pg.store_sampler_transition(observation, sampler_action, reward, self.hidden[ensemble_i])

                    rollout_rewards.append(reward)
                    rollout_sampler_num.append(sampled_train_label.size)

            # update policy network
            sampler_loss = self.pg.learn_sampler()

            # print('#selection: %d  ' % sum(sampled_train_label == 0), end='')
            print('#selection: %d  ' % np.mean(rollout_sampler_num), end='')
            print('reward: %.4f  ' % np.mean(rollout_rewards))

            hist_reward.append(np.mean(rollout_rewards))
            hist_sampler_loss.append(sampler_loss)

        # draw
        plt.plot(hist_reward, label='reward')
        plt.plot(hist_sampler_loss, label='sampler_loss')
        plt.legend()
        plt.show()

        # sampling for training
        for ensemble_i in range(self.ensemble_size):
            observation = np.expand_dims(np.c_[data, label], 0)
            sampler_action_prob, output_state = self.pg.choose_sampler_action(observation, self.hidden[ensemble_i])
            sampler_action = np.random.choice(range(sampler_action_prob.shape[1]), p=sampler_action_prob.ravel())
            sampled_data = data[sampler_action == 1]
            sampled_label = label[sampler_action == 1]

            if self.chunk_count == 0:
                self.ensemble[ensemble_i].fit(sampled_data, sampled_label)
            else:
                self.ensemble[ensemble_i].partial_fit(sampled_data, sampled_label)
            self.hidden[ensemble_i] = output_state
