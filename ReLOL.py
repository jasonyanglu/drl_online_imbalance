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
                 metric='auc', learning_rate=0.0001):

        ChunkBase.__init__(self)

        self.data_num = data_num
        self.feature_dim = feature_dim
        self.chunk_size = chunk_size
        self.max_episode = max_episode
        self.rollout_num = rollout_num
        self.hidden_dim = hidden_dim
        self.hidden = np.zeros([1, hidden_dim])
        self.metric = metric

        self.pg = PolicyGradient(
            action_dim=2,
            hidden_dim=self.hidden_dim,
            feature_dim=self.feature_dim + 1,
            learning_rate=learning_rate
        )

        self.ensemble.append(HoeffdingTree())
        self.ros = RandomOverSampler(random_state=rand_seed)

    def _update_chunk(self, data, label):

        data_num = label.size
        self.pg.init_hidden = self.hidden

        train_data, valid_data, train_label, valid_label = train_test_split(data, label, test_size=0.33,
                                                                            random_state=rand_seed)

        hist_reward = []
        hist_sampler_loss = []

        # RL episodes
        for episode_i in range(self.max_episode):
            print('episode: %d  ' % episode_i, end='')
            rollout_rewards = []
            rollout_sampler_num = []
            for rollout_i in range(self.rollout_num):
                # split training and validation set
                sampled_train_data, sampled_train_label = [], []
                self.hidden = self.pg.init_hidden

                # rollout negative samples
                for data_i in range(train_label.size):
                    observation = np.r_[train_data[data_i], train_label[data_i]].reshape([1, -1])
                    sampler_action, output_hidden = self.pg.choose_sampler_action(observation, self.hidden)
                    # if train_label[data_i] == 0:
                    self.pg.store_sampler_transition(observation, sampler_action, self.hidden)
                    # if sampler_action or train_label[data_i] == 1:
                    if sampler_action:
                        # store sampled train data
                        sampled_train_data.append(train_data[data_i])
                        sampled_train_label.append(train_label[data_i])
                        self.hidden = output_hidden

                # incremental train on sampled training data
                sampled_train_data = np.vstack(sampled_train_data)
                sampled_train_label = np.hstack(sampled_train_label)
                # oversampled_train_data, oversampled_train_label = self.ros.fit_sample(sampled_train_data,
                #                                                                       sampled_train_label)

                oversampled_train_data = sampled_train_data
                oversampled_train_label = sampled_train_label

                copy_model = deepcopy(self.ensemble[0])
                if self.chunk_count == 0:
                    copy_model.fit(oversampled_train_data, oversampled_train_label)
                else:
                    copy_model.partial_fit(oversampled_train_data, oversampled_train_label)

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

                self.pg.store_sampler_reward(reward, train_label.size)

                rollout_rewards.append(reward)
                rollout_sampler_num.append(sampled_train_label.size)

            # update policy network
            sampler_loss = self.pg.learn_sampler()

            # choose detector action
            detector_action = self.pg.choose_detector_action(self.hidden)
            detector_loss = self.pg.learn_detector(np.array([detector_action]), np.array([reward]), self.hidden)

            # print('#selection: %d  ' % sum(sampled_train_label == 0), end='')
            print('#selection: %d  ' % np.mean(rollout_sampler_num), end='')
            print('reward: %.4f  ' % np.mean(rollout_rewards))

            hist_reward.append(reward)
            hist_sampler_loss.append(sampler_loss)

        # incremental train
        sampled_data, sampled_label = [], []
        self.hidden = self.pg.init_hidden

        # # draw
        # plt.plot(hist_reward, label='reward')
        # plt.plot(hist_sampler_loss, label='sampler_loss')
        # plt.legend()
        # plt.show()

        # rollout
        for data_i in range(data_num):
            observation = np.r_[data[data_i], label[data_i]].reshape([1, -1])
            sampler_action, output_hidden = self.pg.choose_sampler_action(observation, self.hidden)
            # if sampler_action or label[data_i] == 1:
            if sampler_action:
                # store sampled train data
                sampled_data.append(data[data_i])
                sampled_label.append(label[data_i])
                self.hidden = output_hidden

        sampled_data = np.vstack(sampled_data)
        sampled_label = np.hstack(sampled_label)

        # oversampled_data, oversampled_label = self.ros.fit_sample(sampled_data, sampled_label)
        oversampled_data = sampled_data
        oversampled_label = sampled_label

        # if detector_action:
        #     # rebuild classifier
        #     print('rebuild classifier!')
        #     self.ensemble[0] = HoeffdingTree()
        #     self.ensemble[0].fit(oversampled_data, oversampled_label)
        # else:
        self.ensemble[0].partial_fit(oversampled_data, oversampled_label)


class ReLOL_DDPG(ChunkBase):

    def __init__(self, data_num, feature_dim, chunk_size=1000, max_episode=1000, rollout_num=10, hidden_dim=10,
                 metric='auc',
                 learning_rate=0.0001):

        ChunkBase.__init__(self)

        self.data_num = data_num
        self.feature_dim = feature_dim
        self.chunk_size = chunk_size
        self.max_episode = max_episode
        self.rollout_num = rollout_num
        self.hidden_dim = hidden_dim
        self.hidden = np.zeros([1, hidden_dim])
        self.metric = metric

        self.pg = PolicyGradient(
            action_dim=2,
            hidden_dim=self.hidden_dim,
            feature_dim=self.feature_dim + 1,
            learning_rate=learning_rate
        )

        self.ensemble.append(HoeffdingTree())
        self.ros = RandomOverSampler(random_state=rand_seed)

    def _update_chunk(self, data, label):

        data_num = label.size
        self.pg.init_hidden = self.hidden

        train_data, valid_data, train_label, valid_label = train_test_split(data, label, test_size=0.33,
                                                                            random_state=rand_seed)

        hist_reward = []
        hist_sampler_loss = []

        # RL episodes
        for episode_i in range(self.max_episode):
            print('episode: %d  ' % episode_i, end='')
            for rollout_i in range(self.rollout_num):
                # split training and validation set
                sampled_train_data, sampled_train_label = [], []
                self.hidden = self.pg.init_hidden

                # rollout negative samples
                for data_i in range(train_label.size):
                    observation = np.r_[train_data[data_i], train_label[data_i]].reshape([1, -1])
                    sampler_action, output_hidden = self.pg.choose_sampler_action(observation, self.hidden)
                    # if train_label[data_i] == 0:
                    self.pg.store_sampler_transition(observation, sampler_action, self.hidden)
                    # if sampler_action or train_label[data_i] == 1:
                    if sampler_action:
                        # store sampled train data
                        sampled_train_data.append(train_data[data_i])
                        sampled_train_label.append(train_label[data_i])
                        self.hidden = output_hidden

                # incremental train on sampled training data
                sampled_train_data = np.vstack(sampled_train_data)
                sampled_train_label = np.hstack(sampled_train_label)
                # oversampled_train_data, oversampled_train_label = self.ros.fit_sample(sampled_train_data,
                #                                                                       sampled_train_label)

                oversampled_train_data = sampled_train_data
                oversampled_train_label = sampled_train_label

                copy_model = deepcopy(self.ensemble[0])
                if self.chunk_count == 0:
                    copy_model.fit(oversampled_train_data, oversampled_train_label)
                else:
                    copy_model.partial_fit(oversampled_train_data, oversampled_train_label)

                # evaluate sampler
                valid_pred = copy_model.predict(valid_data)
                if self.metric == 'auc':
                    reward = auc_measure(valid_pred, valid_label) - 0.5
                elif self.metric == 'f1':
                    reward = f1_measure(valid_pred, valid_label)
                elif self.metric == 'gm':
                    reward = gm_measure(valid_pred, valid_label)
                else:
                    raise Exception('Unknown metric')

                self.pg.store_sampler_reward(reward, train_label.size)

            # update policy network
            sampler_loss = self.pg.learn_sampler()

            # choose detector action
            detector_action = self.pg.choose_detector_action(self.hidden)
            detector_loss = self.pg.learn_detector(np.array([detector_action]), np.array([reward]), self.hidden)

            # evaluate detector

            # print('#selection: %d  ' % sum(sampled_train_label == 0), end='')
            print('#selection: %d  ' % sampled_train_label.size, end='')
            print('reward: %.4f  ' % reward, end='')
            print('sampler_loss: %.4f  ' % sampler_loss, end='')
            print('detector_loss: %.4f' % detector_loss)

            hist_reward.append(reward)
            hist_sampler_loss.append(sampler_loss)

        # incremental train
        sampled_data, sampled_label = [], []
        self.hidden = self.pg.init_hidden

        # # draw
        # plt.plot(hist_reward, label='reward')
        # plt.plot(hist_sampler_loss, label='sampler_loss')
        # plt.legend()
        # plt.show()

        # rollout
        for data_i in range(data_num):
            observation = np.r_[data[data_i], label[data_i]].reshape([1, -1])
            sampler_action, output_hidden = self.pg.choose_sampler_action(observation, self.hidden)
            # if sampler_action or label[data_i] == 1:
            if sampler_action:
                # store sampled train data
                sampled_data.append(data[data_i])
                sampled_label.append(label[data_i])
                self.hidden = output_hidden

        sampled_data = np.vstack(sampled_data)
        sampled_label = np.hstack(sampled_label)

        # oversampled_data, oversampled_label = self.ros.fit_sample(sampled_data, sampled_label)
        oversampled_data = sampled_data
        oversampled_label = sampled_label

        # if detector_action:
        #     # rebuild classifier
        #     print('rebuild classifier!')
        #     self.ensemble[0] = HoeffdingTree()
        #     self.ensemble[0].fit(oversampled_data, oversampled_label)
        # else:
        self.ensemble[0].partial_fit(oversampled_data, oversampled_label)
