"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            action_dim,
            hidden_dim,
            feature_dim,
            learning_rate=0.01,
    ):
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.init_hidden = []
        self.lr = learning_rate

        self.ep_obs, self.ep_as, self.ep_hidden, self.ep_r = [], [], [], []
        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_init_hidden = tf.placeholder(tf.float32, [1, self.hidden_dim], name="initial_hidden")
            self.tf_hidden = tf.placeholder(tf.float32, [None, self.hidden_dim], name="hidden")
            self.tf_obs = tf.placeholder(tf.float32, [None, self.feature_dim], name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

        # gru
        gru = tf.nn.rnn_cell.GRUCell(self.hidden_dim)
        init_state = gru.zero_state(1, dtype=tf.float32)
        _, self.output_state = gru(self.tf_obs, self.tf_hidden)

        # sampler fc1
        sampler_layer = tf.layers.dense(
            inputs=self.output_state,
            units=10,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='sampler_fc1'
        )

        # sampler fc2
        sampler_all_act = tf.layers.dense(
            inputs=sampler_layer,
            units=self.action_dim,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='sampler_fc2'
        )

        # detector fc1
        detector_layer = tf.layers.dense(
            inputs=tf.concat([self.tf_init_hidden, self.tf_hidden], axis=1),
            units=10,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='detector_fc1'
        )

        # detector fc2
        detector_all_act = tf.layers.dense(
            inputs=detector_layer,
            units=self.action_dim,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='detector_fc2'
        )

        # action probabilities
        self.sampler_act_prob = tf.nn.softmax(sampler_all_act, name='sampler_all_act')
        self.detector_act_prob = tf.nn.softmax(detector_all_act, name='detector_all_act')

        with tf.name_scope('loss'):
            sampler_neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sampler_all_act,
                                                                                  labels=self.tf_acts)
            self.sampler_loss = tf.reduce_mean(sampler_neg_log_prob * self.tf_vt)

            detector_neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=detector_all_act,
                                                                                   labels=self.tf_acts)
            self.detector_loss = tf.reduce_mean(detector_neg_log_prob * self.tf_vt)

        with tf.name_scope('train'):
            self.sampler_train_op = tf.train.AdamOptimizer(self.lr).minimize(self.sampler_loss)
            self.detector_train_op = tf.train.AdamOptimizer(self.lr).minimize(self.detector_loss)

    def choose_sampler_action(self, observation, hidden):
        output_hidden, prob_weights = self.sess.run([self.output_state, self.sampler_act_prob],
                                                    feed_dict={self.tf_obs: observation,  # [1, feature_dim]
                                                               self.tf_hidden: hidden})  # [1, hidden_dim]
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action, output_hidden

    def choose_detector_action(self, hidden):
        prob_weights = self.sess.run(self.detector_act_prob, feed_dict={
            self.tf_init_hidden: self.init_hidden,  # [1, hidden_dim]
            self.tf_hidden: hidden})  # [1, hidden_dim]
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action

    def store_sampler_transition(self, s, a, hidden):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_hidden.append(hidden)

    def store_sampler_reward(self, reward, reward_len):
        self.ep_r.extend([reward] * reward_len)

    def learn_sampler(self):
        # train on episode

        _, loss = self.sess.run([self.sampler_train_op, self.sampler_loss], feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),  # [None, feature_dim]
            self.tf_acts: np.array(self.ep_as),  # [None, ]
            self.tf_hidden: np.vstack(self.ep_hidden),  # [None, hidden_dim]
            self.tf_vt: np.array(self.ep_r),  # [None, ]
        })

        self.ep_obs, self.ep_as, self.ep_hidden, self.ep_r = [], [], [], []
        return loss

    def learn_detector(self, act, reward, hidden):
        # train on episode
        _, loss = self.sess.run([self.detector_train_op, self.detector_loss], feed_dict={
            self.tf_init_hidden: self.init_hidden,  # [1, hidden_dim]
            self.tf_hidden: hidden,  # [1, hidden_dim]
            self.tf_acts: act,  # [1]
            self.tf_vt: reward,  # [1]
        })

        return loss
