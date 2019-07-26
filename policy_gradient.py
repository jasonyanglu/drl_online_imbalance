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
            seq_len,
            learning_rate
    ):
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.lr = learning_rate
        self.init_hidden = []

        self.pretrain_lr = 0.01

        self.ep_obs, self.ep_as, self.ep_r, self.ep_hidden = [], [], [], []
        self._build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_init_hidden = tf.placeholder(tf.float32, [None, self.hidden_dim], name="initial_hidden")
            self.tf_obs = tf.placeholder(tf.float32, [None, self.seq_len, self.feature_dim], name="observations")
            self.tf_sampler_acts = tf.placeholder(tf.int32, [None, self.seq_len], name="sampler_actions")
            self.tf_act_probs = tf.placeholder(tf.float32, [None, self.seq_len, 2], name="sampler_action_probs")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

        # gru
        gru = tf.nn.rnn_cell.GRUCell(self.hidden_dim)
        # self.tf_tile_init_hidden = tf.tile(self.tf_init_hidden, [tf.shape(self.tf_obs)[0], 1])
        outputs, self.output_state = tf.nn.dynamic_rnn(gru, self.tf_obs, initial_state=self.tf_init_hidden,
                                                       time_major=False)

        # sampler fc1
        sampler_layer = tf.layers.dense(
            inputs=outputs,
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

        # action probabilities
        self.sampler_act_prob = tf.nn.softmax(sampler_all_act, name='sampler_all_act')

        with tf.name_scope('loss'):
            sampler_neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=sampler_all_act,
                                                                                  labels=self.tf_sampler_acts)
            self.sampler_loss = tf.reduce_mean(tf.multiply(sampler_neg_log_prob, tf.expand_dims(self.tf_vt, 1)))

        with tf.name_scope('train'):
            self.sampler_train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.sampler_loss)

        with tf.name_scope('pretrain'):
            self.sampler_pretrain_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=sampler_all_act,
                                                                                    labels=self.tf_act_probs))
            self.sampler_pretrain_op = tf.train.RMSPropOptimizer(self.pretrain_lr).minimize(self.sampler_pretrain_loss)

    def choose_sampler_action(self, observation, hidden):
        prob_weights, output_state = self.sess.run([self.sampler_act_prob, self.output_state],
                                                   feed_dict={self.tf_obs: observation,  # [1, seq_len, feature_dim]
                                                              self.tf_init_hidden: hidden})  # [1, hidden_dim]
        return prob_weights, output_state

    def store_sampler_transition(self, s, a, r, h):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_r.append(r)
        self.ep_hidden.append(h)

    def learn_sampler(self):
        # train on episode
        _, loss = self.sess.run([self.sampler_train_op, self.sampler_loss], feed_dict={
            self.tf_obs: np.vstack(self.ep_obs),  # [None, seq_len, feature_dim]
            self.tf_init_hidden: np.vstack(self.ep_hidden),  # [None, hidden_dim]
            self.tf_sampler_acts: np.array(self.ep_as),  # [None, ]
            self.tf_vt: np.array(self.ep_r),  # [None, ]
        })

        self.ep_obs, self.ep_as, self.ep_r, self.ep_hidden = [], [], [], []
        return loss

    def learn_pretrain_sampler(self, observation, act_prob, hidden):
        _, loss = self.sess.run([self.sampler_pretrain_op, self.sampler_pretrain_loss], feed_dict={
            self.tf_obs: observation,  # [1, seq_len, feature_dim]
            self.tf_init_hidden: hidden,  # [1, hidden_dim]
            self.tf_act_probs: act_prob,  # [1, seq_len, 2]
        })

        return loss

