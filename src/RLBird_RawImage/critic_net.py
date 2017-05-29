"""
    critic_net class
    Writer      : Eunjoo Yang
    Last Date   : 2017/05/15
    Reference from : https://github.com/stevenpjg/ddpg-aigym.git
"""

import numpy as np
import tensorflow as tf
import tfUtil as tfU
import math

TAU = 0.001
LEARNING_RATE = 0.005
BATCH_SIZE = 10
MODEL_PATH = "./model/critic_model.ckpt"


class CriticNet:
    """ Critic Q value model of the DDPG algorithm """

    def __init__(self, num_states, num_actions):
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()

            # critic_q_model parameters:
            self.w1_c, self.w2_c, self.w3_c, self.w4_c, self.w5_c, self.w5_action_c, self.w6_c, \
            self.critic_q_model, self.critic_state_in, self.critic_action_in = self.create_critic_net(num_states,num_actions)

            # create target_q_model:
            self.t_w1_c, self.t_w2_c, self.t_w3_c, self.t_w4_c, self.t_w5_c, self.t_w5_action_c, self.t_w6_c, \
            self.t_critic_q_model, self.t_critic_state_in, self.t_critic_action_in = self.create_critic_net(num_states,num_actions)

            self.q_value_in = tf.placeholder("float", [None, 1])  # supervisor
            # self.l2_regularizer_loss = tf.nn.l2_loss(self.W1_c)+tf.nn.l2_loss(self.W2_c)+ tf.nn.l2_loss(self.W2_action_c) + tf.nn.l2_loss(self.W3_c)+tf.nn.l2_loss(self.B1_c)+tf.nn.l2_loss(self.B2_c)+tf.nn.l2_loss(self.B3_c)

            self.cost = tf.pow(self.critic_q_model - self.q_value_in, 2) / BATCH_SIZE
            self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost)

            # action gradient to be used in actor network:
            # self.action_gradients=tf.gradients(self.critic_q_model,self.critic_action_in)
            # from simple actor net:
            self.act_grad_v = tf.gradients(self.critic_q_model, self.critic_action_in)
            self.action_gradients = [
                self.act_grad_v[0] / tf.to_float(tf.shape(self.act_grad_v[0])[0])]
                # this is just divided by batch size
            # from simple actor net:
            self.check_fl = self.action_gradients
            self.saver = tf.train.Saver()

            # initialize all tensor variable parameters:

            self.sess.run(tf.initialize_all_variables())

            # To make sure critic and target have same parmameters copy the parameters:
            # copy target parameters
            self.sess.run([
                self.t_w1_c.assign(self.w1_c),
                self.t_w2_c.assign(self.w2_c),
                self.t_w3_c.assign(self.w3_c),
                self.t_w4_c.assign(self.w4_c),
                self.t_w5_c.assign(self.w5_c),
                self.t_w6_c.assign(self.w6_c),
                self.t_w5_action_c.assign(self.w5_action_c)
            ])
            #self.saver.restore(self.sess, MODEL_PATH)

    def return_critic_param(self):

        critic_param = [self.w1_c, self.w2_c, self.w3_c, self.w4_c, self.w5_c, self.w5_action_c, self.w6_c, \
                      self.critic_q_model, self.critic_state_in, self.critic_action_in, \
                      self.t_w1_c, self.t_w2_c, self.t_w3_c, self.t_w4_c, self.t_w5_c, self.t_w5_action_c, self.t_w6_c, \
                      self.t_critic_q_model, self.t_critic_state_in, self.t_critic_action_in]

        return critic_param

    def create_critic_net(self, num_states=4, num_actions=1):
        N_HIDDEN_1 = 192
        N_HIDDEN_2 = 32
        critic_state_in = tf.placeholder("float", [None, num_states[0], num_states[1], num_states[2]])
        critic_action_in = tf.placeholder("float", [None, num_actions])

        total_num_states = num_states[0] * num_states[1] * num_states[2]
        # conv1 (?, 420, 240, 64)
        w1_c = tf.Variable(tf.random_normal([5, 5, 3, 64],  mean = 0.01, stddev = 0.01))
        l1a = tf.nn.relu(tf.nn.conv2d(critic_state_in, w1_c, strides=[1, 1, 1, 1], padding='SAME'))
        l1p = tf.nn.max_pool(l1a, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        l1n = tf.nn.lrn(l1p, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)

        # conv2 (?, 210, 120, 32)
        w2_c = tf.Variable(tf.random_normal([5, 5, 64, 32], mean = 0.001, stddev = 0.003))
        l2a = tf.nn.relu(tf.nn.conv2d(l1n, w2_c, strides=[1, 1, 1, 1], padding='SAME'))
        l2p = tf.nn.max_pool(l2a, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        l2n = tf.nn.lrn(l2p, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)

        # conv3 (?, 105, 60, 3)
        w3_c = tf.Variable(tf.random_normal([5, 5, 32, 3], mean=0.001, stddev = 0.003))
        l3a = tf.nn.relu(tf.nn.conv2d(l2n, w3_c, strides=[1, 1, 1, 1], padding='SAME'))
        l3p = tf.nn.max_pool(l3a, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        shape = l3p.get_shape().as_list()

        fc_shape = shape[1] * shape[2] * shape[3]

        # fc1
        l3 = tf.reshape(l3p, [-1, fc_shape])
        w4_c = tf.Variable(tf.random_normal([fc_shape, N_HIDDEN_1], mean = 0, stddev = 0.001))
        l4 = tf.nn.relu(tf.matmul(l3, w4_c))

        # fc2
        w5_c = tf.Variable(tf.random_normal([N_HIDDEN_1, N_HIDDEN_2], mean = 0.1, stddev = 0.003))
        w5_action_c = tf.Variable(tf.random_normal([num_actions, N_HIDDEN_2], mean = 0,stddev = 0.001))
        l5 = tf.nn.relu(tf.matmul(l4, w5_c)) + tf.matmul(critic_action_in, w5_action_c)

        #
        w6_c = tf.Variable(tf.random_normal([N_HIDDEN_2,1], mean = 0, stddev = 0.5))
        critic_q_model = tf.abs(tf.matmul(l5, w6_c))

        return w1_c, w2_c, w3_c, w4_c, w5_c, w5_action_c, w6_c, critic_q_model, critic_state_in, critic_action_in

    def train_critic(self, state_t_batch, action_batch, y_i_batch):
        self.sess.run(self.optimizer,
                      feed_dict={self.critic_state_in: state_t_batch, self.critic_action_in: action_batch,
                                 self.q_value_in: y_i_batch})

    def evaluate_target_critic(self, state_t_1, action_t_1):
        return self.sess.run(self.t_critic_q_model,
                             feed_dict={self.t_critic_state_in: state_t_1, self.t_critic_action_in: action_t_1})

    def compute_delQ_a(self, state_t, action_t):
        #        print '\n'
        #        print 'check grad number'
        #        ch= self.sess.run(self.check_fl, feed_dict={self.critic_state_in: state_t,self.critic_action_in: action_t})
        #        print len(ch)
        #        print len(ch[0])
        #        raw_input("Press Enter to continue...")
        return self.sess.run(self.action_gradients,
                             feed_dict={self.critic_state_in: state_t, self.critic_action_in: action_t})

    def update_target_critic(self):
        self.sess.run([
            self.t_w1_c.assign(TAU * self.w1_c + (1 - TAU) * self.t_w1_c),
            self.t_w2_c.assign(TAU * self.w2_c + (1 - TAU) * self.t_w2_c),
            self.t_w3_c.assign(TAU * self.w3_c + (1 - TAU) * self.t_w3_c),
            self.t_w4_c.assign(TAU * self.w4_c + (1 - TAU) * self.t_w4_c),
            self.t_w5_c.assign(TAU * self.w5_c + (1 - TAU) * self.t_w5_c),
            self.t_w6_c.assign(TAU * self.w6_c + (1 - TAU) * self.t_w6_c),
            self.t_w5_action_c.assign(TAU * self.w5_action_c + (1 - TAU) * self.t_w5_action_c)
        ])

    def save_critic(self, path = MODEL_PATH):
        save_path = self.saver.save(self.sess, path)
