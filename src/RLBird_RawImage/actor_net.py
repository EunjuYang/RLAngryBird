"""
    actor_net class
    Writer      : Eunjoo Yang
    Last Date   : 2017/05/15
    Reference from : https://github.com/stevenpjg/ddpg-aigym.git
"""
import numpy as np
import tensorflow as tf
import tfUtil as tfU
import math


LEARNING_RATE = 0.001
BATCH_SIZE = 10
TAU = 0.01
MODEL_PATH = "./model/actor_model.ckpt"

class ActorNet:
    """ Actor Network Model of DDPG Algorithm """

    def __init__(self, num_states, num_actions):
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()

            # actor network model parameters:
            self.w1_a, self.w2_a, self.w3_a, self.w4_a, self.w5_a, self.w6_a, \
            self.actor_state_in, self.actor_model = self.create_actor_net(num_states, num_actions)

            # target actor network model parameters:
            self.t_w1_a, self.t_w2_a, self.t_w3_a, self.t_w4_a, self.t_w5_a, self.t_w6_a, \
            self.t_actor_state_in, self.t_actor_model = self.create_actor_net(num_states, num_actions)

            # cost of actor network:
            self.q_gradient_input = tf.placeholder("float", [None,num_actions])  # gets input from action_gradient computed in critic network file
            self.actor_parameters = [self.w1_a, self.w2_a, self.w3_a, self.w4_a, self.w5_a, self.w6_a]
            self.parameters_gradients = tf.gradients(self.actor_model, self.actor_parameters,-self.q_gradient_input)  # /BATCH_SIZE) #set gradient as q_gradient_input * (d(actor_model)/d(actor_parameter))
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(
                zip(self.parameters_gradients, self.actor_parameters))
            # initialize all tensor variable parameters:
            self.sess.run(tf.initialize_all_variables())
            self.saver = tf.train.Saver()
            # To make sure actor and target have same intial parmameters copy the parameters:
            # copy target parameters
            self.sess.run([
                self.t_w1_a.assign(self.w1_a),
                self.t_w2_a.assign(self.w2_a),
                self.t_w3_a.assign(self.w3_a),
                self.t_w4_a.assign(self.w4_a),
                self.t_w5_a.assign(self.w5_a),
                self.t_w6_a.assign(self.w6_a)])
            #self.saver.restore(self.sess, MODEL_PATH)



    def return_actor_param(self):
        actor_parm = [self.w1_a, self.w2_a, self.w3_a, self.w4_a, self.w5_a, self.w6_a, self.actor_state_in, self.actor_model, \
                      self.t_w1_a, self.t_w2_a, self.t_w3_a, self.t_w4_a, self.t_w5_a, self.t_w6_a, \
                      self.t_actor_state_in, self.t_actor_model]
        return actor_parm

    def create_actor_net(self, num_states=480*840, num_actions=3):
        """ Network that takes states and return action """

        actor_state_in = tf.placeholder("float", [None, num_states[0], num_states[1], num_states[2]])

        # conv1 (?, 420, 240, 64)
        w1_a = tf.Variable(tf.random_normal([5, 5, 3, 64], stddev = 0.01))
        l1a = tf.nn.relu(tf.nn.conv2d(actor_state_in, w1_a, strides=[1, 1, 1, 1], padding='SAME'))
        l1p = tf.nn.max_pool(l1a, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        l1n = tf.nn.lrn(l1p, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)

        # conv2 (?, 210, 120, 64)
        w2_a = tf.Variable(tf.random_normal([5, 5, 64, 64], stddev = 0.01))
        l2a = tf.nn.relu(tf.nn.conv2d(l1n, w2_a, strides=[1, 1, 1, 1], padding='SAME'))
        l2p = tf.nn.max_pool(l2a, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        l2n = tf.nn.lrn(l2p, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)

        # conv3 (?, 105, 60, 64)
        w3_a = tf.Variable(tf.random_normal([5, 5, 64, 16], stddev = 0.01))
        l3a = tf.nn.relu(tf.nn.conv2d(l2n, w3_a, strides=[1, 1, 1, 1], padding='SAME'))
        l3p = tf.nn.max_pool(l3a, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        l3n = tf.nn.lrn(l3p, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)

        # fc1
        shape = l3n.get_shape().as_list()
        fc_shape = shape[1] * shape[2] * shape[3]

        # fc1
        l3 = tf.reshape(l3p, [-1, fc_shape])
        w4_a = tf.Variable(tf.random_normal([fc_shape, 384], stddev = 0.01))
        l4 = tf.nn.relu(tf.matmul(l3, w4_a))

        # fc2
        w5_a = tf.Variable(tf.random_normal([384, 192], mean = 0.1, stddev = 0.001))
        l5 = tf.nn.relu(tf.matmul(l4, w5_a))

        # fc3
        w6_a = tf.Variable(tf.random_normal([192, num_actions], mean = 0.1, stddev = 0.001))
        actor_model = tf.nn.relu(tf.matmul(l5, w6_a))

        return w1_a, w2_a, w3_a, w4_a, w5_a, w6_a, actor_state_in, actor_model

    def evaluate_actor(self, state_t):
        return self.sess.run(self.actor_model, feed_dict={self.actor_state_in: state_t})

    def evaluate_target_actor(self, state_t_1):
        return self.sess.run(self.t_actor_model, feed_dict={self.t_actor_state_in: state_t_1})

    def train_actor(self, actor_state_in, q_gradient_input):
        self.sess.run(self.optimizer,
                      feed_dict={self.actor_state_in: actor_state_in, self.q_gradient_input: q_gradient_input})

    def update_target_actor(self):
        self.sess.run([
            self.t_w1_a.assign(TAU * self.w1_a + (1 - TAU) * self.t_w1_a),
            self.t_w2_a.assign(TAU * self.w2_a + (1 - TAU) * self.t_w2_a),
            self.t_w3_a.assign(TAU * self.w3_a + (1 - TAU) * self.t_w3_a),
            self.t_w4_a.assign(TAU * self.w4_a + (1 - TAU) * self.t_w4_a),
            self.t_w5_a.assign(TAU * self.w5_a + (1 - TAU) * self.t_w5_a),
            self.t_w6_a.assign(TAU * self.w6_a + (1 - TAU) * self.t_w6_a)
        ])

    def save_actor(self, path = MODEL_PATH):
        save_path = self.saver.save(self.sess, path)

