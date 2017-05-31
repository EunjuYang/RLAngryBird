"""
    ddpg class
    Writer      : Eunjoo Yang
    Last Date   : 2017/05/12
    Reference from :
"""
import numpy as np
import random
from actor_net import ActorNet
from critic_net import CriticNet
from actor_net_bn import ActorNet_bn
from critic_net_bn import CriticNet_bn
from collections import deque
from tensorflow_grad_inverter import grad_inverter

REPLAY_MEMORY_SIZE = 10000
GAMMA = 0.99
is_grad_inverter = True



class DDPG:
    """ Deep Deterministic Policy Gradient Algorithm"""

    def __init__(self, num_states, num_actions, action_space_high, action_space_low, is_batch_norm, BATCH_SIZE):
        self.BATCH_SIZE = BATCH_SIZE
        self.num_states = num_states
        self.num_actions = num_actions
        self.action_space_high = action_space_high
        self.action_space_low = action_space_low
        self.is_batch_norm = is_batch_norm

        if is_batch_norm:
            self.critic_net = CriticNet_bn(self.num_states, self.num_actions, self.BATCH_SIZE)
            self.actor_net = ActorNet_bn(self.num_states, self.num_actions, self.BATCH_SIZE)

        else:
            self.critic_net = CriticNet(self.num_states, self.num_actions, self.BATCH_SIZE)
            self.actor_net = ActorNet(self.num_states, self.num_actions, self.BATCH_SIZE)

        # Initialize Buffer Network:
        self.replay_memory = deque()

        # Intialize time step:
        self.time_step = 0
        self.counter = 0

        action_max = np.array(self.action_space_high).tolist() #RL_bird: action configuration
        action_min = np.array(self.action_space_low).tolist()
        action_bounds = [action_max, action_min]
        self.grad_inv = grad_inverter(action_bounds)

    def evaluate_actor(self, state_t):
        return self.actor_net.evaluate_actor(state_t)

    def add_experience(self, observation_1, observation_2, action, reward, done):
        self.observation_1 = observation_1
        self.observation_2 = observation_2
        self.action = action
        self.reward = reward
        self.done = done
        self.replay_memory.append((self.observation_1, self.observation_2, self.action, self.reward, self.done))
        self.time_step = self.time_step + 1
        if (len(self.replay_memory) > REPLAY_MEMORY_SIZE):
            self.replay_memory.popleft()

    def minibatches(self):
        batch = random.sample(self.replay_memory, self.BATCH_SIZE)
        # state t
        self.state_t_batch = [item[0] for item in batch]
        self.state_t_batch = np.array(self.state_t_batch)
        # state t+1
        self.state_t_1_batch = [item[1] for item in batch]
        self.state_t_1_batch = np.array(self.state_t_1_batch)
        self.action_batch = [item[2] for item in batch]
        self.action_batch = np.array(self.action_batch)
        self.action_batch = np.reshape(self.action_batch, [len(self.action_batch), self.num_actions])
        self.reward_batch = [item[3] for item in batch]
        self.reward_batch = np.array(self.reward_batch)
        self.done_batch = [item[4] for item in batch]
        self.done_batch = np.array(self.done_batch)

    def train(self):
        # sample a random minibatch of N transitions from R
        self.minibatches()
        self.action_t_1_batch = self.actor_net.evaluate_target_actor(self.state_t_1_batch)
        # Q'(s_i+1,a_i+1)
        q_t_1 = self.critic_net.evaluate_target_critic(self.state_t_1_batch, self.action_t_1_batch)
        self.y_i_batch = []
        for i in range(0, self.BATCH_SIZE):

            if self.done_batch[i]:
                self.y_i_batch.append(self.reward_batch[i])
            else:

                self.y_i_batch.append(self.reward_batch[i] + GAMMA * q_t_1[i][0])

        self.y_i_batch = np.array(self.y_i_batch)
        self.y_i_batch = np.reshape(self.y_i_batch, [len(self.y_i_batch), 1])

        # Update critic by minimizing the loss
        self.critic_net.train_critic(self.state_t_batch, self.action_batch, self.y_i_batch) #weight update which direction to minize (y_i_batch - y value from (state_t_batch,action_batch))

        # Update actor proportional to the gradients:
        action_for_delQ = self.evaluate_actor(self.state_t_batch)

        if is_grad_inverter:
            self.del_Q_a = self.critic_net.compute_delQ_a(self.state_t_batch,
                                                          action_for_delQ)  # /BATCH_SIZE
            self.del_Q_a = self.grad_inv.invert(self.del_Q_a, action_for_delQ)
        else:
            self.del_Q_a = self.critic_net.compute_delQ_a(self.state_t_batch, action_for_delQ)[0]  # /BATCH_SIZE

        # train actor network proportional to delQ/dela and del_Actor_model/del_actor_parameters:
        self.actor_net.train_actor(self.state_t_batch, self.del_Q_a)

        # Update target Critic and actor network
        self.critic_net.update_target_critic()
        self.actor_net.update_target_actor()

        #save Parameters
        self.actor_net.save_actor()
        self.critic_net.save_critic()
        print "###### finish to train"


