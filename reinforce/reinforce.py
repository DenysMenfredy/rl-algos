import numpy as np
import tensorflow as tf
from model import PGN

class Reinforce(object):
    def __init__(self, env):
        self.env = env
        self.num_obs = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.network = PGN(self.num_obs, self.num_actions)
        self.gamma = 0.99
        self.lr = 1e-4
        self.train_episodes = 4
        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        
    def calc_qvals(self, rewards):
        len_r = len(rewards)

        discounted_rewards = tf.pow(self.gamma, np.arange(len_r)) * rewards
        discounted_rewards /= max(discounted_rewards)
        return discounted_rewards
    
    def loss_fn(self, preds, r):
        return -1 * tf.reduce_sum(r * tf.math.log(preds))
    
    
    def take_action(self, state):
        pred = self.network(np.array(state))
        action = np.random.choice(np.array([0, 1]), p=pred.numpy()[0])
        return action
    
    