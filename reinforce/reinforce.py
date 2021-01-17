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
        self.lr = 1e-3
        self.train_episodes = 4
        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        self.print_every = 10
        
    def discounted_rewards(self, rewards):
        discounts = [self.gamma ** i for i in range(len(rewards) + 1)]
        r_discounted = sum([a * b for a, b in zip(discounts, rewards)])
        return r_discounted
    
    def loss_fn(self, preds, r):
        return -1 * tf.reduce_sum(r * tf.math.log(preds))
    
    
    def train(self, max_episodes, max_steps):
        scores = []
        for ep in range(max_episodes):
            saved_log_probs = []
            rewards = []
            state = self.env.reset()
            for t in range(max_steps):
                action, log_prob = self.network.take_action(state)
                # print(action)
                saved_log_probs.append(log_prob)
                state, reward, done, _ = self.env.step(action)
                rewards.append(reward)
                if done: break
            scores.append(sum(rewards))
            discounted_rewards = self.discounted_rewards(rewards)
            policy_loss = []
            with tf.GradientTape() as tape:
                for log_prob in saved_log_probs:
                    policy_loss.append(-log_prob * discounted_rewards)
                policy_loss = tf.reduce_sum(tf.concat(policy_loss, axis=0))
            grads = tape.gradient(policy_loss, self.network.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
            
            if ep % self.print_every == 0:
                print("Episode {}\tAverage Score: {:.2f}".format(ep, np.mean(scores)))
            if np.mean(scores) >= 195.0:
                print("Environment solved in {} episodes!\tAverage Score: {:.2f}".format(ep, np.mean(scores)))
                break
        return scores
        
        

