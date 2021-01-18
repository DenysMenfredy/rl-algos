import numpy as np
import tensorflow as tf
from model import PGN
import tensorflow_probability as tfp

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
        discounted_rewards = []
        sum_reward = 0
        for reward in reversed(rewards):
            sum_reward = reward + self.gamma * sum_reward
            discounted_rewards.append(sum_reward)
        return discounted_rewards.reverse()
    
    def loss_fn(self, prob, action, reward):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        return -log_prob * reward # -Q(s, a) * logPI(a/s)
    
    
    def update(self, states, actions, rewards):
        # rewards = self.discounted_rewards(rewards)
        
        sum_reward = 0
        discounted_rewards = []
        for reward in reversed(rewards):
            sum_reward = reward + self.gamma * sum_reward
            discounted_rewards.append(sum_reward)
        discounted_rewards.reverse()
        
        for state, action, reward in zip(states, actions, rewards):
            with tf.GradientTape() as tape:
                prob = self.network([state], training=True)
                loss = self.loss_fn(prob, action, reward)
                grads = tape.gradient(loss, self.network.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.network.trainable_variables))
    
    def train(self, max_episodes, max_steps):
        scores = []
        for ep in range(max_episodes):
            saved_log_probs = []
            states = []
            actions = []
            rewards = []
            state = self.env.reset()
            for t in range(max_steps):
                action = self.network.take_action(state)
                # print(action)
                # saved_log_probs.append(log_prob)
                next_state, reward, done, _ = self.env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state
                if done: 
                    break
            scores.append(sum(rewards))
            self.update(states, actions, rewards)
            
            # policy_loss = []
            
            if ep % self.print_every == 0:
                print("Episode {}\tAverage Score: {:.2f}".format(ep, np.mean(scores)))
            if np.mean(scores) >= 195.0:
                print("Environment solved in {} episodes!\tAverage Score: {:.2f}".format(ep, np.mean(scores)))
                break
        return scores
    
    
    def play(self, episodes=100, steps=200):
        for ep in range(episodes):
            self.env.render()
            state = self.env.reset()
            for t in range(steps):
                action = self.network.take_action(state)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                if done: break
        
        

