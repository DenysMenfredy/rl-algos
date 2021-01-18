import numpy as np
import gym
from reinforce import Reinforce
import tensorflow as tf
import matplotlib.pyplot as plt

MAX_EPISODES = 5000
MAX_STEPS = 200

def plot_graphic(data):
    plt.plot(np.arange(len(data)), data)
    plt.title("Reward per episode")
    plt.xlabel("Episode")
    plt.show()
    
    

def main():
    env = gym.make("CartPole-v0")
    agent = Reinforce(env)
    scores = agent.train(MAX_EPISODES, MAX_STEPS)
    print(len(scores))
    plot_graphic(scores)
    


if __name__ == '__main__':
    main()