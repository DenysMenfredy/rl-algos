import gym
from reinforce import Reinforce
import tensorflow as tf

MAX_EPISODES = 500
MAX_STEPS = 200

def main():
    env = gym.make("CartPole-v0")
    agent = Reinforce(env)
    scores = agent.train(2000, 200)
    


if __name__ == '__main__':
    main()