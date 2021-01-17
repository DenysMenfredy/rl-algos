import gym
from reinforce import Reinforce
import tensorflow as tf

MAX_EPISODES = 500
MAX_STEPS = 200

def main():
    env = gym.make("CartPole-v0")
    agent = Reinforce(env)
    score = []
    
    for ep in range(MAX_EPISODES):
        curr_state = agent.env.reset()
        done= False
        transitions = []
        for t in range(MAX_STEPS):
            action = agent.take_action(curr_state)
            prev_state = curr_state
            curr_state, _, done, _ = agent.env.step(action)
            transitions.append((prev_state, action, t+1))
            if done:
                break
            ep_len = len(transitions)
            score.append(ep_len)
            reward_batch = tf.reverse(tf.convert_to_tensor([r for (s, a, r) in transitions]), axis=0)
            discounted_rewards = agent.calc_qvals(reward_batch)
            state_batch = tf.convert_to_tensor([s for (s, a, r) in transitions])
            action_batch = tf.convert_to_tensor([a for (s, a, r) in transitions])
            pred_batch = agent.take_action(state_batch)
            prob_batch = tf.gather(pred_batch, axis=1, indices=action_batch)
            loss = agent.loss_fn(prob_batch, discounted_rewards)
            with tf.GradientTape as tape:
                grads = tape.gradient(loss, agent.network.variables)
                agent.optimizer.apply_gradients(grads, agent.network.variables)
                



if __name__ == '__main__':
    main()