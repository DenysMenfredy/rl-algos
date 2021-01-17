import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
import tensorflow_probability as tfp

class PGN(Model):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()
        
        self.net = Sequential([
            tf.keras.layers.Input(shape=(input_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(n_actions, activation='softmax')
        ])
    
    def format_input(self, state):
        x = state
        if not tf.is_tensor(x):
            x = tf.convert_to_tensor(x)
            x = tf.expand_dims(x, axis=0)
        return x
    
    def call(self, x):
        # print(x)
        """ Receive a state batch and return the logits """
        x = self.format_input(x)
        # print(x)
        x = self.net(x)
        return x
    
    def take_action(self, state):
        logits = self.call(state)
        dist = tfp.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.numpy()[0], log_prob
    
    
    
# pgn = PGN(4, 2)
# state = [-1, 2, 2, 3]
# raw_logits = pgn(state)
# print("Raw logits: {}".format(raw_logits))
# normalized_logits = tf.nn.softmax(raw_logits)
# print('Normalized probabilities: {}\nProbabilities sum: {}'.format(normalized_logits, \
#                                                             tf.reduce_sum(normalized_logits)))
# action, log_prob = pgn.take_action(state)
# print("Action: {}\nLog prob: {}".format(action, log_prob))