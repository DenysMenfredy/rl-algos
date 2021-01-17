import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential

class PGN(Model):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()
        
        self.net = Sequential([
            tf.keras.layers.Input(shape=(input_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(n_actions, activation=None)
        ])
    
    def call(self, x):
        x = tf.convert_to_tensor(x)
        x = tf.expand_dims(x, axis=0)
        return self.net(x)
    
    
    
# pgn = PGN(4, 3)
# pred = pgn(np.array([-1, 2, 2, 3]))
# print(pred)