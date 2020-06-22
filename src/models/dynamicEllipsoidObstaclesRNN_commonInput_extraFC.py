"""
Trajectory prediction neural network that considers an arbitrary number of quadrotors and constant-speed ellipsoidal obstacles 
Stacks dynamic obstacles and quadrotors together before they get processed by the bidirectional layer which learns their influence on the query robot.
"""

import tensorflow as tf
import models.dynamicEllipsoidObstaclesRNN_commonInput as mod

tfk = tf.keras
tfkm = tf.keras.models
tfkl = tf.keras.layers
tfkr = tf.keras.regularizers

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class FullModel(mod.FullModel):
    def __init__(self, args):
        super().__init__(args)
        self.fc_extra = tfkl.Dense(args.size_fc_layer, activation = 'relu')
        
    def call(self, x):
        x1 = self.state_encoder(x["query_input"])
        x2 = self.common_encoder( x )
        concat = self.concat([x1, x2])
        concat = self.fc_extra(concat)
        repeated = self.repeat(concat)
        out = self.decoder(repeated)
        return out
    