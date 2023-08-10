#%%
import tensorflow as tf
from tensorflow.keras import layers
from logging import raiseExceptions

'''
Some custom activation function
'''
class Act_swish(layers.Layer):
    '''
    Swish activation
    '''
    def __init__(self, b=1.0, learnable=False, **kwargs):
        self.b = b
        self.learnable = learnable
        super(Act_swish, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.learnable:
            # self.bb = tf.Variable(initial_value=self.b, 
            #            trainable = True, 
                    #    dtype=tf.float32, name='scale_b')
            self.bb = self.add_weight(
                       shape=(1,),
                       initializer = tf.initializers.Ones(),
                       constraint = lambda x: tf.clip_by_value(x, 1e-2, 10),
                       trainable = True,
                       name='scale_b'
            )
    def call(self, x):
        if self.learnable:
            return x*tf.nn.sigmoid(self.bb*x)
        else:
            return x*tf.nn.sigmoid(self.b*x)

class Act_tanh(layers.Layer):
    '''
    tanh activation
    '''
    def __init__(self, b=1.0, learnable=False, **kwargs):
        self.b = b
        self.learnable = learnable
        super(Act_tanh, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.learnable:
            # self.bb = tf.Variable(initial_value=self.b, 
            #            trainable = True, 
                    #    dtype=tf.float32, name='scale_b')
            self.bb = self.add_weight(
                       shape=(1,),
                       initializer = tf.initializers.Ones(),
                       constraint = lambda x: tf.clip_by_value(x, 0.8, 1.2),
                       trainable = True,
                       name='scale_b'
            )
    def call(self, x):
        if self.learnable:
            return self.bb*tf.nn.tanh(x)
        else:
            return self.b*tf.nn.tanh(x)

