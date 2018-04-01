import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class RL_model(object):

    def __init__(self, input_units, hidden_units_1, hidden_units_2, output_units):
        self.input_units = input_units
        self.hidden_units_1 = hidden_units_1
        self.hidden_units_2 = hidden_units_2
        self.output_units = output_units
        
        self.W0_init = tf.random_uniform([self.input_units, self.hidden_units_1])-0.5
        self.B0_init = tf.random_uniform([1,self.hidden_units_1])-0.5
        self.W1_init = tf.random_uniform([self.hidden_units_1, self.hidden_units_2])-0.5
        self.B1_init = tf.random_uniform([1,self.hidden_units_2])-0.5
        self.W2_init = tf.random_uniform([self.hidden_units_2, self.output_units])-0.5
        self.B2_init = tf.random_uniform([1,self.output_units])-0.5
        
    def build_graph(self, learn_rate):
        
        # Data and expected rewards 
        self.X = tf.placeholder(tf.float32)
        self.Y = tf.placeholder(tf.float32)
        
        # Model parameters (W - weights , B - Bias)
        self.W0 = tf.Variable(self.W0_init)
        self.B0 = tf.Variable(self.B0_init)
        self.W1 = tf.Variable(self.W1_init)
        self.B1 = tf.Variable(self.B1_init)
        self.W2 = tf.Variable(self.W2_init)
        self.B2 = tf.Variable(self.B2_init)

        # Optimization
        self.forward_1 = tf.add(tf.matmul(self.X, self.W0), self.B0)        # Forward from input to hidden (result is 1xh)
        self.sigmoid_1 = tf.sigmoid(self.forward_1)
        self.forward_2 = tf.add(tf.matmul(self.sigmoid_1, self.W1), self.B1)   # Forward from hidden1 to hidden2
        self.sigmoid_2 = tf.sigmoid(self.forward_2)
        self.predict = tf.add(tf.matmul(self.sigmoid_2, self.W2), self.B2)   # Forward hidden2 to output
        
        self.loss = tf.losses.mean_squared_error(self.Y, self.predict)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
        self.grad_descent = self.optimizer.minimize(self.loss)
