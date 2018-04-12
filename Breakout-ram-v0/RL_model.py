import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class RL_model(object):
    
    def __init__(self, input_units, hidden_units_1, hidden_units_2, output_units):
        self.input_units = input_units
        #self.hidden_units_1 = hidden_units_1
        #self.hidden_units_2 = hidden_units_2
        self.output_units = output_units
        
        #self.W0_init = tf.random_normal([self.input_units, self.hidden_units_1])-0.5
        #self.B0_init = tf.random_normal([1,self.hidden_units_1])-0.5
        #self.W1_init = tf.random_normal([self.hidden_units_1, self.hidden_units_2])-0.5
        #self.B1_init = tf.random_normal([1,self.hidden_units_2])-0.5
        #self.W2_init = tf.random_normal([self.hidden_units_2, self.output_units])-0.5
        #self.B2_init = tf.random_normal([1,self.output_units])-0.5
        #print("New model")
    
    
    def build_graph(self, learn_rate, model_name):
        
        # Data and expected rewards 
        self.X = tf.placeholder(tf.float32, shape=[None, self.input_units], name = str(model_name + '-X'))
        self.Y = tf.placeholder(tf.float32, name = str(model_name + '-Y'))
        #self.X = tf.placeholder(tf.float32, shape=[None, self.input_units, 1])
        
        
        # Model parameters (W - weights , B - Bias)
        #self.W0 = tf.Variable(self.W0_init)
        #self.B0 = tf.Variable(self.B0_init)
        #self.W1 = tf.Variable(self.W1_init)
        #self.B1 = tf.Variable(self.B1_init)
        #self.W2 = tf.Variable(self.W2_init)
        #self.B2 = tf.Variable(self.B2_init)

        # Optimization
        #self.normalize_0 = tf.nn.batch_normalization(self.X, 0, 1, 0, 1, 0.00001)
        #self.forward_1 = tf.add(tf.matmul(self.X, self.W0), self.B0)          # Forward from input to hidden (result is 1xh)
        #self.forward_1 = tf.add(tf.matmul(self.normalize_0, self.W0), self.B0)   # Forward from input to hidden (result is 1xh)
        #self.normalize_1 = tf.nn.batch_normalization(self.forward_1, 0, 1, 0, 1, 0.00001)
        #self.sigmoid_1 = tf.sigmoid(self.normalize_1)
        #self.forward_2 = tf.add(tf.matmul(self.sigmoid_1, self.W1), self.B1)   # Forward from hidden1 to hidden2
        #self.normalize_2 = tf.nn.batch_normalization(self.forward_2, 0, 1, 0, 1, 0.00001)
        #self.sigmoid_2 = tf.sigmoid(self.normalize_2)
        #self.predict = tf.add(tf.matmul(self.sigmoid_2, self.W2), self.B2)     # Forward hidden2 to output
        #self.loss = tf.losses.mean_squared_error(self.Y, self.predict)
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
        #self.grad_descent = self.optimizer.minimize(self.loss)

        # Optimization
        '''
        #self.norm_0 = tf.layers.batch_normalization(self.X)
        self.forward_1 = tf.layers.dense(self.X, self.hidden_units_1, activation=tf.nn.sigmoid)
        #self.norm_1 = tf.layers.batch_normalization(self.forward_1)
        self.forward_2 = tf.layers.dense(self.forward_1, self.hidden_units_2, activation=tf.nn.sigmoid)
        #self.norm_2 = tf.layers.batch_normalization(self.forward_2)
        self.predict = tf.layers.dense(self.forward_2, self.output_units)
        '''

        # With Con1D
        #self.conv_1 = tf.layers.conv1d(self.X, 16, kernel_size = 8, strides = 8, activation = tf.nn.relu)         # Convolutions along bits in a byte (64 params)
        #self.conv_2 = tf.layers.conv1d(self.conv_1, 16, kernel_size = 4, strides = 2, activation = tf.nn.relu)   # Convolutions from adjacent values
        
        #self.flat = tf.reshape(self.conv_2, [-1, 1])
        #self.fc_1 = tf.layers.dense(tf.reshape(self.conv_2, [-1,1]), 1, activation = tf.nn.relu)
        #self.fc_2 = tf.layers.dense(tf.reshape(self.fc_1, [1,1]), 1, activation = tf.nn.relu)
        #self.flat = tf.layers.flatten(self.conv_2)
        
        #self.fc_0 = tf.layers.dense(tf.layers.flatten(self.conv_1), 64 , activation = tf.nn.sigmoid)
        
        # Optimization
        self.fc_0 = tf.layers.dense(self.X, 128 , activation = tf.nn.relu, name = str(model_name + '-fc0'))
        self.fc_1 = tf.layers.dense(self.fc_0, 128 , activation = tf.nn.relu, name = str(model_name + '-fc1'))
        self.fc_2 = tf.layers.dense(self.fc_1, 128 , activation = tf.nn.relu, name = str(model_name + '-fc2'))
        self.predict = tf.layers.dense(self.fc_2, self.output_units, name = str(model_name + '-pred'))
        
        self.loss = tf.losses.mean_squared_error(self.Y, self.predict)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
        self.grad_descent = self.optimizer.minimize(self.loss)
