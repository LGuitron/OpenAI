from Transition import Transition
from RL_model import RL_model
from copy import deepcopy
import tensorflow as tf
import numpy as np
import random
import gym

'''
Global Variables
'''


class Agent():

    def __init__(self):
        
        self.env = gym.make('CartPole-v0')
        
        # Store transitions for Deep Q Learning
        self.experience = []
        self.experience_limit = 100000
        self.current_index = 0

        # Network parameters
        self.batch_size = 16
        self.learning_rate = 0.01
        self.input_layer_size = 4
        self.hidden_layer1_size = 12
        self.hidden_layer2_size = 24
        self.output_layer_size = 2
        
        # Q-Learning Paraeters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        self.discount = 1.00
        self.step_delta = 50          # Steps to make before updating Q targets
        self.current_step_delta = 0   # Steps taken since last update

        # Network for decision making
        self.model = RL_model(self.input_layer_size, self.hidden_layer1_size, self.hidden_layer2_size, self.output_layer_size)
        self.model.build_graph(self.learning_rate)

        # Target network (fixed Q updates)
        self.target_model = RL_model(self.input_layer_size, self.hidden_layer1_size, self.hidden_layer2_size, self.output_layer_size)
        self.target_model.build_graph(self.learning_rate)

        self.init_op = tf.global_variables_initializer()        # Weight initialization
        
        # Parameter updates
        self.updateW0 = self.target_model.W0.assign(self.model.W0)
        self.updateB0 = self.target_model.B0.assign(self.model.B0)
        self.updateW1 = self.target_model.W1.assign(self.model.W1)
        self.updateB1 = self.target_model.B1.assign(self.model.B1)
        self.updateW2 = self.target_model.W2.assign(self.model.W2)
        self.updateB2 = self.target_model.B2.assign(self.model.B2)
    '''
    Helper Functions
    '''

    #Select action with epsilon greedy policy
    def epsilonGreedy(self, state):

        #Act Greedly
        if(random.random() > self.epsilon):
            input_matrix = np.zeros((1,self.input_layer_size))
            input_matrix[0,:] = state
            output = session.run(self.model.forward_3, feed_dict={self.model.X: input_matrix})
            return int(output[0,1] > output[0,0])
        
        #Act Randomly
        else:
            return self.env.action_space.sample()

    # Add experience
    def addExperience(self, transition):
        if(len(self.experience)<self.experience_limit):
            self.experience.append(transition)
        else:
            self.experience[self.current_index] = transition
            current_index = (self.current_index+1)%self.experience_limit

    #Get optimal Actions to be performed on S' states according to the frozen model
    def optimalFutureActions(self, state):
        #output = session.run(self.target_model.forward_3, feed_dict={self.target_model.X: state})
        output = session.run(self.model.forward_3, feed_dict={self.model.X: state})
        return output

    # Update Q targets after a number of steps
    def updateTargets(self):
        if(self.current_step_delta == self.step_delta):
            session.run(self.updateW0)
            session.run(self.updateB0)
            session.run(self.updateW1)
            session.run(self.updateB1)
            session.run(self.updateW2)
            session.run(self.updateB2)
            self.current_step_delta=0
        self.current_step_delta += 1


    def preprocess_state(self, state):
        return np.reshape(state, [1, 4])

    # Update weights for the current model
    def updateWeights(self):
        
        self.updateTargets()

        #Only update weights when the required batch size has been reached
        if(len(self.experience) > self.batch_size):
            x_batch = []
            y_batch = []
            minibatch = random.sample(self.experience, self.batch_size)
            for state, action, reward, next_state, done in minibatch:
                
                y_target = session.run(self.model.forward_3, feed_dict={self.model.X: self.preprocess_state(state)})
                if(done):
                    y_target[0][action] = reward
                else: 
                    future_vals = self.optimalFutureActions(self.preprocess_state(next_state))
                    y_target[0][action] = reward + self.discount * np.max(future_vals)
                x_batch.append(state)
                y_batch.append(y_target[0])
            session.run(self.model.grad_descent, feed_dict={self.model.X: np.array(x_batch), self.model.Y: np.array(y_batch)})
            
            #session.run(self.model.grad_descent, feed_dict={self.model.X: states1, self.model.Y: target_forward})
            #loss = session.run(self.model.loss, feed_dict={self.model.X: states1, self.model.Y: target_forward})
            #if (loss>10E+12):
            #    print("ERROR")
            #    print(a)

if __name__ == "__main__":
    agent = Agent()
    session = tf.Session()
    session.run(agent.init_op)
    session.run(agent.updateW0)
    session.run(agent.updateB0)
    session.run(agent.updateW1)
    session.run(agent.updateB1)
    session.run(agent.updateW2)
    session.run(agent.updateB2)

    for i_episode in range(10000):
        state = agent.env.reset()
        for t in range(100):
            
            #agent.env.render()
            action = agent.epsilonGreedy(state)                                    # Pick action with epsilonGreedy policy
            next_state, reward, done, info = agent.env.step(action)
            agent.addExperience((state, action, reward, next_state, done))         # Store transition if it is not a losing state
            agent.updateWeights()                                                  # Update network weights
            state = next_state
            
            if done:
                if(agent.epsilon > agent.min_epsilon):
                    agent.epsilon = agent.epsilon*agent.epsilon_decay
                print("Episode " + str(i_episode) + " finished after {} timesteps".format(t+1))
                break
