from RL_model import RL_model
from copy import deepcopy
import time
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
        self.learning_rate = 0.001
        self.input_layer_size = 4
        self.hidden_layer1_size = 24
        self.hidden_layer2_size = 48
        self.output_layer_size = 2
        
        # Q-Learning Paraeters
        self.epsilon = 1.0            # Exploration Rate
        self.epsilon_decay = 0.99
        self.min_epsilon = 0.0003     
        self.discount = 1.00          # Discount for future rewards
        self.step_delta = 100          # Steps to make before updating Q targets
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
            output = session.run(self.model.predict, feed_dict={self.model.X: input_matrix})
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
        output = session.run(self.target_model.predict, feed_dict={self.target_model.X: state})
        #output = session.run(self.model.predict, feed_dict={self.model.X: state})
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

    # Update weights for the current model
    def updateWeights(self):
        
        self.updateTargets()

        #Only update weights when the required batch size has been reached
        if(len(self.experience) > self.batch_size):
            x_batch = np.zeros((self.batch_size, self.input_layer_size))        # Initial states
            next_states = np.zeros((self.batch_size, self.input_layer_size))    # State reached
            actions = np.zeros(self.batch_size)                                 # Action taken in initial state
            rewards = np.zeros(self.batch_size)                                 # Immediate rewards observed
            done_arr = np.zeros(self.batch_size)                                # Values determining if the initial state is terminal
            
            # Get random samples from experience
            minibatch = random.sample(self.experience, self.batch_size)

            i = 0
            for state, action, reward, next_state, done in minibatch:
                x_batch[i] = state
                actions[i] = action
                rewards[i] = reward
                next_states[i] = next_state
                done_arr[i] = done
                i += 1
            
            y_batch = session.run(self.model.predict, feed_dict={self.model.X: x_batch})          # Predict Q values for state with current model
            future_vals = session.run(self.model.predict, feed_dict={self.model.X: next_states})   # Predict Q values for next_state with frozen model
            
            for i in range(self.batch_size):
                # Current state is terminal, so we add only observed reward
                if(done_arr[i]):
                    y_batch[i,int(actions[i])] = rewards[i]
                
                # Calculate value of current state with estimated optimal value of future state
                else:
                    y_batch[i,int(actions[i])] = rewards[i] + self.discount*np.max(future_vals[i])

            session.run(self.model.grad_descent, feed_dict={self.model.X: x_batch, self.model.Y: y_batch})

if __name__ == "__main__":
    
    threshold = 195.0                 # Value to achieve consecutively to solve de problem
    consecutive = 100                 # Number of consecutive times required to get above the threshold to solve problem
    scores = np.zeros(consecutive)    # Store scores for last episodes
    display_frequency = 20            # Amount of episodes before displaying information in the console
    start_time = time.time()
    
    solved = False
    i_episode = 0
    
    agent = Agent()
    session = tf.Session()
    session.run(agent.init_op)
    session.run(agent.updateW0)
    session.run(agent.updateB0)
    session.run(agent.updateW1)
    session.run(agent.updateB1)
    session.run(agent.updateW2)
    session.run(agent.updateB2)

    print("\nCartPole-v0")
    print("------------------------------------------------")
    print("Stats for last " , consecutive, " episodes")
    print("------------------------------------------------")

    while(not solved):
        
        state = agent.env.reset()
        time_step = 0
        
        while(True):    

            action = agent.epsilonGreedy(state)                                    # Pick action with epsilonGreedy policy
            next_state, reward, done, info = agent.env.step(action)
            agent.addExperience((state, action, reward, next_state, done))         # Store transition if it is not a losing state
            agent.updateWeights()                                                  # Update network weights
            state = next_state
            time_step += 1

            if done:
                i_episode += 1
                if(agent.epsilon > agent.min_epsilon):
                    agent.epsilon = agent.epsilon*agent.epsilon_decay
                scores[i_episode % consecutive] = time_step+1
                
                mean_score = np.mean(scores)
                
                
                if((i_episode+1>=consecutive and (i_episode+1) % display_frequency == 0) or mean_score>=threshold):
                    print("Episode ", i_episode+1 , "  | avg timesteps " , np.mean(scores))
                    if(mean_score >= threshold):
                        solved = True
                break
    print("Solved after " , i_episode+1 , " episodes")
    print("Time taken " , "{0:.2f}".format(time.time() - start_time) , " s")
