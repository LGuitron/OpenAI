from RL_model import RL_model
from Prep_Input import Prep_Input
from copy import copy
import tensorflow as tf
import numpy as np
import random
import gym
import pickle

class Agent():

    def __init__(self, environment = None, hidden_units_1 = 0, hidden_units_2 = 0, preprocess = Prep_Input.identity, save_dir = "savedAgent"):
        
        # Gym environment
        self.env = gym.make(environment)
        
        # Input preprocessing method to use
        self.preprocess = preprocess

        # Memory required to store state representation
        self.state_memory = len(self.env.observation_space.high)
        
        # Store transitions for Deep Q Learning
        self.experience_limit = 1000000
        # Each experience has a size of (2 x len(state) | 1 x reward | 1 x action | 1 x done)
        self.experience = np.zeros((self.experience_limit, 2 * self.state_memory  + 3 ))
        
        self.current_index = 0

        # Network topology
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.input_units = self.preprocess(self.env.reset().reshape((1,-1))).shape[1]
        self.hidden_layer1_size = hidden_units_1
        self.hidden_layer2_size = hidden_units_2
        self.output_layer_size = self.env.action_space.n
        
        # Network for decision making
        self.model = RL_model(self.input_units, self.hidden_layer1_size, self.hidden_layer2_size, self.output_layer_size)
        self.model.build_graph(self.learning_rate, "model")

        # Target network (fixed Q updates)
        self.target_model = RL_model(self.input_units, self.hidden_layer1_size, self.hidden_layer2_size, self.output_layer_size)
        self.target_model.build_graph(self.learning_rate, "target")  
        
        # Q-Learning Parameters
        self.epsilon = 1.0            # Exploration Rate
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.05    
        self.discount = 0.99          # Discount for future rewards
        self.step_delta = 40000       # Steps to make before updating Q targets
        self.current_step_delta = 0   # Steps taken since last update
        
        self.w_update_frequency = 32  # Number of steps to perform before updating network weights
        self.w_update_step = 0        # Number of steps since last weight update

        # Tensorflow session
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())     # Weight initialization

        self.updateTargets()                                    # Set the target model parameters equal to the prediction model parameters
        
        # Operation for saving tensorflow models (policy and target networks)
        self.save_dir = save_dir
        self.saver = tf.train.Saver()
    
    
    # Define how an Agent's instance will be pickled
    def __getstate__(self):
        state = self.__dict__.copy()
        
        # Remove the unpicklable entries.
        del state['saver']
        del state['session']
        del state['model']
        del state['target_model']
        del state['experience']
        return state
    
    
    
    '''
    Training function
    '''
    
    #def train(self, threshold, consecutive, display_frequency, scores, i_episode, render_game):
    def train(self, max_episodes = 10000, display_frequency = 100, save = True):        

        avg_score = 0
        step_count = 0

        for i in range(max_episodes):
            state = self.env.reset()
            score = 0
            done = False
            while(not done):
                action = self.epsilonGreedy(state)                                     # Pick action with epsilonGreedy policy
                next_state, reward, done, info = self.env.step(action)
                score += reward
                self.addExperience(np.concatenate(
                                  (state.reshape(-1), np.full(1,action), np.full(1,reward), next_state.reshape(-1), np.full(1,done))
                                  ))
                self.updateWeights()                                                   
                state = next_state
                step_count += 1
                
            #if done:
            if(self.epsilon > self.min_epsilon):
                self.epsilon = self.epsilon*self.epsilon_decay
            avg_score += score
            if((i+1) % display_frequency == 0):
                print("Episode ", i+1 ," | Step " , step_count, "  | Rew " , avg_score/display_frequency, " | Epsilon "  , str(round(self.epsilon, 3)))
                avg_score = 0
        
        # Save progress after training session
        if(save):
            self.save()

    #Select action with epsilon greedy policy
    def epsilonGreedy(self, state):
        #Act Greedly
        if(random.random() > self.epsilon):
            output = self.session.run(self.model.predict, feed_dict={self.model.X: self.preprocess(state.reshape((1,-1)))})
            return np.argmax(output)
        
        #Act Randomly
        else:
            return self.env.action_space.sample()

    # Add experience
    def addExperience(self, transition):
        self.experience[self.current_index % self.experience_limit] = transition
        self.current_index += 1

    # Update Q targets after a number of steps
    def updateTargets(self):
        print("Updated Target model")
        self.target_model = copy(self.model)
        self.current_step_delta=0

    # Update weights for the current model
    def updateWeights(self):
        
        if(self.current_step_delta == self.step_delta):
            self.updateTargets()
        else:
            self.current_step_delta += 1

        # Make updates after specified number of steps only
        if(self.w_update_step + 1 == self.w_update_frequency):
            
            minibatch = self.experience[np.random.choice(np.minimum(self.experience_limit, self.current_index), self.batch_size, replace=False)]
            x_batch = minibatch[:, 0:self.state_memory]                                     # Initial states
            actions = minibatch[:, self.state_memory]                                       # State reached
            rewards = minibatch[:, self.state_memory+1]                                     # Action taken in initial state
            next_states = minibatch[:, self.state_memory+2 : 2*self.state_memory+2]         # Immediate rewards observed
            done_arr = minibatch[:, 2*self.state_memory+2]                                  # Values determining if the initial state is terminal

            y_batch = self.session.run(self.model.predict, feed_dict={self.model.X: self.preprocess(x_batch)})            # Predict Q values for state with current model
            future_vals = self.session.run(self.model.predict, feed_dict={self.target_model.X: self.preprocess(next_states)})   # Predict Q values for next_state with frozen model
            
            for i in range(self.batch_size):
                # Current state is terminal, so we add only observed reward
                if(done_arr[i]):
                    y_batch[i,int(actions[i])] = rewards[i]
                
                # Calculate value of current state with estimated optimal value of future state
                else:
                    y_batch[i,int(actions[i])] = rewards[i] + self.discount*np.max(future_vals[i])

            self.session.run(self.model.grad_descent, feed_dict={self.model.X: self.preprocess(x_batch), self.model.Y: y_batch})
            self.w_update_step = 0
        else:
            self.w_update_step += 1
    
    
    '''
    
    Functions for saving and loading agents
    
    '''
    # Save tensorflow model and class attributes
    def save(self):
        
        # Save tensorflow session
        filename = './' + self.save_dir + '/model.ckpt'
        self.saver = tf.train.Saver()
        save_path = self.saver.save(self.session, filename)
        
        # Save agent's data
        filename = './' + self.save_dir + '/agent_data.pkl'
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
            
        # Save agent's experience
        filename = './' + self.save_dir + '/experience.npy'
        np.save(filename, self.experience)
        
        print("Agent saved in path: %s" % self.save_dir)
    
    # Load tensorflow model and class attributes
    def load(self, directory="savedAgent"):
        
        # Run a single episode in order to restore the session properly
        #self.train(max_episodes=1, save=False)
        
        # Restore Tensorflow session
        #filename = './' + directory + '/model.ckpt'
        #self.saver.restore(self.session, filename)
        #print("Model restored from: %s" % directory)
        
        filename = './' + directory + '/model.ckpt.meta'
        with tf.Session() as sess:
            self.saver = tf.train.import_meta_graph(filename)
            dir_ = './' + directory
            self.saver.restore(sess,tf.train.latest_checkpoint(dir_))
        
        
        
        # Restore agent's data
        #filename = './' + directory + '/agent_data.pkl'
        #with open(filename, 'rb') as f:
        #    self = pickle.load(f)
            #pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        
        #print("Agent's data restored")
        
        # Restore Agent's experience
        #directory = './' + directory + '/experience.npy'
        #self.experience = np.load(path_to_file)
        
