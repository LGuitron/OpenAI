from Agent import Agent
import numpy as np
import time

def run(agent, threshold, consecutive, display_frequency, scores, i_episode, render_game):
        
        state = agent.env.reset()
        time_step = 0
        
        while(True):    
            
            if(render_game):
                agent.env.render()
            action = agent.epsilonGreedy(state)                                    # Pick action with epsilonGreedy policy
            next_state, reward, done, info = agent.env.step(action)
            agent.addExperience((state, action, reward, next_state, done))         # Store transition if it is not a losing state
            agent.updateWeights()                                                  # Update network weights
            state = next_state
            time_step += 1

            if done:

                # When training only show avg score  after specified episodes
                if (not render_game):
                    i_episode += 1
                    if(agent.epsilon > agent.min_epsilon):
                        agent.epsilon = agent.epsilon*agent.epsilon_decay
                    scores[i_episode % consecutive] = time_step+1
                    mean_score = np.mean(scores)
                    if((i_episode+1>=consecutive and (i_episode+1) % display_frequency == 0) or mean_score>=threshold):
                        print("Episode ", i_episode+1 , "  | avg timesteps " , np.mean(scores))
                        if(mean_score >= threshold):
                            return scores, True
                
                # When rendering print score
                else:
                    print("Score: " , time_step+1)

                break
        return scores, False



if __name__ == "__main__":
    
    threshold = 195.0                 # Value to achieve consecutively to solve de problem
    consecutive = 100                 # Number of consecutive times required to get above the threshold to solve problem
    scores = np.zeros(consecutive)    # Store scores for last episodes
    display_frequency = 20            # Amount of episodes before displaying information in the console
    start_time = time.time()
    solved_episodes = 10              # Number of episodes to show after the problem has been solved
    
    solved = False
    i_episode = 0
    
    agent = Agent()

    print("\nCartPole-v0")
    print("--------")
    print("Training")
    print("--------")

    while(not solved):
        scores, solved = run(agent, threshold, consecutive, display_frequency, scores, i_episode ,False)
        i_episode += 1

    print("Solved after " , i_episode+1 , " episodes")
    print("Time taken " , "{0:.2f}".format(time.time() - start_time) , " s")
    
    print("\n-------")
    print("Results")
    print("-------")

    agent.epsilon = 0                   # Set agent exploration to 0 for display purposes
    i_episode = 0
    for i in range(solved_episodes):
        scores, solved = run(agent, threshold, consecutive, display_frequency, scores, i_episode ,True)
    
    
    
