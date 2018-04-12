from Agent import Agent
from Prep_Input import Prep_Input
import numpy as np

if __name__ == "__main__":
    #agent = Agent("Breakout-ram-v0", 32,64, preprocess = Prep_Input.binary)
    #agent = Agent("Breakout-ram-v0", 32,64)
    agent = Agent("Breakout-ram-v0")
    agent.train(max_episodes=15, display_frequency = 10)

    load_agent = Agent("Breakout-ram-v0")
    #load_agent.train(max_episodes=1)
    load_agent.load('savedAgent')
