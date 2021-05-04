import numpy as np
from agent_v2 import Agent
from game import SnakeGameAI
from helper import *

# here the action representation is integer not array !!!

input_dims = 11  # how many elements does the state representation have?
n_actions = 3
scores, avg_scores, eps_history = [], [], []
epochs = 50000

env = SnakeGameAI()
agent = Agent(gamma=0.99, epsilon=1.0, batch_size=4, n_actions=n_actions, eps_end=0.01, input_dims=[input_dims],
              lr=0.003)

for epoch in range(epochs):
    env.reset()
    score = 0
    done = False
    # print(state_old[0].type)
    while not done:  # iterating over every timestep (state)
        # env.start_display()
        state_old = get_state(env)
        action = agent.choose_action(state_old)
        reward, done, score = env.play_step(action, update=False)
        state_new = get_state(env)
        score += reward

        agent.store_transition(state_old, action, reward, state_new, done)
        agent.learn()
        state_old = state_new

    scores.append(score)
    eps_history.append(agent.epsilon)
    avg_score = np.mean(scores[-100:])
    avg_scores.append(avg_score)
    print("epoch: ", epoch, "score: %.2f " % score, "avg_score: %.2f " % avg_score, "epsilon: %.2f" % agent.epsilon)