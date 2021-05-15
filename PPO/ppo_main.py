import gym
import numpy as np
from ppo_agent import Agent
from plot import plot_learning_curve
from game import SnakeGameAI

# TODO: tailor this for snake

if __name__ == '__main__':
    env = SnakeGameAI()
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=3, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=(11,))
    n_games = 300

    figure_file = 'plots/snake.png'

    best_score = 0
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0
    display = True

    if display:
        env.start_display()

    for i in range(n_games):
        env.reset()
        observation = env.get_state()
        done = False
        score = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            reward, done, score = env.play_step(action, display)
            observation_ = env.get_state()
            n_steps += 1
            score += score
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
