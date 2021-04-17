import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from Model import Linear_QNet, QTrainer
from helper import *

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

# how big one block is in pixels
BLOCK_SIZE = 20
# how many games until start displaying
DISPLAY_GAMES = 0


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.extensions = 16 + 3  # how many points (body squares) do i wanna track
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(11 + self.extensions, 256, 3).to(
            device)  # 11 value state input, 3 action as output (s,l, r)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):

        # TODO: fix state so it doesnt trap itself

        head = game.snake[0]
        tail = game.snake[-1]
        snake = game.snake
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # 11 value state : [danger straight, danger right, danger left, direction left, direction right,
        # direction up, direction down, food left, food right, food up, food down] --> 0/1 for T/F
        # TODO: cuda
        # TODO: idea 1 : track tail
        # TODO: idea 2 : track average location of body
        # TODO: idea 3 : penalty for time? penalty for loop?
        # TODO: idea 4 : give whole board as input  xxxx
        # TODO: idea 5 : -1 reward if head is "inside of snake"  xxxx
        # TODO: idea 6 : give 2 vision squares
        # TODO: idea 7 : give reward when he is in a position where nothing is in the way for 2 tiles (free square in front)
        # TODO: check if path to food is not free !!!!
        # TODO: idea 7 : punish when it has to reset (maybe prevent idle animations)

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down

        ]

        state = track_positions(state, snake, game) # + 16 extensions

        state = two_tile_sight(state, game, head, dir_r, dir_l, dir_u, dir_d) # + 3 extensions

        # state = board(game, snake) # + 757 extensions

        return np.array(state, dtype=int)
        # return np.append(np.array(state, dtype=int), np.array(board.flatten(), dtype=int))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # popleft if MAX_MEMORY is reached

    # after every episode
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    # after every action
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation, shrinking epsilon
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    # additional helper methods


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:  # the good old while true
        if agent.n_games == DISPLAY_GAMES:
            game.start_display()

        # get old state
        state_old = agent.get_state(game)

        # get move prediction
        final_move = agent.get_action(state_old)

        # perform move and get new state
        if agent.n_games <= DISPLAY_GAMES:
            reward, done, score = game.play_step(final_move, False)
        else:
            reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory with the information we just got by playing A in state S and getting reward R and ending
        # up in S' (new state)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember for after epoch learning
        agent.remember(state_old, final_move, reward, state_new, done)

        # after every epoch
        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores, agent.n_games)


if __name__ == '__main__':
    train()
