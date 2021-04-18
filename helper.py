import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from game import SnakeGameAI, Direction, Point


BLOCK_SIZE = 20


def plot(scores, mean_scores, epoch):
    # display.clear_output(wait=True)
    # display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    if epoch % 10 == 0:
        plt.show()


def two_tile_sight(state, game, head, dir_r, dir_l, dir_u, dir_d):
    point_2l = Point(head.x - 40, head.y)
    point_2r = Point(head.x + 40, head.y)
    point_2u = Point(head.x, head.y - 40)
    point_2d = Point(head.x, head.y + 40)

    state.extend([  # Danger 2straight
        (dir_r and game.is_collision(point_2r)) or
        (dir_l and game.is_collision(point_2l)) or
        (dir_u and game.is_collision(point_2u)) or
        (dir_d and game.is_collision(point_2d)),

        # Danger 2right
        (dir_u and game.is_collision(point_2r)) or
        (dir_d and game.is_collision(point_2l)) or
        (dir_l and game.is_collision(point_2u)) or
        (dir_r and game.is_collision(point_2d)),

        # Danger 2left
        (dir_d and game.is_collision(point_2r)) or
        (dir_u and game.is_collision(point_2l)) or
        (dir_r and game.is_collision(point_2u)) or
        (dir_l and game.is_collision(point_2d))])

    return state


def track_positions(state, snake, game):
    # track position of #self.extension bodyparts relative to head
    for idx in range(-2, 3):
        if idx == 0:
            continue
        state.extend([snake[idx].x < game.head.x, snake[idx].x > game.head.x, snake[idx].y < game.head.y,
                      snake[idx].y > game.head.y, ])
    return state


def board(game, snake):
    # board idea
    board = np.zeros((24, 32))
    for point in snake:
        board[int((point.y - BLOCK_SIZE) // BLOCK_SIZE)][int((point.x - BLOCK_SIZE) // 20)] = 1

    board[int((game.food.y - BLOCK_SIZE) // BLOCK_SIZE)][int((game.food.x - BLOCK_SIZE) // 20)] = -1

    return board.flatten()
