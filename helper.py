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


# TODO: make this recursive for more tiles
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


def add_free_path_check(state, game):
    # adds a 1 (True) to the state when its free, 0 (false) otherwise
    snake = game.snake
    head = snake[0]
    food = game.food

    # TODO: look at when head is on food

    if head.x == food.x:
        # take the one that is further up
        low_y = head.y if head.y < food.y else food.y
        high_y = head.y if head.y > food.y else food.y

        while low_y < high_y:
            if Point(head.x, low_y) in snake:
                state.append(False)
                return state

            low_y += BLOCK_SIZE
    if head.y == food.y:
        # take the one that is further left
        low_x = head.x if head.x < food.x else food.x
        high_x = head.x if head.x > food.x else food.x

        while low_x < high_x:
            if Point(low_x, head.y) in snake:
                state.append(False)
                return state

            low_x += BLOCK_SIZE

    state.append(True)
    return state


def get_state(game):
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

    # TODO: dynamic extension manager

    # state = track_positions(state, snake, game)  # + 16 extensions
    # state = two_tile_sight(state, game, head, dir_r, dir_l, dir_u, dir_d)  # + 3 extensions
    # state = add_free_path_check(state, game)  # + 1

    # state = board(game, snake) # + 757 extensions

    return np.array(state, dtype=int)
