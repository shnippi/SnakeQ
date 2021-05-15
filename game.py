import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)


# font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
GREEN = (124, 252, 0)
PINK1 = (255, 20, 147)
PINK2 = (255, 105, 180)
BLACK = (0, 0, 0)

# how big one block is in pixels
BLOCK_SIZE = 20
# speeeed go brr
SPEED = 10000


class SnakeGameAI:

    def __init__(self, w=640, h=480):  # 32 x 24
        self.w = w
        self.h = h
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        # start with length 3 snake
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.was_in_cage_already = 0  # only give negative reward the first time it enters a cage

    def start_display(self):
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action, update=True):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -100
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 100
            self._place_food()
        else:
            self.snake.pop()

        # # check if in cage
        # if self.cage_check() and self.was_in_cage_already < 3:
        #     reward = -10
        #     self.was_in_cage_already += 1
        #
        # # check if adjacent, punish loops
        # if self.is_adjacent(self.head):
        #     reward = -1
        #
        # # TODO: snake always goes down??
        # # check view
        # if not self.free_view(2,2):
        #     reward = -1

        # 5. update ui and clock
        if update:
            self._update_ui()
        if self.score > 10:
            self.clock.tick(int(SPEED / 1))
        else:
            self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def cage_check(self):

        # check if snake is in itself, PYGAME 0,0 IN TOP LEFT CORNER and Y INCREASE FROM TOP TO BOTTOM!!!!
        up = False
        down = False
        left = False
        right = False
        for point in self.snake[1:]:
            if point.x > self.head.x:  # and point.x - self.head.x < 3*BLOCK_SIZE:
                right = True

            if point.x < self.head.x:  # and self.head.x - point.x < 3*BLOCK_SIZE:
                left = True

            if point.y > self.head.y:  # and point.y - self.head.y < 3*BLOCK_SIZE:
                down = True
            if point.y < self.head.y:  # and self.head.y - point.y < 3*BLOCK_SIZE:
                up = True

        # print(self.snake[1:])
        # print(self.head.x, self.head.y)
        # print(self.snake[-1].x, self.snake[-1].y)
        # print(up,down,left,right)
        return up and down and left and right

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        return False

    # see if a point is next to a snake part
    def is_adjacent(self, pt):
        point_l = Point(pt.x - BLOCK_SIZE, pt.y)
        point_r = Point(pt.x + BLOCK_SIZE, pt.y)
        point_u = Point(pt.x, pt.y - BLOCK_SIZE)
        point_d = Point(pt.x, pt.y + BLOCK_SIZE)

        # ignore head and first tile since theyre always adjacent
        if point_l in self.snake[2:] or point_r in self.snake[2:] or point_u in self.snake[2:] or point_d in self.snake[
                                                                                                             2:]:
            return True

        return False

    # see if anything is infront on the snake in a height x width rectangle FROM SNAKE POV (height parallel, with ortho)
    def free_view(self, height, width):
        right = self.direction = Direction.RIGHT
        left = self.direction = Direction.LEFT
        up = self.direction = Direction.UP
        down = self.direction = Direction.DOWN

        side = width // 2
        head = self.head

        if up:
            start = Point(head.x - side * BLOCK_SIZE, head.y - BLOCK_SIZE)
            for h in range(height + 1):
                for w in range(width + 1):
                    if self.is_collision(Point(start.x + w * BLOCK_SIZE, start.y - h * BLOCK_SIZE)):
                        return False

        elif down:
            start = Point(head.x - side * BLOCK_SIZE, head.y + BLOCK_SIZE)
            for h in range(height + 1):
                for w in range(width + 1):
                    if self.is_collision(Point(start.x + w * BLOCK_SIZE, start.y + h * BLOCK_SIZE)):
                        return False

        # for right and left width/height are switched
        elif right:
            start = Point(head.x + BLOCK_SIZE, head.y - side * BLOCK_SIZE)
            for h in range(height + 1):
                for w in range(width + 1):
                    if self.is_collision(Point(start.x + h * BLOCK_SIZE, start.y + w * BLOCK_SIZE)):
                        return False

        elif left:
            start = Point(head.x - BLOCK_SIZE, head.y - side * BLOCK_SIZE)
            for h in range(height + 1):
                for w in range(width + 1):
                    if self.is_collision(Point(start.x - h * BLOCK_SIZE, start.y + w * BLOCK_SIZE)):
                        return False

        return True

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, PINK1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, PINK2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight,right,left] seen from the snake POV
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        #  translating from snake POV in absolute directions seen from the board
        if np.array_equal(action, [1, 0, 0]) or action == 0:
            new_dir = clock_wise[idx]  # straight
        elif np.array_equal(action, [0, 1, 0]) or action == 1:
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right ( move clockwise )
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left (move counterclockwise)

        self.direction = new_dir

        # update head
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def get_state(self):

        # TODO: fix state so it doesnt trap itself

        head = self.snake[0]
        tail = self.snake[-1]
        snake = self.snake
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

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
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.food.x < self.head.x,  # food left
            self.food.x > self.head.x,  # food right
            self.food.y < self.head.y,  # food up
            self.food.y > self.head.y,  # food down

        ]

        # TODO: dynamic extension manager

        # state = track_positions(state, snake, game)  # + 16 extensions
        # state = two_tile_sight(state, game, head, dir_r, dir_l, dir_u, dir_d)  # + 3 extensions
        # state = add_free_path_check(state, game)  # + 1

        # state = board(game, snake) # + 757 extensions

        return np.array(state, dtype=int)
