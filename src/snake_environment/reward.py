import numpy as np
class Reward:
    def moved_towards_food(self):
        if np.sum(np.abs((self.snake.snake_body[-1] - self.snake.direction_of_snake) - self.food_coordinates)) > np.sum(np.abs(self.snake.snake_body[-1]-self.food_coordinates)) or self.snake.eaten_food:
            return 10
        return -1
    def ate_food(self):
        if self.snake.eaten_food :
            return 500
        return 0
    def died(self):
        if not self.snake.alive:
            return -50
        return 0
    def total_reward(self):
        return self.died() + self.moved_towards_food() + self.ate_food()