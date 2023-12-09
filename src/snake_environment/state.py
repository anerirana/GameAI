import numpy as np
from utils import *
class States:
 
    def one_hot_vector(self,key):
        key = tuple(key)
        direction_to_one_hot = {(-1, 0): [1, 0, 0, 0], (1, 0): [0, 1, 0, 0], (0, 1): [0, 0, 1, 0], (0, -1): [0, 0, 0, 1]}
        return direction_to_one_hot[key]
  
    def one_hot_vector_food_coordinates(self):
        new_direction = np.zeros(2, dtype=int)
        idx =  np.argmax(np.abs(self.food_coordinates -self.snake.snake_body[-1]))
        new_direction[idx] = np.sign((self.food_coordinates -self.snake.snake_body[-1])[idx])
        return self.one_hot_vector(new_direction)
    
    def obstacle(self):
        def check_obstacle(dir):
            if np.any(np.logical_or(self.snake.snake_body[-1] + dir < 0, self.snake.snake_body[-1] + dir >= self.n)) or is_in_body(self.snake.snake_body[-1],self.snake.snake_body[:-1]) :
                return 1
            return 0
        return np.array([check_obstacle([-1,0]),check_obstacle([1,0]),check_obstacle([0,1]),check_obstacle([0,-1])])
    
    def get_state(self):
        if (self.food_coordinates == self.snake.snake_body[-1]).all():
            self.set_food()
        return np.array([*self.one_hot_vector_food_coordinates(),*self.one_hot_vector(self.snake.direction_of_snake),*self.obstacle()])

