import pygame
import time
import numpy as np
from utils import * 

class Snake:
    def __init__(self,n):
        self.n = n
        self.alive = True
        self.length = 1
        self.direction_to_move = None
        
        self.snake_body = [np.random.randint(0,high = n,size=2)]
        self.direction_of_snake = [-1,0]
        self.eaten_food = False


    def is_alive(self):
        self.alive = True
        if len(self.snake_body) > 1:
            if np.any(np.all(self.snake_body[-1] == np.array(self.snake_body[:-1]), axis=1)):
                self.alive = False
        if np.any((np.array(self.snake_body) < 0) | np.array(self.snake_body >=  self.n), axis=None):
            self.alive = False
        return self.alive
    
    def eat_food(self):
        self.length+=1
    #0 : up , 1:down,2:right,3:left
  
    def move(self,direction):
        move_vector = direction
        head = self.snake_body[-1]
        new_head = head + move_vector
        self.snake_body.append(new_head)
        if len(self.snake_body) > self.length :
            self.snake_body.pop(0)
        self.direction_of_snake = direction