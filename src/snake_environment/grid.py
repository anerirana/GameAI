import numpy as np
import pygame 
import time
from reward import *
from state import *
from visualization import *
from snake import *



class Grid(Reward,States,Visualize): 
    def __init__(self,n):
        self.n = n
        self.states = States()
        self.reward = Reward()
        #self.display = pygame.display.set_mode((600, 600))
        #pygame.display.set_caption('Snake Game')
        #self.clock = pygame.time.Clock()
        #self.visualize = Visualize()
        self.curr_score = 0
        self.actions = np.array([[-1,0],[1,0],[0,-1],[0,1]])
        self.snake = Snake(self.n)
        self.board = np.zeros((self.n,self.n))
        self.food_coordinates =  np.random.randint(0, self.n, 2)
        self.food_coordinates = np.zeros(2)
        self.high_score = 0
        self.eaten_this_round = False
        self.high_score = 0 
  
    def set_food(self):
        #self.food_coordinates =  np.random.randint(0, self.n, 2)
        #check
        while np.all(self.food_coordinates == self.snake.snake_body,axis = 1).any():
           
            self.food_coordinates =  np.random.randint(0, self.n, 2)
           
    def check_if_food_is_eaten(self):
        if np.sum(np.abs(self.snake.snake_body[-1] - self.food_coordinates)) == 0:
            self.snake.eat_food()
            #self.max_score = max(self.curr_score,self.max_score)
            self.set_food()
            self.curr_score +=1
            self.snake.eaten_food = True
        else:
            self.snake.eaten_food = False


    def reset_board(self):
        self.snake = Snake(self.n)
        self.food_coordinates = np.random.randint(0, self.n, 2)
        self.curr_score = 0


    def run(self,dir,step):
        dead = False
        cur_state = self.get_state()
        action = dir
        self.snake.move(action)
        self.check_if_food_is_eaten()
        reward = self.total_reward()
        #self.draw()
        #pygame.time.delay(300)  # Adjust delay for visualization
        #self.clock.tick(30)
        self.snake.is_alive()
        if not self.snake.alive:
            self.reset_board()
            self.set_food()
            self.curr_score = 0
            dead = True
        new_state = self.get_state()
        return cur_state,new_state,action,reward,dead,self.curr_score
    




