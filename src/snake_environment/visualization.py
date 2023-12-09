import pygame
import time
import numpy as np

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0,0,0)

class Visualize:
    def draw_snake(self):
        for segment in self.snake.snake_body[:-1]:
            pygame.draw.rect(self.display, GREEN,(segment[0]*40, segment[1]*40, 40, 40))
        pygame.draw.rect(self.display,BLACK,(self.snake.snake_body[-1][0]*40, self.snake.snake_body[-1][1]*40, 40, 40))
    def draw_food(self):
        pygame.draw.rect(self.display, RED,(self.food_coordinates[0]*40, self.food_coordinates[1]*40, 40, 40))
    def draw(self):
        self.display.fill((255, 255, 255))
        self.draw_snake()
        self.draw_food()
        pygame.display.flip()