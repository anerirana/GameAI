import numpy as np
from grid import *
import sys
from Reinforce_with_baseline import *
from utils import *
import torch.optim as optim
import matplotlib.pyplot as plt

GAMMA = 0.9
NUM_EPISODES = 10000
all_scores = []

def main():
    grid = Grid(5)
    Reinforce = Policy(12,4,0.01)
    baseline = Baseline(12,1,0.01)
    policy_optimizer = optim.Adam(Reinforce.parameters(), lr=1e-3)
    baseline_optimizer = optim.Adam(baseline.parameters(), lr=1e-3)
    i = 0
    max_scores = []
    scores = []
    cur_score_array = []
    for episode in range(NUM_EPISODES):
        flag = False
        total_reward = 0 
        rewards = []
        states = []
        actions = []
        log_actions = []
        state = grid.get_state()
        max_score = 0
        cur_scores = 0
        step = 0
        while not flag:
            _,action,log_action = Reinforce.action(state)
            act  = convert_to_action_vector(action)
            state,next_state,action, reward,flag,cur_score= grid.run(act,episode)
            max_score = max(max_score,cur_score)
            cur_scores += cur_score
            rewards.append(reward)
            states.append(state)
            log_actions.append(log_action)
            total_reward += reward
            if flag :
                print("At episode"+str(episode) + "max score is" + str(max_score))
                #cur_score_array.append(cur_score)

                break
            state = next_state
            step +=1
        G = convert_reward(rewards,GAMMA)
        vals = convert_state_to_tensor(states,baseline)
        baseline.train_baseline(baseline_optimizer,G,vals)
        deltas = [gt - val for gt, val in zip(G, vals)]
        deltas = torch.tensor(deltas)
        Reinforce.train_policy(deltas,log_actions,policy_optimizer)
        scores.append(total_reward)
        cur_score_array.append(cur_scores)
        max_scores.append(max_score)
    all_scores.append(max_scores)
    #print(cur_score_array)
    plot(max_scores)
    
        

if __name__ == "__main__":
  main()

#check the distance of the food coordinate systerm
#check the function one_hot_vector_food_coorfinates
#the snake eats the food and then dies. Check why