import numpy as np
import statistics
import torch
import matplotlib.pyplot as plt

def convert_to_action_vector(direction):
    move_vector = {0:(-1, 0),1:(1, 0),2:(0, 1),3:(0, -1)}
    return move_vector[direction]

def convert_reward(rewards,gamma):
    cur_sum,out= 0,[]
    rewards.reverse()  
    
    for r in rewards:  
        cur_sum = r + gamma* cur_sum
        out.append(cur_sum)  
    out = list(reversed(out))
    out = torch.tensor(out)
    if len(rewards) != 1:
        out = (out - out.mean())/out.std()
    return out

def convert_state_to_tensor(states,baseline):
    tensor_states = []
    for s in states:
        s = torch.from_numpy(s).float().unsqueeze(0)
        tensor_states.append(baseline(s))
    if len(tensor_states) != 1:
        tensor_states = torch.stack(tensor_states).squeeze()
    else:
        tensor_states = tensor_states[0].view(-1)
    return tensor_states

def plot(steps):
    #print(steps)
    #ax = plt.subplot(111)
    #ax.cla()
    plt.plot(steps)
    plt.title('Training_snake')
    plt.xlabel('Episode')
    plt.ylabel("Score in each episode")
    path =  "Snake_maxscore.jpg"
    plt.savefig(path)
   

def is_in_body(head,body):
    if len(body) > 1 :
        return (np.all(head == body,axis=1)).any()
    return False

def plot_final_scores(all_scores):
  x_values = [i for i in range(len(all_scores[0]))]
  y_values = [val for sublist in all_scores for val in sublist]
  window_size = 4
  y_smooth = np.convolve(y_values, np.ones(window_size)/window_size, mode='valid')
  x_smooth = x_values[:len(y_smooth)]
  plt.scatter(x_smooth, y_smooth, label='Smoothed Curve', color='red')
  plt.xlabel('Episodes')
  plt.ylabel('Scores')
  plt.title('Scatter Plot of Episodes vs Scores')
  plt.savefig('all_scores.jpg')