
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.distributions import Categorical
import tqdm 


class CliffWalking:
  def __init__(self):
    self.cliff_matrix = np.zeros((4,12))
    self.goal_state = [3,11]
    self.cliff_states = np.array([[3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[3,10]])
    self.state = np.array([3,0])
  
  def limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.cliff_matrix.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.cliff_matrix.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

  def termination(self):
    return (self.state == self.goal_state).all()
  
  def convert_action(self,action):
    #up,right,down,left
    transition_dictionary = {0:[-1,0],1:[0,1],2:[1,0],3:[0,-1]}
    return transition_dictionary[action]
  
  def transition_to_next_state(self,action):
    next_state = self.state + self.convert_action(action)
    next_state = self.limit_coordinates(next_state)
    return next_state
  
  def reward(self):
    if (np.all(self.cliff_states == self.state, axis=1)).any():
      return -100
    else:
      return -1

  def step(self,action):
    self.state = self.transition_to_next_state(action)
    reward,terminated = self.reward(),self.termination()
    if (np.all(self.cliff_states == self.state, axis=1)).any():
      self.state = np.array([3,0])
    return self.state,reward,terminated
  
  def reset(self):
    self.__init__()
    return self.state
  

class Policy(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Policy,self).__init__()
        self.fc1 = nn.Linear(n_input,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_output)
        self.loss = 0
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x,dim=1)
    
    def action(self,x):
        x = torch.from_numpy(x).unsqueeze(0).float()
        prob = self(x)
        x = x.detach()
        m = Categorical(prob)
        action = m.sample()
        return action,action.item(),m.log_prob(action)
    
    def train_policy(self,delta,probs,optim):
        policy_loss = []
        for d,p in zip(delta,probs):
            policy_loss.append(-d * p)
        optim.zero_grad()
        sum(policy_loss).backward()
        self.loss += sum(policy_loss)
        optim.step()
   
   
class Baseline(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Baseline,self).__init__()
        self.fc1 = nn.Linear(n_input,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_output)
        self.total_loss = 0
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def train_baseline(self,optim,G,vals):
        loss = F.mse_loss(vals,G)
        optim.zero_grad()
        loss.backward()    
        self.total_loss += loss
        optim.step()



def convert_reward(rewards,gamma):
    cur_sum,out= 0,[]
    rewards.reverse()  
    for r in rewards:  
        cur_sum = r + gamma* cur_sum
        out.append(cur_sum)  
    out = list(reversed(out))
    out = torch.tensor(out)
    out = (out - out.mean())/out.std()
    return out

def convert_state_to_tensor(states,baseline):
    tensor_states = []
    for s in states:
        s = torch.from_numpy(s).float().unsqueeze(0)
        tensor_states.append(baseline(s))
    tensor_states = torch.stack(tensor_states).squeeze()
    return tensor_states

def stopping_criteria(scores):
    if sum(scores)/len(scores) == 400:
        return True
    return False

def convert_to_one_hot_vector(state):
    one_hot_vector = np.zeros(48)
    state= np.ravel_multi_index(tuple(state), (4,12))
    one_hot_vector[state] = 1
    return one_hot_vector
def plot(steps,name,title_name):
    ax = plt.subplot(111)
    ax.cla()
    ax.set_title(title_name)
    ax.set_xlabel('Episode')
    ax.set_ylabel(name)
    ax.plot(steps)
    RunTime = len(steps)
    path =  './ReinforceBaseline'+title_name+'.jpg'
    plt.savefig(path)
    plt.pause(0.0000001)

def main():
    env =  CliffWalking()
    input_size = 48
    output_size = 4
    hidden_layer_size = 128

    policy = Policy(input_size,hidden_layer_size,output_size)
    baseline = Baseline(input_size,hidden_layer_size,1)
    policy_optimizer = optim.Adam(policy.parameters(),1e-3)
    baseline_optimizer = optim.Adam(baseline.parameters(),1e-2)
    scores = []
    times = []
    MAX_EPISODES = 3000
    gamma = 0.99

    for episode in tqdm(range(MAX_EPISODES)):
        
        states,actions,rewards,log_actions = [],[],[],[]
        score = 0
        state= env.reset()
        done = False
        step = 0
        while not done:
            one_hot_vector_state = convert_to_one_hot_vector(state)
            a,action,log_action = policy.action(one_hot_vector_state)
            #print("action",action)
            new_state, reward, done = env.step(action)
            #print("new_state",new_state)
            score+=reward
            states.append(one_hot_vector_state)
            actions.append(action)
            rewards.append(reward)
            log_actions.append(log_action)
            if done or step == 5000:
                times.append(step)
                print("Episode {}, live time = {}".format(episode, step))
                break
            state = new_state
            step+=1
        
        scores.append(score)  
        states = convert_state_to_tensor(states,baseline)
        G = convert_reward(rewards,gamma)
        baseline.train_baseline(baseline_optimizer,G,states)
        deltas = [gt - val for gt, val in zip(G, states)]
        deltas = torch.tensor(deltas)
        policy.train_policy(deltas,log_actions,policy_optimizer)
    plot(scores,"scores","CliffWalk Episode vs Scores")  
    plot(times,"times","CliffWalk Episode vs Times")
    
if __name__ == "__main__":
   main()
