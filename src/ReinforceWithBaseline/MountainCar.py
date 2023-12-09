import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categori

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

def plot(steps):
    ax = plt.subplot(111)
    ax.cla()
    ax.set_title('Training')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Run Time')
    ax.plot(steps)
    RunTime = len(steps)
    path =  './ReinforceBaseline'+str(RunTime)+'.jpg'
    if len(steps) % 100 == 0:
        plt.savefig(path)

def main(): 
    env = gym.make('MountainCar-v0')
    env = env.unwrapped
    torch.manual_seed(1)
    plt.ion()
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    policy = Policy(input_size,output_size,1e-3)
    baseline = Baseline(input_size,1,1e-3)
    policy_optimizer = optim.Adam(policy.parameters(),1e-3)
    baseline_optimizer = optim.Adam(baseline.parameters(),1e-3)
    scores = []
    steps = []
    MAX_EPISODES = 300
    gamma = 0.99
    for episode in tqdm(range(60000)):
       
        states,actions,rewards,log_actions = [],[],[],[]
        score = 0
        state,info = env.reset()
        done = False
        i = 0
        for t in range(10000):
            a,action,log_action = policy.action(state)
            new_state, reward, done, truncated,info = env.step(action)
            score =score * gamma - t*0.01
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_actions.append(log_action)
            if done or truncated:
                print("Episode {}, live time = {}".format(episode, t))
                steps.append(t)
                plot(steps)
                break
            state = new_state
        scores.append(score)  
        states = convert_state_to_tensor(states,baseline)
        G = convert_reward(rewards,gamma)
        baseline.train_baseline(baseline_optimizer,G,states)
        deltas = [gt - val for gt, val in zip(G, states)]
        deltas = torch.tensor(deltas)
        policy.train_policy(deltas,log_actions,policy_optimizer)
    plt.plot(scores)  
    plt.show()
    plt.close()
    env.close()

if __name__ == "__main__":
    main()










