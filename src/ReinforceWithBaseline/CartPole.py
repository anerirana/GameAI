import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

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

def plot_averages(J_thetas):
    J_thetas = np.array(J_thetas)
    J_thetas_mean = np.mean(J_thetas,axis=0)
    J_std = np.std(J_thetas,axis=0)
   
    fig, ax = plt.subplots()
    ax.plot(J_thetas_mean,label='Mean')
    plt.fill_between(np.arange(len(J_thetas_mean)), J_thetas_mean - J_std, J_thetas_mean + J_std, alpha=0.3,label='Standard Deviation')
    ax.set_xlabel("Episode Length")
    ax.set_ylabel("Average Reward(20 Trials) ")
    ax.set_title("Cart Pole Average Reward vs Episode over 20 Trials")
    ax.legend()
    plt.savefig("Cartpole_over_20.png")

def plot_hyperparameters(runs):
    
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(runs[0])
    axs[0, 0].set_title('Alpha:0.0001,Hidden:32,gamma:0.92')
    axs[0, 1].plot(runs[1], 'tab:orange')
    axs[0, 1].set_title('Alphs:0.001,Hidden:32,gamma:0.97')
    axs[1, 0].plot(runs[2], 'tab:green')
    axs[1, 0].set_title('Aplha:0.01,Hidden:64,gamma:0.92')
    axs[1, 1].plot(runs[3], 'tab:red')
    axs[1, 1].set_title('Aplha:0.01,Hidden:128,gamma:0.99')
    

    for ax in axs.flat:
        ax.set(xlabel='Episodes', ylabel='Scores')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.savefig("Cartpole_over_20_hyperparameter.png")

def run(env): 
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    hidden_layer_size = 128
    alpha = 1e-3
    policy = Policy(input_size,hidden_layer_size,output_size)
    baseline = Baseline(input_size,hidden_layer_size,1)
    policy_optimizer = optim.Adam(policy.parameters(),alpha)
    baseline_optimizer = optim.Adam(baseline.parameters(),alpha)
    scores = []
    MAX_EPISODES = 500
    gamma = 0.99

    for episode in tqdm(range(MAX_EPISODES)):
        states,actions,rewards,log_actions = [],[],[],[]
        score = 0
        state,info = env.reset()
        done = False
        i = 0
        while not done:
            a,action,log_action = policy.action(state)
            new_state, reward, done, truncated,info = env.step(action)
            score+=reward
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_actions.append(log_action)
            if done or truncated:
                break
            state = new_state
        scores.append(score)  
        states = convert_state_to_tensor(states,baseline)
        if len(scores) % 100 == 0 and stopping_criteria(scores[-100:]):
            break
        G = convert_reward(rewards,gamma)
        baseline.train_baseline(baseline_optimizer,G,states)
        deltas = [gt - val for gt, val in zip(G, states)]
        deltas = torch.tensor(deltas)
        policy.train_policy(deltas,log_actions,policy_optimizer)
    return scores

def main():  
    all_scores = []
    for x in range(20):
        env = gym.make("CartPole-v1")
        score_of_run = run(env)
        all_scores.append(score_of_run)
        env.close()
    plot_averages(all_scores)

if __name__ == "__main__":
    main()










