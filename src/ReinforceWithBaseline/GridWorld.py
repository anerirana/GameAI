import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, LambdaLR
import matplotlib.pyplot as plt
import gym
from tqdm import tqdm_notebook
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class GridWorld:
    def __init__(self):
        self.goal = [4,4]
        self.water_state = [4,2]
        self.obstacles = np.array([[3,2],[2,2]])
        self.state = np.array([0,0])

    def termination(self):
        return (self.state == self.goal).all()

    def convert_action(self,action):
    #up,down,left,right
        transition_dictionary = {0:[-1,0],1:[0,1],2:[-1,0],3:[1,0]}
        return transition_dictionary[action]

    def transition(self,action):
        probs = [0.8,0.05,0.05,0.1]
        if action == 0:
            next_state = [self.state + [-1,0],self.state + [0,1],self.state + [0,-1],self.state]
        elif action  == 1:
             next_state = [self.state + [1,0],self.state + [0,-1],self.state + [0,1],self.state]
        elif action == 2:
            next_state = [self.state + [0,1],self.state + [1,0],self.state + [-1,0],self.state]
        else:
            next_state = [self.state + [0,-1],self.state + [-1,0],self.state + [1,0],self.state]
        idx = np.random.choice([0,1,2,3],p=probs)
        return next_state[idx]

    def reward(self):

        if (self.state == self.water_state).all():
            return -10

        elif (self.state == self.goal).all():
            return 10

        else:
            return 0
    def limit_coordinates(self,coord):
        coord[0] = min(coord[0], 4)

        coord[0] = max(coord[0], 0)

        coord[1] = min(coord[1], 4)

        coord[1] = max(coord[1], 0)


        return coord

    def transition_to_next_state(self,action):

        next_state = self.transition(action)


        next_state = self.limit_coordinates(next_state)

        if (np.all(next_state == self.obstacles, axis=1)).any():
            next_state = self.state

        return next_state
    def step(self,action):
        self.state = self.transition_to_next_state(action)
        reward,terminated = self.reward(),self.termination()
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
        x = torch.from_numpy(x).float().unsqueeze(0).to(DEVICE)
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
        return loss.detach().item()
    
def convert_reward(rewards,gamma):
    cur_sum,out= 0,[]
    rewards.reverse()
    for r in rewards:
        cur_sum = r + gamma* cur_sum
        out.append(cur_sum)
    out = list(reversed(out))
    out = torch.tensor(out).to(DEVICE)
    out = (out - out.mean())/out.std()
    return out

def convert_state_to_tensor(states,baseline):
    tensor_states = []
    for s in states:
        s = torch.from_numpy(s).float().unsqueeze(0).to(DEVICE)
        tensor_states.append(baseline(s))
    tensor_states = torch.stack(tensor_states).squeeze()
    return tensor_states

def stopping_criteria(scores):
    if sum(scores)/len(scores) == 400:
        return True
    return False

def convert_to_one_hot_vector(state):
    one_hot_vector = np.zeros(25)
    state= np.ravel_multi_index(tuple(state), (5,5))
    one_hot_vector[state] = 1
    return one_hot_vector

def plot_policy(policy):
  for r in range(5):
    for c in range(5):
      if (r == 3 and c == 2) or (r == 2 and c == 2):
        max_a = ''
      else:
        state = convert_to_one_hot_vector([r,c])
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probabilities = policy(state)
        max_a = torch.argmax(action_probabilities)
      if max_a == 0:
        print(u"\u2191 \t", end =" ")
      elif max_a == 1:
        print(u"\u2193 \t", end =" ")
      elif max_a == 2:
        print(u"\u2192 \t", end =" ")
      elif max_a == 3:
        print(u"\u2190 \t", end =" ")
      else:
        print(max_a + "\t", end =" ")
    print("\n")

def plot_averages(J_thetas,title, ylabel):
    J_thetas = np.array(J_thetas)
    J_thetas_mean = np.mean(J_thetas,axis=0)
    J_std = np.std(J_thetas,axis=0)

    #plt.plot(np.arange(4),J_thetas)
    #plt.show()
    #plt.close()
    fig, ax = plt.subplots()
    ax.plot(J_thetas_mean,label='Mean')
    plt.fill_between(np.arange(len(J_thetas_mean)), J_thetas_mean - J_std, J_thetas_mean + J_std, alpha=0.3,label='Standard Deviation')
    ax.set_xlabel("Episode Number")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

def run(verbose=False):
    env =  GridWorld()
    input_size = 25
    output_size = 4
    hidden_layer_size = 32

    policy = Policy(input_size,hidden_layer_size,output_size).to(DEVICE)
    baseline = Baseline(input_size,hidden_layer_size,1).to(DEVICE)
    policy_optimizer = optim.Adam(policy.parameters(),1e-3)
    baseline_optimizer = optim.Adam(baseline.parameters(),1e-3)
    scores = []
    times = []
    MAX_EPISODES = 500
    gamma = 0.99
    baseline_losses = []

    for episode in tqdm_notebook(range(MAX_EPISODES)):

        states,actions,rewards,log_actions = [],[],[],[]
        score = 0
        state= env.reset()

        done = False
        step = 0
        while not done:
            one_hot_vector_state = convert_to_one_hot_vector(state)
            #print(state)
            #print(one_hot_vector_state)
            a,action,log_action = policy.action(one_hot_vector_state)
            #print("action",action)
            new_state, reward, done = env.step(action)
            #print("new_state",new_state)
            score+=reward
            states.append(one_hot_vector_state)
            actions.append(action)
            rewards.append(reward)
            log_actions.append(log_action)
            if done :
                times.append(step)
                if verbose:
                  print("Episode {}, score= {}".format(episode, score))
                break
            state = new_state
            step+=1
        scores.append(score)
        states = convert_state_to_tensor(states,baseline)
        G = convert_reward(rewards,gamma)
        loss = baseline.train_baseline(baseline_optimizer,G,states)
        baseline_losses.append(loss)
        deltas = [gt - val for gt, val in zip(G, states)]
        deltas = torch.tensor(deltas).to(DEVICE)
        policy.train_policy(deltas,log_actions,policy_optimizer)
    plot_policy(policy)
    return scores,times, baseline_losses

# Single run of algorithm
scores,times, baseline_losses = run()

#Plot learning curve
plt.plot(scores)
plt.xlabel("Episode Number")
plt.ylabel("Total reward (undiscounted)")
plt.title("REINFORCE with Baseline learning curve for Grid World")
plt.show()

# Mutliple trials
trial_rewards = []
losses = []
all_episode_lens = []
for i in range(10):
  scores,times, baseline_losses = run(verbose=False)
  trial_rewards.append(scores)
  losses.append(baseline_losses)
  all_episode_lens.append(times)

# Plot learning curve across trials
plot_averages(trial_rewards, title="Learning curve of REINFORCE with Baseline across 20 trials", ylabel="Total Reward (across trials)")

# Plot loss across trials
plot_averages(losses, title="Loss curve of REINFORCE with Baseline across 20 trials", ylabel="MSE (across trials)")

# Plot episode lenght across trials
plot_averages(all_episode_lens, title="Episode length plot for REINFORCE with Baseline across 20 trials", ylabel="Episode Length (across trials)")