import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, LambdaLR
import matplotlib.pyplot as plt
import gym
import numpy as np

class GridWorld:
    def __init__(self):
        self.grid = [5,5]
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

        if (next_state[0] == 3 and next_state[1] == 2) or (next_state[0] == 2 and next_state[1] == 2):
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
  def __init__(self, observation_space, action_space):
    super(Policy, self).__init__()

    self.l1 = nn.Linear(observation_space, 32)
    self.output = nn.Linear(32, action_space)
    self.softmax = nn.Softmax()

  def forward(self, x):
    x = F.relu(self.l1(x))
    x = self.output(x)
    return F.softmax(x, dim=1)


class Critic(nn.Module):
  def __init__(self, observation_space):
    super(Critic, self).__init__()

    self.l1 = nn.Linear(observation_space, 32)
    self.output = nn.Linear(32,1)

  def forward(self, x):
    x = F.relu(self.l1(x))
    return self.output(x)

def getAction(policy, state):
  state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
  state = state.detach()
  action_probabilities = policy(state)
  action_space_distribution = Categorical(action_probabilities)
  next_action = action_space_distribution.sample()
  return next_action.item(), action_space_distribution.log_prob(next_action)

def convert_to_one_hot_vector(state):
  one_hot_vector = np.zeros(25)
  state= np.ravel_multi_index(tuple(state), (5,5))
  one_hot_vector[state] = 1
  return one_hot_vector

def plot_policy():
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

def plot_averages(J_thetas,name):
    J_thetas = np.array(J_thetas)
    J_thetas_mean = np.mean(J_thetas,axis=0)
    J_std = np.std(J_thetas,axis=0)

    fig, ax = plt.subplots()
    ax.plot(J_thetas_mean,label='Mean')
    plt.fill_between(np.arange(len(J_thetas_mean)), J_thetas_mean - J_std, J_thetas_mean + J_std, alpha=0.3,label='Standard Deviation')
    ax.set_xlabel("Number of Iterations")
    ax.set_ylabel(name)
    ax.set_title("Learning curve of actor critic across 20 trials")
    ax.legend()

# Set parameters
NUM_EPISODES = 500
DISCOUNT = 0.99
P_ALPHA = 1e-3
C_ALPHA = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
environment = GridWorld()
input_size = 25
output_size = 4
hidden_layer_size = 64
policy = Policy(input_size, output_size).to(DEVICE)
critic = Critic(input_size).to(DEVICE)
policy_opt = optim.Adam(policy.parameters(), lr=P_ALPHA)
critic_opt = optim.Adam(critic.parameters(), lr=C_ALPHA)

# Start training
all_rewards = []
len = 0
critic_losses = []
eps_lens = []
max_norm_losses = []
for episode in range(NUM_EPISODES):
  state = environment.reset()
  state = convert_to_one_hot_vector(state)
  I = 1
  total_reward = 0
  episode_len = 0
  loss = 0
  max_norm = 0
  while True:
    len+=1
    episode_len += 1

    next_action, action_prob = getAction(policy, state)
    next_state, reward, terminal = environment.step(next_action)

    curr_state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
    curr_state_val = critic(curr_state_tensor)
    next_state = convert_to_one_hot_vector(next_state)
    next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(DEVICE)
    next_state_val = critic(next_state_tensor)

    if terminal:
      next_state_val = torch.tensor([0]).float().unsqueeze(0).to(DEVICE)

    total_reward += reward


    critic_loss = F.mse_loss(reward + (DISCOUNT * next_state_val), curr_state_val)
    critic_opt.zero_grad()
    critic_loss.requires_grad_()
    critic_loss.backward()
    critic_opt.step()
    loss += critic_loss.detach().item()
    max_norm = max(max_norm, critic_loss.detach().item())

    curr_state_val = critic(curr_state_tensor)
    next_state_val = critic(next_state_tensor)
    delta = reward + (DISCOUNT * next_state_val.detach().item()) - curr_state_val.detach().item()

    policy_loss = - action_prob * delta * I
    policy_opt.zero_grad()
    policy_loss.requires_grad_()
    policy_loss.backward(retain_graph=True)
    policy_opt.step()




    if terminal or episode_len == 500:
      print("Episode {}, score = {}".format(episode, total_reward))
      break


    I = I * DISCOUNT
    state = next_state

  critic_losses.append(loss/episode_len)
  eps_lens.append(episode_len)
  max_norm_losses.append(max_norm)
  all_rewards.append(total_reward)

# Run to plot the policy
plot_policy()

# Plot the leanring curve
plt.plot(all_rewards)
plt.xlabel("Episode Number")
plt.ylabel("Total reward (undiscounted)")
plt.title("Actor-Critic learning curve for Grid World")
plt.show()

# Plot losses
plt.plot(critic_losses)
plt.xlabel("Episode Number")
plt.ylabel("MSE")
plt.title("Actor-Critic loss curve for Grid World")
plt.show()

# Plot episode lengths
plt.plot(eps_lens)
plt.xlabel("Episode Number")
plt.ylabel("Episode length")
plt.title("Actor-Critic episode lenght plot for Grid World")
plt.show()


