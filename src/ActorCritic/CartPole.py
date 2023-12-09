import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, LambdaLR
import matplotlib.pyplot as plt
import gym
import tqdm
import numpy as np
import matplotlib.pyplot as plt

NUM_EPISODES = 1000
DISCOUNT = 0.99
P_ALPHA = 1e-4
C_ALPHA = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Policy(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Policy, self).__init__()

        self.l1 = nn.Linear(observation_space, 128)
        self.output = nn.Linear(128, action_space)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.l1(x))
        # x = F.relu(self.l2(x))
        x = self.output(x)
        return F.softmax(x, dim=1)
    
class Critic(nn.Module):
    def __init__(self, observation_space):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(observation_space, 128)
        self.output = nn.Linear(128,1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        # x = F.relu(self.l2(x))
        return self.output(x)
    
def getAction(policy, state):
    state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
    action_probabilities = policy(state)
    action_space_distribution = Categorical(action_probabilities)
    next_action = action_space_distribution.sample()
    return next_action.item(), action_space_distribution.log_prob(next_action)

def run():
    environment = gym.make('CartPole-v1') #MountainCar-v0
    environment = environment.unwrapped
    policy = Policy(environment.observation_space.shape[0], environment.action_space.n).to(DEVICE)
    critic = Critic(environment.observation_space.shape[0]).to(DEVICE)
    policy_opt = optim.AdamW(policy.parameters(), lr=P_ALPHA) #, momentum=0.9, nesterov=True
    critic_opt = optim.AdamW(critic.parameters(), lr=C_ALPHA)

    policy_scheduler = CosineAnnealingWarmRestarts(policy_opt, T_0=10, eta_min=1e-7)
    critic_scheduler = CosineAnnealingWarmRestarts(critic_opt, T_0=10, eta_min=1e-7)

    all_rewards = []
    critic_losses = []
    for episode in tqdm(range(NUM_EPISODES)):
        state,info = environment.reset()
        I = 1
        total_reward = 0
        episode_len = 0
       
        max_loss = 0
        while True:
            episode_len += 1
            next_action, action_prob = getAction(policy, state)
            next_state, reward, terminal, truncated, info  = environment.step(next_action)

            curr_state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            curr_state_val = critic(curr_state_tensor)
            next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(DEVICE)
            next_state_val = critic(next_state_tensor)

            if terminal:
                next_state_val = torch.tensor([0]).float().unsqueeze(0).to(DEVICE)

            total_reward += reward



            critic_loss = F.mse_loss(reward + (DISCOUNT * next_state_val), curr_state_val)
            cur_loss = critic_loss
            max_loss = max(max_loss,cur_loss.detach().item())
            critic_opt.zero_grad()
            critic_loss.requires_grad_()
            critic_loss.backward()
            critic_opt.step()


            curr_state_val = critic(curr_state_tensor)
            next_state_val = critic(next_state_tensor)
            delta = reward + (DISCOUNT * next_state_val.detach().item()) - curr_state_val.detach().item()
            poicy_loss = - action_prob * delta * I
            policy_opt.zero_grad()
            poicy_loss.requires_grad_()
            poicy_loss.backward(retain_graph=True)
            policy_opt.step()



            if terminal or truncated or episode_len>500:
                break


            I = I * DISCOUNT
            state = next_state
            policy_scheduler.step(episode)
            critic_scheduler.step(episode)


        critic_losses.append(max_loss)
        all_rewards.append(total_reward)

    environment.close()
    return all_rewards,critic_losses

def plot_averages(J_thetas):
    J_thetas = np.array(J_thetas)
    J_thetas_mean = np.mean(J_thetas,axis=0)
    J_std = np.std(J_thetas,axis=0)

    fig, ax = plt.subplots()
    ax.plot(J_thetas_mean,label='Mean')
    plt.fill_between(np.arange(len(J_thetas_mean)), J_thetas_mean - J_std, J_thetas_mean + J_std, alpha=0.3,label='Standard Deviation')
    ax.set_xlabel("Episode Number")
    ax.set_ylabel("Total Reward(Across trials)")
    ax.set_title("Average over 20 trials")
    ax.legend()

def main():
    all_rewards_over_five = []
    losses = []
    for x in range(20):
        reward,loss = run()
        all_rewards_over_five.append(reward)
        losses.append(loss)
    plot_averages(all_rewards_over_five)

if __name__ == "__main__":
    main()


