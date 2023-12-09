import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import matplotlib.pyplot as plt

import torch.optim as optim
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

class Policy(nn.Module):
    def __init__(self,n_input,n_output,lr_optimizer):
        super(Policy,self).__init__()
        self.fc1 = nn.Linear(n_input,128)
        #self.fc2 = nn.Linear(30,15)
        self.output = nn.Linear(128,n_output)
        self.lr = lr_optimizer
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        #x =  F.relu(self.fc2(x))
        x = self.output(x)
        return F.softmax(x,dim=-1)
    
    def action(self,x):
        x = torch.from_numpy(x)
        x = x.to(torch.float32)
        prob = self(x)
        m = Categorical(prob)
        action = m.sample()
        return action,action.item(),m.log_prob(action)
    
    def train_policy(self,delta,probs,optim):
        policy_loss = []
        for d,p in zip(delta,probs):
            policy_loss.append(-d * p)
        optim.zero_grad()
        sum(policy_loss).backward()
        optim.step()
   

class Baseline(nn.Module):
    def __init__(self,n_input,n_output,lr_optimizer):
        super(Baseline,self).__init__()
        self.fc1 = nn.Linear(n_input,128)
        #self.fc2 = nn.Linear(30,15)
        self.output = nn.Linear(128,n_output)
        self.lr = lr_optimizer
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x
    
    def train_baseline(self,optim,G,vals):
        loss = F.mse_loss(vals,G)
        optim.zero_grad()
        loss.backward()    
        optim.step()





#make a global train function




    
    




