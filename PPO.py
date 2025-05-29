'''
@Author Yangzhong Guo
@Creat data 2025.5.28

The code is used for the model of the discrete environment
'''
import numpy as np
import  torch
from torch import nn
from torch.nn import Fuctional as F

# ---------------------------------------------------
# Build a policy network --actor
# ---------------------------------------------------

class Policynet(nn.Module):
    def __init__(self,n_stats,n_hidden,n_actions):
        super(Policynet,self).__init__()
        self.fc1 = nn.Linear(n_stats,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_stats,)
    
    def forward(self,x):
        x = self.fc1(x) # b-n_stats--->b-n_hiddens
        x = self.relu(x)
        x = self.fc2(x)  # b-n_actions
        x = self.softmax(x,dim=1)  # b-n_actions Calculate the probability of each action
        return x


# ---------------------------------------------------
# Build a Value network --critic
# ---------------------------------------------------

class Valuenet(nn.Module):
    def __init__(self,n_stats,n_hidden):
        super(Valuenet,self).__init__()
        self.fc1 = nn.Linear()
        self.fc2 = 
