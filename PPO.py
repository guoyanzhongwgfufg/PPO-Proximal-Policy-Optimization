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
    def __init__(self,n_stats,n_hiddens,n_actions):
        super(Policynet,self).__init__()
        self.fc1 = nn.Linear(n_stats,n_hiddens)
        self.fc2 = nn.Linear(n_hiddens,n_stats)
    
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
    def __init__(self,n_stats,n_hiddens):
        super(Valuenet,self).__init__()
        self.fc1 = nn.Linear(n_stats,n_hiddens)
        self.fc2 = nn.Linear(n_hiddens,)

    def forward(self,x):
        x = self.fc1(x) # [b,n_states]-->[b,n_hiddens]
        x = F.Relu(x)
        x =self.fc2(x)# [b,n_hiddens]-->[b,1]  evlution current work status state_value
        return x


# ---------------------------------------------------
# Build model 
# ---------------------------------------------------

class PPO:
    def __init__(self,n_stats,n_hiddens,n_actions,actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        # Instantiate the policy network
        self.actor = Policynet(n_stats,n_hiddens,n_stats).to(device)
        # Instantiate the value network
        self.critic = Valuenet(n_stats,n_hiddens,n_stats).to(device)
        # policy net optimizer
        self.actor.optimizer =  torch.optim.Adam(self.actor.paratermer(),lr=actor_lr)
        # value net optimizer
        self.critic.optimizer = torch.optim.Adam(self.actor.parameter(),lr=critic_lr)

        self.gama = gamma # discount factor
        self.lmbda = lmbda # GAE advantage scale function
        self.epoch = epochs # The data of a sequence is used for the number of training rounds
        self.eps = eps # The parameters of the truncation range in PPO
        self.device = device #

    #action choose
    def take_action(self, states):
        # Dimension transformation: [n_state] --> tensor [1, n_states]
        states = torch.tensor(states[np.newaxis, :], dtype=torch.float).to(self.device)
        # Get the probability distribution over actions from the policy network: [1, n_actions]
        probs = self.actor(states)
        # Create a categorical distribution based on the probabilities
        action_dist = torch.distributions.Categorical(probs)
        # Sample an action according to the distribution
        action = action_dist.sample().item()
        return action

    # train
    def learn(self, transition_dict):
        # Extract dataset
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device).view(-1, 1)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device).view(-1, 1)

        # Get value of next states (target)
        next_q_target = self.critic(next_states)
        # Calculate TD target
        td_target = rewards + self.gamma * next_q_target * (1 - dones)
        # Predicted value of current states
        td_value = self.critic(states)
        # TD error (delta)
        td_delta = td_target - td_value

        # Convert to NumPy for advantage estimation
        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0
        advantage_list = []

        # Generalized Advantage Estimation (GAE)
        for delta in td_delta[::-1]:  # Traverse TD deltas in reverse order
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)

        # Get log probs of taken actions under old policy
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # Perform multiple epochs of update
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # Actor and Critic losses
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            # Update parameters
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


import matplotlib.pyplot as plt
import gym
import torch

if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # ------------------------ #
    # Hyperparameters
    # ------------------------ #

    num_episodes = 300
    gamma = 0.9
    actor_lr = 1e-3
    critic_lr = 1e-2
    n_hiddens = 16
    env_name = 'CartPole-v0'
    return_list = []

    # ------------------------ #
    # Environment setup
    # ------------------------ #

    env = gym.make(env_name)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # ------------------------ #
    # Agent initialization
    # ------------------------ #

    agent = PPO(n_states=n_states,
                n_hiddens=n_hiddens,
                n_actions=n_actions,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                lmbda=0.95,
                epochs=10,
                eps=0.2,
                gamma=gamma,
                device=device
                )

    # ------------------------ #
    # Training loop (on-policy)
    # ------------------------ #

    for i in range(num_episodes):

        state = env.reset()
        done = False
        episode_return = 0

        transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
        }

        while not done:
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_return += reward

        return_list.append(episode_return)
        agent.learn(transition_dict)

        print(f'iter:{i}, return:{np.mean(return_list[-10:])}')

    # ------------------------ #
    # Plotting results
    # ------------------------ #

    plt.plot(return_list)
    plt.title('Return over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.show()


'''
This code is adapted with translated comments based on the original work available at CSDN Blog:
https://blog.csdn.net/qq_45889056/article/details/130297960
Author credit preserved. Changes include English translations and minor formatting improvements.
'''


