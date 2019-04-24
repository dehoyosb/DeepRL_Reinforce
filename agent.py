import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from model import Policy

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

class Agent(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, output_size=2, gamma = 1, LR = 1e-2):
        super(Agent, self).__init__()
        self.gamma = gamma
        self.policy = Policy(input_size,hidden_size,output_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
    
    def act(self, state):
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.policy(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    
    def learn(self,rewards,log_probs):
        
        discounts = [self.gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])
        
        policy_loss = []
        for log_prob in log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()