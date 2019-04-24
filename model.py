import torch
import torch.nn as nn
import torch.nn.functional as F

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

torch.manual_seed(0) # set random seed

class Policy(nn.Module):
    def __init__(self, input_size=4, hidden_size=16, output_size=2):
        super(Policy, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)