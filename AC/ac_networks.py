import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions,
                 name, chkpt_dir='./models'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        # I think this breaks if the env has a 2D state representation
        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device(os.environ.get('DEVICE') if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state, action):
        # concatenate along dim1, since dim0 is batch dim
        q1_action_value = F.relu(self.fc1(T.cat([state, action], dim=1)))
        q1_action_value = F.relu(self.fc2(q1_action_value))
        q1 = self.q1(q1_action_value)

        return q1

    def save_checkpoint(self):
        if not os.path.exists('./models'):
            os.makedirs('./models')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))