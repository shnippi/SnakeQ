import os
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from dotenv import load_dotenv

load_dotenv()


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, chkpt_dir='./models'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(5, 5), stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.prelu_act = nn.PReLU()

        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device(os.environ.get('DEVICE') if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = self.prelu_act(self.pool(self.conv1(state))) # torch.Size([1, 3, 3, 3])
        x = x.view(-1, 27)
        dist = self.actor(x)
        dist = Categorical(dist)

        return dist

    def save_checkpoint(self):
        if not os.path.exists('./models'):
            os.makedirs('./models')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
                 chkpt_dir='./models'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(5, 5), stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.prelu_act = nn.PReLU()

        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device(os.environ.get('DEVICE') if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = self.prelu_act(self.pool(self.conv1(state)))
        x = x.view(-1, 27)
        value = self.critic(x)

        return value

    def save_checkpoint(self):
        if not os.path.exists('./models'):
            os.makedirs('./models')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
