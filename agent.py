from collections import deque
import random, numpy as np
import torch
from torch import optim

from Config import Config
from q_network import QNetwork


class Agent:

    def __init__(self, state_dim, action_dim, save_dir, checkpoint=None):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.exploration_rate = Config.exploration_rate
        self.exploration_rate_decay = Config.exploration_rate_decay
        self.exploration_rate_min = Config.exploration_rate_min

        self.discount_factor = Config.discount_factor

        self.curr_step = 0

        self.save_every = Config.save_every
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.q_network = QNetwork(self.state_dim, self.action_dim).cuda()
            self.q_network = self.q_network.to(device='cuda')
        else:
            self.q_network = QNetwork(self.state_dim, self.action_dim)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=Config.lr)

        if checkpoint:
            self.load(checkpoint)

    def action(self, state):

        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = self.to_tensor(np.reshape(state, [1, self.state_dim]))
            action_values = self.q_network(state)
            action_idx = torch.argmax(action_values).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def to_tensor(self, state):
        if self.use_cuda:
            return torch.FloatTensor(state).cuda()
        return torch.FloatTensor(state)

    def update_Q_function(self, state_tensor, action, reward, next_state_tensor):
        self.optimizer.zero_grad()

        state_tensor = self.to_tensor(np.reshape(state_tensor, [1, self.state_dim]))

        q_values = self.q_network(state_tensor)
        q_value = q_values[0][action]
        next_state_tensor = self.to_tensor(np.reshape(next_state_tensor, [1, self.state_dim]))

        target_q_value = reward + self.discount_factor * torch.max(self.q_network(next_state_tensor))

        loss = torch.nn.MSELoss()(q_value, target_q_value.detach())
        loss.backward()
        self.optimizer.step()
        return loss.item(), target_q_value

    def learn(self, state, next_state, action, reward):
        if self.curr_step % self.save_every == 0:
            self.save()

        loss, q = self.update_Q_function(state_tensor=state, action=action, reward=reward, next_state_tensor=next_state)
        return q, loss

    def save(self):
        save_path = self.save_dir / f"agent_net_{int(self.curr_step // self.save_every)}.chkpt"

        torch.save(
            dict(
                model=self.q_network.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"agent saved to {save_path} at step {self.curr_step}")

    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path.name, map_location=('cuda' if self.use_cuda else 'cpu'))
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.q_network.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate
