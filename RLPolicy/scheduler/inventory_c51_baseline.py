from scheduler.inventory_random_policy import BaselinePolicy, ConsumerBaselinePolicy
from utility.replay_memory import replay_memory
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os

class c51_net(nn.Module):
    def __init__(self, num_states=44, num_actions=9, num_atoms=51):
        """
        Initialize a deep Q-learning network for testing algorithm
            in_features: number of features of input.
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(c51_net, self).__init__()
        self.hidden_size = 128
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.fc1 = nn.Linear(num_states, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, num_actions * num_atoms)

        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.bn2 = nn.BatchNorm1d(self.hidden_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(x.size(0), self.num_actions, self.num_atoms)
        x = F.softmax(x, 2)
        return x


class ConsumerC51TorchPolicy(BaselinePolicy):


    def __init__(self, observation_space, action_space, config, dqn_config):
        BaselinePolicy.__init__(self, observation_space, action_space, config)

        self.dqn_config = dqn_config
        self.epsilon = 1
        self.Vmin = -1
        self.Vmax = 1
        self.atoms = 51
        self.device = torch.device('cpu')

        self.num_states =  int(np.product(observation_space.shape))
        self.num_actions = int(action_space.n)
        print(f'dqn state space:{self.num_states}, action space:{self.num_actions}')

        self.eval_net = c51_net(self.num_states, self.num_actions, self.atoms)
        self.target_net = c51_net(self.num_states, self.num_actions, self.atoms)

        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter = 0
        self.memory = replay_memory(dqn_config['replay_capacity'])

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=dqn_config['lr'])
        self.support = torch.linspace(self.Vmin, self.Vmax, self.atoms)

    def choose_action(self, state, explore=True):
        self.eval()
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.random() >= self.epsilon or explore == False: # greedy policy
            with torch.no_grad():
                action_value = self.eval_net(state)
                action = (action_value * self.support).sum(2).max(1)[1]
                action = action.data.numpy()
                action = action[0]
        else: # random policy
            action = np.random.randint(self.num_actions)
            action = action
        return action

    def store_transition(self, state, action, reward, next_state, done):

        state = torch.tensor(state.copy(), dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(float(reward), dtype=torch.float32)
        next_state = torch.tensor(next_state.copy(), dtype=torch.float32)
        done = torch.tensor(float(done), dtype=torch.float32)

        self.memory.push([state, action, reward, next_state, done])

    def projection_distribution(self, next_state, reward, done):
        gam = self.dqn_config['gamma']
        with torch.no_grad():
            batch_size = next_state.size(0)
            delta_z = float(self.Vmax - self.Vmin) / (self.atoms - 1)
            support = torch.linspace(self.Vmin, self.Vmax, self.atoms).to(self.device)

            next_dist   = self.eval_net(next_state) * support
            next_action = next_dist.sum(2).max(1)[1]
            next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))

            #DoubleDQN
            next_dist   = self.target_net(next_state).gather(1, next_action).squeeze(1)

            reward  = reward.unsqueeze(1).expand_as(next_dist)
            done    = done.unsqueeze(1).expand_as(next_dist)
            # gam     = gam.expand_as(next_dist)
            support = support.unsqueeze(0).expand_as(next_dist)

            Tz = reward + (1 - done) * gam * support
            Tz = Tz.clamp(self.Vmin, self.Vmax)
            b  = (Tz - self.Vmin) / delta_z
            l  = b.floor().long()
            u  = b.ceil().long()
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            offset = torch.linspace(0, (batch_size - 1) * self.atoms, batch_size).long()\
                    .unsqueeze(1).expand(batch_size, self.atoms)
            offset = offset.to(self.device)

            proj_dist = torch.zeros(next_dist.size()).to(self.device)

            proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

            return proj_dist

    def learn(self, batch_size):
        if len(self.memory) < self.dqn_config['min_replay_history']:
            return 0, 0

        # update the parameters
        if self.learn_step_counter % self.dqn_config['target_update_period'] == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        self.epsilon = 0.999 * self.epsilon + 0.001 * self.dqn_config['epsilon_train']

        batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.memory.sample(batch_size)

        self.train()

        batch_action = batch_action.unsqueeze(1).unsqueeze(1).expand(batch_action.size(0), 1, self.atoms)
        dist_pred = self.eval_net(batch_state).gather(1, batch_action).squeeze(1)
        dist_true = self.projection_distribution(batch_next_state, batch_reward, batch_done)

        dist_pred.data.clamp_(0.001, 0.999)
        loss = - (dist_true * dist_pred.log()).sum(1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            q_eval = (dist_pred * self.support).sum(1).mean()
        return loss.item(), q_eval


    def _action(self, state, state_info):
        return self.choose_action(state)

    def save_param(self, name):
        if not os.path.exists('model'):
            os.mkdir('model')
        torch.save(self.eval_net.state_dict(), f'model/{name}.pkl')

    def load_param(self, name):
        self.policy.load_state_dict(torch.load(f'model/{name}.pkl'))

    def train(self):
        self.eval_net.train()
        self.target_net.train()

    def eval(self):
        self.eval_net.eval()
        self.target_net.eval()



