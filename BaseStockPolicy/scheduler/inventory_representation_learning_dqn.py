from scheduler.inventory_random_policy import BaselinePolicy, ConsumerBaselinePolicy
from utility.replay_memory import replay_memory
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
from scheduler.forecasting_model import Forecasting_model


mixed_dqn_net_config_example ={
    "controllable_state_num": 79,
    "action_num": 9,
    "controllable_hidden_size": 32,

    "uncontrollable_state_num": 31,
    "uncontrollable_pred_num": 3,
    "uncontrollable_hidden_size": 32,

    "dqn_hidden_size": 32,
    "dqn_embedding_size": 32,
}

class mixed_dqn_net(nn.Module):
    def __init__(self, config):
        """
        Initialize a deep Q-learning network for testing algorithm
            in_features: number of features of input.
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(mixed_dqn_net, self).__init__()
        # this is part of dqn
        self.dqn_hidden_size = config['dqn_hidden_size']
        self.dqn_embedding_size = config['dqn_embedding_size']
        self.action_num = config['action_num']

        self.dqn_fc1 = nn.Linear(self.dqn_embedding_size, self.dqn_hidden_size)
        self.dqn_fc2 = nn.Linear(self.dqn_hidden_size, self.action_num)
        self.dqn_bn1 = nn.BatchNorm1d(self.dqn_hidden_size)

        # this is part of controllable state
        self.controllable_state_num = config['controllable_state_num']

        self.controllable_embed_fc1 = nn.Linear(self.controllable_state_num, self.dqn_embedding_size)
        self.controllable_embed_bn1 = nn.BatchNorm1d(self.dqn_embedding_size)

        # this is part of uncontrollable state
        self.uncontrollable_state_num = config['uncontrollable_state_num']
        self.uncontrollable_pred_num = config['uncontrollable_pred_num']
        self.uncontrollable_hidden_size = config['uncontrollable_hidden_size']

        self.uncontrollable_embed_fc1 = nn.Linear(self.uncontrollable_state_num, self.uncontrollable_hidden_size)
        self.uncontrollable_embed_fc2 = nn.Linear(self.uncontrollable_hidden_size, self.dqn_embedding_size)

        self.uncontrollable_embed_bn1 = nn.BatchNorm1d(self.uncontrollable_hidden_size)
        self.uncontrollable_embed_bn2 = nn.BatchNorm1d(self.dqn_embedding_size)

        self.uncontrollable_pred = nn.Linear(self.dqn_embedding_size, self.uncontrollable_pred_num)


    def uncontrollable_pred_head(self, uncontrollable_state):
        x = self.uncontrollable_embed_fc1(uncontrollable_state)
        x = self.uncontrollable_embed_bn1(x)
        x = F.sigmoid(x)
        x = self.uncontrollable_embed_fc2(x)
        x = self.uncontrollable_embed_bn2(x)
        x = F.sigmoid(x)
        x = self.uncontrollable_pred(x)
        return x

    def uncontrollable_embedding(self, uncontrollable_state):
        with torch.no_grad():
            x = self.uncontrollable_embed_fc1(uncontrollable_state)
            x = self.uncontrollable_embed_bn1(x)
            x = F.sigmoid(x)
            x = self.uncontrollable_embed_fc2(x)
            x = self.uncontrollable_embed_bn2(x)
            x = F.sigmoid(x)
            return x

    def controllable_embedding(self, controllable_state):
        cs = self.controllable_embed_fc1(controllable_state)
        cs = self.controllable_embed_bn1(cs)
        cs = F.sigmoid(cs)
        return cs

    def forward(self, controllable_state, uncontrollable_state=None):
        cs_embedding = self.controllable_embedding(controllable_state)
        if uncontrollable_state != None:
            ucs_emebdding = self.uncontrollable_embedding(uncontrollable_state)
            cs_embedding = cs_embedding * ucs_emebdding

        x = self.dqn_fc1(cs_embedding)
        x = self.dqn_bn1(x)
        x = F.sigmoid(x)
        x = self.dqn_fc2(x)

        return x

'''
TODO : train and eval in uncontrollable part batchnorm
'''

class ConsumerRepresentationLearningDQNTorchPolicy(BaselinePolicy):


    def __init__(self, observation_space, action_space, config, dqn_config):
        BaselinePolicy.__init__(self, observation_space, action_space, config)

        self.dqn_config = dqn_config
        self.epsilon = 1
        mixed_dqn_config = mixed_dqn_net_config_example.copy()
        mixed_dqn_config.update({
            "controllable_state_num" : int(np.product(observation_space.shape)),
            "action_num": int(action_space.n),
            "uncontrollable_state_num": 31,
            "uncontrollable_pred_num": 3,
        })
        self.num_states =  int(np.product(observation_space.shape))
        self.num_actions = int(action_space.n)
        print(f'dqn state space:{self.num_states}, action space:{self.num_actions}')
        self.use_unc_part = dqn_config['use_unc_part']

        self.eval_net = mixed_dqn_net(mixed_dqn_net_config_example)
        self.target_net = mixed_dqn_net(mixed_dqn_net_config_example)

        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter = 0
        self.memory = replay_memory(dqn_config['replay_capacity'])

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=dqn_config['lr'])
        self.loss_func = nn.SmoothL1Loss()

        self.rand_action = 0
        self.greedy_action = 0

    def choose_action(self, state, infos):

        self.eval()
        unc_state = None
        if self.use_unc_part:
            unc_state = torch.tensor(infos['uncontrollable_part_state'], dtype=torch.float32).unsqueeze(0)
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.random() >= self.epsilon: # greedy policy
            self.greedy_action += 1
            with torch.no_grad():
                action_value = self.eval_net(state, unc_state)
                action = torch.max(action_value, 1)[1].data.numpy()
                action = action[0]
        else: # random policy
            self.rand_action += 1
            action = np.random.randint(self.num_actions)
            action = action
        return action

    def store_transition(self, state, action, reward, next_state, done, unc_state, unc_pred, next_unc_state, agent_id):

        state = torch.tensor(state.copy(), dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(float(reward), dtype=torch.float32)
        next_state = torch.tensor(next_state.copy(), dtype=torch.float32)
        done = torch.tensor(float(done), dtype=torch.float32)
        unc_state = torch.tensor(unc_state.copy(), dtype=torch.float32)
        unc_pred = torch.tensor(unc_pred.copy(), dtype=torch.float32)
        next_unc_state = torch.tensor(next_unc_state.copy(), dtype=torch.float32)

        self.memory.push([state, action, reward, next_state, done, unc_state, unc_pred, next_unc_state])


    def learn(self, batch_size):
        if len(self.memory) < self.dqn_config['min_replay_history']:
            return 0, 0, 0

        # update the parameters
        if self.learn_step_counter % self.dqn_config['target_update_period'] == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        self.epsilon = 0.999 * self.epsilon + 0.001 * self.dqn_config['epsilon_train']

        batch_state, batch_action, batch_reward, batch_next_state, batch_done, \
            batch_unc_state, batch_unc_pred, batch_next_unc_state = self.memory.sample(batch_size)
        if not self.use_unc_part:
            batch_unc_state, batch_unc_pred, batch_next_unc_state = None, None, None
        self.train()

        with torch.no_grad():
            q_next = self.target_net(batch_next_state, batch_next_unc_state)
            q_next = q_next.detach()
            if self.dqn_config['double_q']:
                q_eval_next = self.eval_net(batch_next_state, batch_next_unc_state)
                q_eval_next = q_eval_next.detach()
                q_argmax = q_eval_next.max(1)[1]
            else:
                q_argmax = q_next.max(1)[1]
            q_next = q_next.gather(1, q_argmax.unsqueeze(1)).squeeze()
            q_target = batch_reward + self.dqn_config['gamma'] * (1 - batch_done) * q_next

        q_eval = self.eval_net(batch_state, batch_unc_state)
        q_eval = q_eval.gather(1, batch_action.unsqueeze(1))
        q_eval = q_eval.squeeze()

        loss = self.loss_func(q_eval, q_target)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), q_eval.mean().item(), 0

    def learn_embeding(self, batch_size=128):

        self.train()

        batch_state, batch_action, batch_reward, batch_next_state, batch_done, \
            batch_unc_state, batch_unc_pred, batch_next_unc_state = self.memory.sample(batch_size)

        unc_pred_from_net = self.eval_net.uncontrollable_pred_head(batch_unc_state)
        loss = self.loss_func(unc_pred_from_net, batch_unc_pred)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_one_epoch(self, iter = 50):
        loss = []
        for i in range(iter):
            loss.append(self.learn_embeding())
        return np.mean(loss)

    def pre_train(self):
        # todo evaluation
        for i in range(50):
            loss = self.train_one_epoch()
            print(f"pretraining epoch {i}, loss {loss}")

        self.target_net.load_state_dict(self.eval_net.state_dict())

    def _action(self, state, state_info):
        return self.choose_action(state, state_info)

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



