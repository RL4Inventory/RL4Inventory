from scheduler.inventory_random_policy import BaselinePolicy, ConsumerBaselinePolicy
from utility.replay_memory import replay_memory
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
from scheduler.forecasting_model import Forecasting_model
from scheduler.data_augmentation import demand_augmentation

mixed_dqn_net_config_example ={
    "controllable_state_num": 79,
    "action_num": 9,
    "controllable_hidden_size": 32,

    "uncontrollable_state_num": 31,
    "uncontrollable_pred_num": 3,
    "uncontrollable_hidden_size": 32,

    "dqn_hidden_size": 32,
    "dqn_embedding_size": 32,

    "uncontrollable_use_cnn": False,
    "cnn_in_channel": 7,
    "cnn_state_length" : 21,

    'fixed_uncontrollable_param': True,
    'embeddingmerge': 'cat', # 'cat' or 'dot'
    'activation_func': 'sigmoid', # 'sigmoid', 'relu', 'tanh'
    'use_bn': True

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
        self.fixed_uncontrollable_param = config['fixed_uncontrollable_param']
        self.use_bn = config['use_bn']
        if config['activation_func'] == 'relu':
            self.act_func = F.relu
        elif config['activation_func'] == 'sigmoid':
            self.act_func = torch.sigmoid
        elif config['activation_func'] == 'tanh':
            self.act_func = torch.tanh
        else:
            raise Exception("embedding merge error, three legal option 'relu', 'sigmoid' or 'tanh'")

        self.embedding_merge = config['embeddingmerge']
        if self.embedding_merge == 'cat':
            self.dqn_fc1 = nn.Linear(self.dqn_embedding_size*2, self.dqn_hidden_size)
        elif self.embedding_merge == 'dot':
            self.dqn_fc1 = nn.Linear(self.dqn_embedding_size, self.dqn_hidden_size)
        else:
            raise Exception("embedding merge error, two legal option 'cat' or 'dot'")
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
        if self.use_bn: x = self.uncontrollable_embed_bn1(x)
        x = self.act_func(x)
        x = self.uncontrollable_embed_fc2(x)
        if self.use_bn: x = self.uncontrollable_embed_bn2(x)
        x = self.act_func(x)
        x = self.uncontrollable_pred(x)
        return x

    def __uncontrollable_embedding(self, uncontrollable_state):
        x = self.uncontrollable_embed_fc1(uncontrollable_state)
        if self.use_bn: x = self.uncontrollable_embed_bn1(x)
        x = self.act_func(x)
        x = self.uncontrollable_embed_fc2(x)
        if self.use_bn: x = self.uncontrollable_embed_bn2(x)
        x = self.act_func(x)
        return x

    def uncontrollable_embedding(self, uncontrollable_state):
        if self.fixed_uncontrollable_param:
            with torch.no_grad():
                return self.__uncontrollable_embedding(uncontrollable_state)
        else:
            return self.__uncontrollable_embedding(uncontrollable_state)


    def controllable_embedding(self, controllable_state):
        cs = self.controllable_embed_fc1(controllable_state)
        if self.use_bn: cs = self.controllable_embed_bn1(cs)
        cs = self.act_func(cs)
        return cs

    def forward(self, controllable_state, uncontrollable_state=None):
        cs_embedding = self.controllable_embedding(controllable_state)
        if uncontrollable_state != None:
            ucs_emebdding = self.uncontrollable_embedding(uncontrollable_state)
            if self.embedding_merge == 'cat':
                cs_embedding = torch.cat([cs_embedding, ucs_emebdding], -1)
            elif self.embedding_merge == 'dot':
                cs_embedding = cs_embedding * ucs_emebdding
            else:
                raise Exception("embedding merge error, two legal option 'cat' or 'dot'")

        x = self.dqn_fc1(cs_embedding)
        if self.use_bn: x = self.dqn_bn1(x)
        x = self.act_func(x)
        x = self.dqn_fc2(x)

        return x

class mixed_dqn_unc_cnn_net(nn.Module):
    def __init__(self, config):
        """
        Initialize a deep Q-learning network for testing algorithm
            in_features: number of features of input.
            num_actions: number of action-value to output, one-to-one correspondence to action in game.
        """
        super(mixed_dqn_unc_cnn_net, self).__init__()
        # this is part of dqn
        self.dqn_hidden_size = config['dqn_hidden_size']
        self.dqn_embedding_size = config['dqn_embedding_size']
        self.action_num = config['action_num']
        self.fixed_uncontrollable_param = config['fixed_uncontrollable_param']
        self.use_bn = config['use_bn']

        if config['activation_func'] == 'relu':
            self.act_func = F.relu
        elif config['activation_func'] == 'sigmoid':
            self.act_func = torch.sigmoid
        elif config['activation_func'] == 'tanh':
            self.act_func = torch.tanh
        else:
            raise Exception("embedding merge error, three legal option 'relu', 'sigmoid' or 'tanh'")
        self.embedding_merge = config['embeddingmerge']
        if self.embedding_merge == 'cat':
            self.dqn_fc1 = nn.Linear(self.dqn_embedding_size*2, self.dqn_hidden_size)
        elif self.embedding_merge == 'dot':
            self.dqn_fc1 = nn.Linear(self.dqn_embedding_size, self.dqn_hidden_size)
        else:
            raise Exception("embedding merge error, two legal option 'cat' or 'dot'")
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
        self.cnn_state_in_channel = config['cnn_in_channel']
        self.cnn_state_length = config['cnn_state_length']

        self.uncontrollable_embed_conv1 = nn.Conv1d(in_channels=7, out_channels=16, kernel_size=3, padding=1)
        self.uncontrollable_embed_conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.uncontrollable_embed_bn1 = nn.BatchNorm1d(self.cnn_state_length)
        self.uncontrollable_embed_bn2 = nn.BatchNorm1d(self.cnn_state_length)

        self.uncontrollable_pred = nn.Linear(self.dqn_embedding_size, self.uncontrollable_pred_num)

    def __uncontrollable_embedding(self, uncontrollable_state):
        x = self.uncontrollable_embed_conv1(uncontrollable_state) # batch x 5 x 21 -> batch x 16 x 21
        x = x.permute(0, 2, 1)
        if self.use_bn: x = self.uncontrollable_embed_bn1(x)
        x = x.permute(0, 2, 1)
        x = self.act_func(x)
        x = self.uncontrollable_embed_conv2(x)          # batch x 16 x 21 -> batch x 32 x 21
        x = x.permute(0, 2, 1)
        if self.use_bn: x = self.uncontrollable_embed_bn2(x)
        x = x.permute(0, 2, 1)
        x = x.mean(-1)
        x = self.act_func(x)
        return x

    def uncontrollable_pred_head(self, uncontrollable_state):
        x = self.__uncontrollable_embedding(uncontrollable_state)
        x = self.uncontrollable_pred(x)
        return x

    def uncontrollable_embedding(self, uncontrollable_state):
        if self.fixed_uncontrollable_param:
            with torch.no_grad():
                return self.__uncontrollable_embedding(uncontrollable_state)
        else:
            return self.__uncontrollable_embedding(uncontrollable_state)


    def controllable_embedding(self, controllable_state):
        cs = self.controllable_embed_fc1(controllable_state)
        if self.use_bn: cs = self.controllable_embed_bn1(cs)
        cs = self.act_func(cs)
        return cs

    def forward(self, controllable_state, uncontrollable_state=None):
        cs_embedding = self.controllable_embedding(controllable_state)
        if uncontrollable_state != None:
            ucs_emebdding = self.uncontrollable_embedding(uncontrollable_state)
            if self.embedding_merge == 'cat':
                cs_embedding = torch.cat([cs_embedding, ucs_emebdding], -1)
            elif self.embedding_merge == 'dot':
                cs_embedding = cs_embedding * ucs_emebdding
            else:
                raise Exception("embedding merge error, two legal option 'cat' or 'dot'")

        x = self.dqn_fc1(cs_embedding)
        if self.use_bn: x = self.dqn_bn1(x)
        x = self.act_func(x)
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
            "uncontrollable_state_num": 21*7, # 31
            "uncontrollable_pred_num": 6,
            'fixed_uncontrollable_param': dqn_config['fixed_uncontrollable_param'],
            'uncontrollable_use_cnn': dqn_config['use_cnn_state'],
            'embeddingmerge': dqn_config['embeddingmerge'],
            'activation_func': dqn_config['activation_func'],
            'use_bn': dqn_config['use_bn'],
        })
        self.num_states =  int(np.product(observation_space.shape))
        self.num_actions = int(action_space.n)
        print(f'dqn state space:{self.num_states}, action space:{self.num_actions}')

        self.use_unc_part = dqn_config['use_unc_part']
        #self.pre_train = dqn_config['pretrain']
        self.fixed_uncontrollable_param = dqn_config['pretrain']

        if dqn_config['use_cnn_state']:
            dqn_net = mixed_dqn_unc_cnn_net
        else:
            dqn_net = mixed_dqn_net
        self.eval_net = dqn_net(mixed_dqn_config)
        self.target_net = dqn_net(mixed_dqn_config)

        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter = 0
        self.memory = replay_memory(dqn_config['replay_capacity'])
        self.eval_memory = replay_memory(dqn_config['replay_capacity'])

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=dqn_config['lr'], weight_decay=dqn_config['weight_decay'])
        self.loss_func = nn.SmoothL1Loss()

        self.rand_action = 0
        self.greedy_action = 0
        self.evaluation = False # only for epsilon-greedy

        # augmentation setting
        self.train_augmentation = dqn_config['train_augmentation']
        self.demand_augmentation = demand_augmentation(noise_type=dqn_config['train_augmentation'],
                                                       noise_scale=dqn_config['noise_scale'],
                                                       sparse_scale=dqn_config['sparse_scale'])

    def switch_mode(self, eval=False): # only for epsilon-greedy
        self.evaluation = eval

    def choose_action(self, state, infos):

        self.eval()
        unc_state = None
        if self.use_unc_part:
            if self.dqn_config['use_cnn_state']:
                key = 'uncontrollable_part_state_cnn'
            else:
                key = 'uncontrollable_part_state'
            unc_state = torch.tensor(infos[key], dtype=torch.float32).unsqueeze(0)
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.random() >= self.epsilon or self.evaluation: # greedy policy
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

    def store_transition(self, state, action, reward, next_state, done, unc_state, unc_pred, next_unc_state, agent_id, eval=False):

        state = torch.tensor(state.copy(), dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        reward = torch.tensor(float(reward), dtype=torch.float32)
        next_state = torch.tensor(next_state.copy(), dtype=torch.float32)
        done = torch.tensor(float(done), dtype=torch.float32)
        unc_state = torch.tensor(unc_state.copy(), dtype=torch.float32)
        unc_pred = torch.tensor(unc_pred.copy(), dtype=torch.float32)
        next_unc_state = torch.tensor(next_unc_state.copy(), dtype=torch.float32)
        if eval:
            self.eval_memory.push([state, action, reward, next_state, done, unc_state, unc_pred, next_unc_state])
        else:
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
        self.train(fix_uncontrollable_part=self.fixed_uncontrollable_param)

        batch_unc_state = self.demand_augmentation.augment(batch_unc_state)
        batch_next_unc_state = self.demand_augmentation.augment(batch_next_unc_state)

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

        batch_unc_state = self.demand_augmentation.augment(batch_unc_state)

        unc_pred_from_net = self.eval_net.uncontrollable_pred_head(batch_unc_state)
        loss = self.loss_func(unc_pred_from_net, batch_unc_pred)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_embedding(self, batch_size=1024):
        self.eval()
        batch_state, batch_action, batch_reward, batch_next_state, batch_done, \
            batch_unc_state, batch_unc_pred, batch_next_unc_state = self.eval_memory.sample(batch_size)
        with torch.no_grad():
            unc_pred_from_net = self.eval_net.uncontrollable_pred_head(batch_unc_state)
            loss = self.loss_func(unc_pred_from_net, batch_unc_pred)
        return loss.item()


    def train_one_epoch(self, iter = 50):
        loss = []
        for i in range(iter):
            loss.append(self.learn_embeding())
        return np.mean(loss)

    def eval_one_epoch(self, iter = 30):
        loss = []
        for i in range(iter):
            loss.append(self.eval_embedding())
        return np.mean(loss)

    def pre_train(self, writer):
        # todo evaluation
        for i in range(self.dqn_config['pretrain_epoch']):
            eval_loss = self.eval_one_epoch()
            train_loss = self.train_one_epoch()
            print(f"pretraining epoch {i}, before train eval loss {eval_loss} train loss {train_loss}")
            writer.add_scalar('pretrain/train_loss', train_loss, i)
            writer.add_scalar('pretrain/eval_loss', eval_loss, i)
        # exit(0) #!!!!!!!!!!!!!!!
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def _action(self, state, state_info):
        return self.choose_action(state, state_info)

    def save_param(self, name):
        if not os.path.exists('model'):
            os.mkdir('model')
        torch.save(self.eval_net.state_dict(), f'model/{name}.pkl')

    def load_param(self, name):
        self.policy.load_state_dict(torch.load(f'model/{name}.pkl'))

    def train(self, fix_uncontrollable_part=False):
        self.eval_net.train()
        self.target_net.train()
        # TODO
        if fix_uncontrollable_part:
            self.eval_net.uncontrollable_embed_bn1.eval()
            self.eval_net.uncontrollable_embed_bn2.eval()
            self.target_net.uncontrollable_embed_bn1.eval()
            self.target_net.uncontrollable_embed_bn2.eval()


    def eval(self):
        self.eval_net.eval()
        self.target_net.eval()



