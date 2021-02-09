import ray
import argparse

from env.inventory_env import InventoryManageEnv
from env.inventory_utils import Utils
from scheduler.inventory_random_policy import ProducerBaselinePolicy, BaselinePolicy

from scheduler.inventory_eoq_policy import ConsumerEOQPolicy as ConsumerBaselinePolicy

from config.inventory_config import env_config
from utility.visualization import visualization


from scheduler.inventory_dqn_baseline import ConsumerDQNTorchPolicy
from scheduler.Trainer import Trainer
import numpy as np
import os
from utility.tensorboard import TensorBoard
from scheduler.inventory_representation_learning_dqn import ConsumerRepresentationLearningDQNTorchPolicy
import torch
import random

import global_config
# Configuration ===============================================================================

dqn_config_default = {
    "env": InventoryManageEnv,
    "gamma": 0.99,
    "min_replay_history": 20000,
    "update_period": 4,
    "target_update_period": 2000,
    "epsilon_train": 0.01,
    # "epsilon_eval": 0.001,
    "lr": 0.0003,
    # "num_iterations": 200,
    "training_steps": 25000,
    # "eval_steps": 500, eval one episode
    # "max_steps_per_episode": 60,
    "replay_capacity": 1000000,
    "batch_size": 2048,
    "double_q": True,
    "use_unc_part": True,
    "pretrain": False,
    "fixed_uncontrollable_param": False,
    "use_cnn_state": False,
    "pretrain_epoch": 100,
    "embeddingmerge": 'cat',  # 'cat' or 'dot'
    "activation_func": 'sigmoid',  # 'relu', 'sigmoid', 'tanh'
    "use_bn": False,
    # "nstep": 1,

}
env_config_for_rendering = env_config.copy()
episod_duration = env_config_for_rendering['episod_duration']
env = InventoryManageEnv(env_config_for_rendering)

all_policies = {
    'baseline_producer': ProducerBaselinePolicy(env.observation_space, env.action_space_producer, BaselinePolicy.get_config_from_env(env)),
    'baseline_consumer': ConsumerBaselinePolicy(env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env)),
    'dqn_store_consumer':ConsumerBaselinePolicy(env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env)),
}
def policy_map_fn(agent_id):
    if Utils.is_producer_agent(agent_id):
        return 'baseline_producer', False
    else:
        if agent_id.startswith('SKUStoreUnit') or agent_id.startswith('OuterSKUStoreUnit'):
            return 'dqn_store_consumer', True
        else:
            return 'baseline_consumer', False

def print_result(result, writer, iter):
    rewards_all = result['rewards_all']
    episode_reward_all = result['episode_reward_all']
    policies_to_train_loss = result['policies_to_train_loss']
    policies_to_train_qvalue = result['policies_to_train_qvalue']
    action_distribution = result['action_distribution']
    policies_to_train_pred_loss = result['policies_to_train_pred_loss']
    all_reward = []
    all_episode_reward = []
    all_loss = []
    all_pred_loss = []
    all_qvalue = []
    all_action_distribution = [0 for i in range(9)]
    for agent_id, rewards in rewards_all.items():
        _, tmp = policy_map_fn(agent_id)
        if tmp:
            all_reward.extend(rewards)
            all_loss.extend(policies_to_train_loss[agent_id])
            all_pred_loss.extend(policies_to_train_pred_loss[agent_id])
            all_episode_reward.extend(episode_reward_all[agent_id])
            all_qvalue.extend(policies_to_train_qvalue[agent_id])
            all_action_distribution = [all_action_distribution[i] + action_distribution[agent_id][i] for i in range(9)]
    print(f"all_step: {result['all_step']}, average episode step: {result['episode_step']}")
    print(f"step_reward    max: {np.max(all_reward):13.6f} mean: {np.mean(all_reward):13.6f} min: {np.min(all_reward):13.6f}")
    print(f"episode_reward max: {np.max(all_episode_reward):13.6f} mean: {np.mean(all_episode_reward):13.6f} min: {np.min(all_episode_reward):13.6f}")
    print(f"loss           max: {np.max(all_loss):13.6f} mean: {np.mean(all_loss):13.6f} min: {np.min(all_loss):13.6f}")
    print(f"pred loss      max: {np.max(all_pred_loss):13.6f} mean: {np.mean(all_pred_loss):13.6f} min: {np.min(all_pred_loss):13.6f}")
    print(f"qvalue         max: {np.max(all_qvalue):13.6f} mean: {np.mean(all_qvalue):13.6f} min: {np.min(all_qvalue):13.6f}")
    print(f"action dist    {all_action_distribution}")
    print(f"epsilon        {result['epsilon']}")

    writer.add_scalar('train/step_reward', np.mean(all_reward), iter)
    writer.add_scalar('train/episode_reward', np.mean(all_episode_reward), iter)
    writer.add_scalar('train/loss', np.mean(all_loss), iter)
    writer.add_scalar('train/qvalue', np.mean(all_qvalue), iter)
    writer.add_scalar('train/epsilon', result['epsilon'], iter)
    writer.add_scalar('train/loss_pred', np.mean(all_pred_loss), iter)

    sum_action = sum(all_action_distribution)
    for i in range(9):
        writer.add_scalar(f'action/{i}', all_action_distribution[i]/sum_action, iter)
    return np.mean(all_reward)

def print_eval_result(result, writer, iter):
    print("  == evaluation result == ")
    rewards_all = result['rewards_all']
    episode_reward_all = result['episode_reward_all']
    all_reward = []
    all_episode_reward = []
    sum_agent = 0
    for agent_id, rewards in rewards_all.items():
        _, tmp = policy_map_fn(agent_id)
        if tmp:
            sum_agent += 1
            if all_reward == []:
                all_reward = [x for x in rewards]
                all_episode_reward = [x for x in episode_reward_all[agent_id]]
            else :
                all_reward = [a+b for a, b in zip(all_reward, rewards)]
                all_episode_reward = [a+b for a,b in zip(all_episode_reward, episode_reward_all[agent_id])]
    all_reward = [x/sum_agent for x in all_reward]
    all_episode_reward = [x/sum_agent for x in all_episode_reward]
    print(f"all_step: {result['all_step']}, average episode step: {result['episode_step']}")
    print(f"step_reward    max: {np.max(all_reward):13.6f} mean: {np.mean(all_reward):13.6f} min: {np.min(all_reward):13.6f}")
    print(f"episode_reward max: {np.max(all_episode_reward):13.6f} mean: {np.mean(all_episode_reward):13.6f} min: {np.min(all_episode_reward):13.6f}")
    print(f"profit(dollar) {result['profit']}")
    print(f"epsilon        {result['epsilon']}")

    writer.add_scalar('train/eval_step_reward', np.mean(all_reward), iter)
    writer.add_scalar('train/eval_episode_reward', np.mean(all_episode_reward), iter)
    writer.add_scalar('train/eval_epsilon', result['epsilon'], iter)
    writer.add_scalar('train/eval_profit', result['profit'], iter)

    return result['profit']

def print_eval_on_trainingset_result(result, writer, iter):
    print("  == evaluation on training set result == ")
    rewards_all = result['rewards_all']
    episode_reward_all = result['episode_reward_all']
    all_reward = []
    all_episode_reward = []
    sum_agent = 0
    for agent_id, rewards in rewards_all.items():
        _, tmp = policy_map_fn(agent_id)
        if tmp:
            sum_agent += 1
            if all_reward == []:
                all_reward = [x for x in rewards]
                all_episode_reward = [x for x in episode_reward_all[agent_id]]
            else :
                all_reward = [a+b for a, b in zip(all_reward, rewards)]
                all_episode_reward = [a+b for a,b in zip(all_episode_reward, episode_reward_all[agent_id])]
    all_reward = [x/sum_agent for x in all_reward]
    all_episode_reward = [x/sum_agent for x in all_episode_reward]
    print(f"all_step: {result['all_step']}, average episode step: {result['episode_step']}")
    print(f"step_reward    max: {np.max(all_reward):13.6f} mean: {np.mean(all_reward):13.6f} min: {np.min(all_reward):13.6f}")
    print(f"episode_reward max: {np.max(all_episode_reward):13.6f} mean: {np.mean(all_episode_reward):13.6f} min: {np.min(all_episode_reward):13.6f}")
    print(f"profit(dollar) {result['profit']}")
    print(f"epsilon        {result['epsilon']}")

    writer.add_scalar('train/eval_on_trainingset_step_reward', np.mean(all_reward), iter)
    writer.add_scalar('train/eval_on_trainingset_episode_reward', np.mean(all_episode_reward), iter)
    writer.add_scalar('train/eval_on_trainingset_profit', result['profit'], iter)
    #writer.add_scalar('train/eval_epsilon', result['epsilon'], iter)
'''
TODO: visualize training set 
'''

def train_dqn(args):
    if not os.path.exists('train_log'):
        os.mkdir('train_log')
    writer = TensorBoard(f'train_log/{args.run_name}')

    dqn_config = dqn_config_default.copy()

    dqn_config.update({
        "batch_size": 1024,
        "min_replay_history": 10000,
        "training_steps": 1500,
        "lr": 0.0003,
        "target_update_period": 1000,
        "gamma": args.gamma,
        "use_unc_part": args.use_unc_part,
        "pretrain": args.pretrain,
        "fixed_uncontrollable_param": args.fixed_uncontrollable_param,
        "use_cnn_state": args.use_cnn_state,
        "pretrain_epoch": args.pretrain_epoch,
        "embeddingmerge": args.embeddingmerge, # 'cat' or 'dot'
        "activation_func": args.activation_func, # 'relu', 'sigmoid', 'tanh'
        "use_bn": args.use_bn,
        "weight_decay": args.weight_decay,

        # data augmentation setting
        "train_augmentation": args.train_augmentation,
        "noise_scale": args.noise_scale,
        "sparse_scale": args.sparse_scale,
        # ....
    })

    print(dqn_config)

    if dqn_config["use_cnn_state"]:
        global_config.use_cnn_state = True

    if args.training_length != 4*365:
        global_config.training_length = args.training_length

    if args.oracle:
        global_config.oracle = True
        if dqn_config['pretrain']:
            raise Exception("dqn oracle does not support pretrain")
        if dqn_config['train_augmentation'] != 'none':
            raise Exception("dqn oracle should not use augmentation")

    # if args.env_demand_noise != 'none':
    #     env.env_config['init'] = 'rst'


    # all_policies['dqn_store_consumer'] = ConsumerDQNTorchPolicy(env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env), dqn_config)
    all_policies['dqn_store_consumer'] = ConsumerRepresentationLearningDQNTorchPolicy(env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env), dqn_config)

    obss = env.reset()
    agent_ids = obss.keys()
    policies = {}
    policies_to_train = []

    for agent_id in agent_ids:
        policy, if_train = policy_map_fn(agent_id)
        policies[agent_id] = all_policies[policy]
        if if_train:
            policies_to_train.append(agent_id)

    dqn_trainer = Trainer(env, policies, policies_to_train, dqn_config)
    max_mean_reward = - 10000000000
    if dqn_config['pretrain']:
        # global_config.random_noise = 'none'

        print('start load data ...')
        dqn_trainer.load_data(eval=False)
        dqn_trainer.load_data(eval=True)
        # now_mean_reward = print_result(result, writer, -1)
        print('load success!')
        print('start pre-training ...')
        all_policies['dqn_store_consumer'].pre_train(writer)
        print('pre-training success!')

    debug = False

    for i in range(args.num_iterations):

        # if args.env_demand_noise != 'none':
        #     global_config.random_noise = args.env_demand_noise
        result = dqn_trainer.train(i)
        print_result(result, writer, i)
        # global_config.random_noise = 'none'

        eval_on_trainingset_result = dqn_trainer.eval(i, eval_on_trainingset=True)
        print_eval_on_trainingset_result(eval_on_trainingset_result, writer, i)

        eval_result = dqn_trainer.eval(i)
        eval_mean_reward = print_eval_result(eval_result, writer, i)

        if eval_mean_reward > max_mean_reward or debug:
            max_mean_reward = max(max_mean_reward, eval_mean_reward)
            dqn_trainer.save(args.run_name, i)
            # print("checkpoint saved at", checkpoint)
            visualization(env, policies, i, args.run_name)

bool_map = lambda s: {'True': True, 'False': False}[s]

parser = argparse.ArgumentParser()

# random init stock, demand noise uniform([-1, 1])*sale_mean, forecasting soft feature
parser.add_argument("--run-name", type=str, default='test')

parser.add_argument("--lr", type=float, default=0.0003)
parser.add_argument("--use-unc-part", default=True, type=bool_map)
parser.add_argument("--pretrain", default=False, type=bool_map)
parser.add_argument("--fixed-uncontrollable-param", default=False, type=bool_map)
parser.add_argument("--use-cnn-state", default=False, type=bool_map)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--pretrain-epoch", default=300, type=int)
parser.add_argument("--embeddingmerge", default='cat', type=str, help="'cat' or 'dot'")
parser.add_argument("--activation-func", default='sigmoid', type=str, help="'sigmoid', 'relu' or 'tanh'")
parser.add_argument("--use-bn", default=False, type=bool_map)
parser.add_argument("--num-iterations", default=200, type=int)
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--weight-decay", default=0, type=float)

# parser.add_argument("--env-demand-noise", default='none', type=str, help="simulator noise")
parser.add_argument("--train-augmentation", default='none', type=str, help="pretrain and train augmentation")
parser.add_argument("--sparse-scale", default=0.1, type=float)
parser.add_argument("--noise-scale", default=0.2, type=float)
parser.add_argument("--training-length", default=365*4, type=int)

parser.add_argument("--oracle", default=False, type=bool_map)

# parser.add_argument("--only-pretrain", default=False, type=bool_map)



if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    print(args)

    ray.init()
    train_dqn(args)

