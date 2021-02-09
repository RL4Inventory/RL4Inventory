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
    "max_steps_per_episode": 60,
    "replay_capacity": 1000000,
    "batch_size": 2048,
    "double_q": True,
    "use_unc_part": False,
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

def train_dqn(args):
    if not os.path.exists('train_log'):
        os.mkdir('train_log')
    writer = TensorBoard(f'train_log/{args.run_name}')

    dqn_config = dqn_config_default.copy()

    dqn_config.update({
        "batch_size": 1024,
        "min_replay_history": 10000,
        "training_steps": 15000,
        "lr": 0.0003,
        "target_update_period": 1000,
        "use_unc_part": False,
        # "pretrain": False,
        # ....
    })


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
    max_mean_reward = - 1000
    debug = False
    if dqn_config['use_unc_part']:
        print('start load data ...')
        result = dqn_trainer.train('load_data', just_load_data=True)
        now_mean_reward = print_result(result, writer, -1)
        print('load success!')
        print('start pre-training ...')
        all_policies['dqn_store_consumer'].pre_train()
        print('pre-training success!')


    for i in range(args.num_iterations):
        result = dqn_trainer.train(i)
        now_mean_reward = print_result(result, writer, i)

        if now_mean_reward > max_mean_reward or debug:
            max_mean_reward = max(max_mean_reward, now_mean_reward)
            dqn_trainer.save(args.run_name, i)
            # print("checkpoint saved at", checkpoint)
            visualization(env, policies, i, args.run_name)


parser = argparse.ArgumentParser()
parser.add_argument("--torch", action="store_true")
parser.add_argument("--batch-size", type=int, default=2048)
parser.add_argument("--use-prev-action-reward", action="store_true")
parser.add_argument("--num-iterations", type=int, default=1000)
parser.add_argument("--visualization-frequency", type=int, default=100)
parser.add_argument("--run-name", type=str, default='1225_dqn_baseline')

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()
    train_dqn(args)

