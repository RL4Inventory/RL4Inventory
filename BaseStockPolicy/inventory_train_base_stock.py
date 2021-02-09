'''
noted that there is a `if oracle`-`or` block in the following script
noted that policy_mode should be changed to reflect the mode we use
use oracle bs:
    run inventory_train_base_stock.py
        --buyin_gap ?: how often to place an order 
        --stop_order ?: the factor that controls the stop order point
    by default, the `update_interval` is exactly `evaluation_len`
                the `start_step` is exactly 0
use dynamic bs:
    run inventory_train_base_stock.py
        --update_interval 1: how often to update the base-stock levels
        --start_step -7: where to start the base-stock levels pickup amid the horizon
        --buyin_gap ?
        --stop_order ?
use static bs:
    run with commented `if oracle` and uncommented `or`,
        --update_interval 1 
        --start_step -7
        --static [necessary]
'''
import random
import numpy as np
import time
import os

import ray
from ray.tune.logger import pretty_print
from ray.rllib.utils import try_import_tf, try_import_torch
import ray.rllib.agents.trainer_template as tt
from ray.rllib.models.tf.tf_action_dist import MultiCategorical
from ray.rllib.models import ModelCatalog
from functools import partial

import ray.rllib.agents.ppo.ppo as ppo
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
import ray.rllib.agents.qmix.qmix as qmix
from ray.rllib.agents.qmix.qmix_policy import QMixTorchPolicy
import ray.rllib.env.multi_agent_env
import ray.rllib.models as models
from ray.rllib.policy.tf_policy_template import build_tf_policy
from gym.spaces import Box, Tuple, MultiDiscrete, Discrete

import tensorflow as tf
import ray.rllib.agents.trainer_template as tt
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from render.inventory_renderer import AsciiWorldRenderer
from env.inventory_env import InventoryManageEnv
from env.inventory_utils import Utils
from scheduler.inventory_random_policy import ProducerBaselinePolicy, BaselinePolicy
from scheduler.inventory_base_stock_policy import ConsumerBaseStockPolicy
from scheduler.inventory_minmax_policy import ConsumerMinMaxPolicy as ConsumerBaselinePolicy
from utility.tools import SimulationTracker
from config.inventory_config import env_config
from agents.inventory import *
from utility.visualization import visualization

import ray
from ray import tune
from tqdm import tqdm as tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument("--torch", action="store_true")
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--episod", type=int, default=60)
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--use-prev-action-reward", action="store_true")
parser.add_argument("--stop-iters", type=int, default=20)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--stop-reward", type=float, default=150.0)
parser.add_argument("--pt", type=int, default=0)
parser.add_argument("--static", action="store_true", default=False)
parser.add_argument("--update_interval", type=int, default=env_config["evaluation_len"])
parser.add_argument("--start_step", type=int, default=0)
parser.add_argument("--buyin_gap", type=int, default=0)
parser.add_argument("--stop_order", type=float, default=1.)

def get_policies(is_static):
    env_config_for_rendering = env_config.copy()
    
    episod_duration = args.episod
    env_config_for_rendering['episod_duration'] = episod_duration
    env = InventoryManageEnv(env_config_for_rendering)
    obss = env.reset()

    ConsumerBaseStockPolicy.env_config = env_config_for_rendering
    ConsumerBaseStockPolicy.facilities = env.world.facilities
    ConsumerBaseStockPolicy.stop_order_factor = args.stop_order

    starter_step = env.env_config['episod_duration']+env.env_config['tail_timesteps']
    env.set_retailer_step(starter_step-episod_duration)
    env.set_iteration(1, 1)
    print(f"Environment: Producer action space {env.action_space_producer}, Consumer action space {env.action_space_consumer}, Observation space {env.observation_space}")

    if is_static: # base-stock levels are predefined before eval
        def load_base_policy(agent_id):
            if Utils.is_producer_agent(agent_id):
                return ProducerBaselinePolicy(env.observation_space, env.action_space_producer, BaselinePolicy.get_config_from_env(env))
            else:
                return ConsumerBaselinePolicy(env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env))

        base_policies = {}
        for agent_id in env.agent_ids():
            base_policies[agent_id] = load_base_policy(agent_id)

        _, infos = env.state_calculator.world_to_state(env.world)
        rnn_states = {}
        rewards = {}
        for agent_id in obss.keys():
            rnn_states[agent_id] = base_policies[agent_id].get_initial_state()
            rewards[agent_id] = 0

        # initializing for base stock policies
        for epoch in tqdm(range(args.episod)):
            action_dict = {}
            for agent_id, obs in obss.items():
                policy = base_policies[agent_id]
                action, _, _ = policy.compute_single_action(obs, state=rnn_states[agent_id], info=infos[agent_id], explore=True ) 
                action_dict[agent_id] = action
            obss, rewards, _, infos = env.step(action_dict)
        ConsumerBaseStockPolicy.update_base_stocks()

    def load_policy(agent_id):
        _facility = env.world.facilities[Utils.agentid_to_fid(agent_id)] 
        if Utils.is_producer_agent(agent_id):
            return ProducerBaselinePolicy(env.observation_space, env.action_space_producer, BaselinePolicy.get_config_from_env(env))
        # elif isinstance(_facility, SKUStoreUnit) or isinstance(_facility, SKUWarehouseUnit):
        elif isinstance(_facility, SKUStoreUnit):
            policy = ConsumerBaseStockPolicy(env.observation_space, env.action_space_consumer,
                        BaselinePolicy.get_config_from_env(env), is_static)
            return policy
        else:
            return ConsumerBaselinePolicy(env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env))

    policies = {}
    for agent_id in env.agent_ids():
        policies[agent_id] = load_policy(agent_id)
    return policies

if __name__ == "__main__":
    args = parser.parse_args()
    # is static? is oracle?
    is_static = args.static
    is_oracle = (args.start_step==0)
    if is_oracle:
        policy_mode = "base_stock_oracle"
    else:
        policy_mode = f"base_stock_"+("static" if is_static else "dynamic")+\
            f"_gap{args.buyin_gap}_updt{args.update_interval}_start{args.start_step}"
    
    # ConsumerBaseStockPolicy static fields

    def policy_oracle_setup():
        ConsumerBaseStockPolicy.time_hrz_len = env_config['evaluation_len']
        ConsumerBaseStockPolicy.oracle = True
        ConsumerBaseStockPolicy.has_order_cost = False

    ConsumerBaseStockPolicy.buyin_gap = args.buyin_gap
    
    # setup
    if is_oracle:
        policy_oracle_setup()
    else:
        ConsumerBaseStockPolicy.time_hrz_len = env_config['sale_hist_len']
    
    # always set these fields
    ConsumerBaseStockPolicy.update_interval = args.update_interval
    ConsumerBaseStockPolicy.start_step = args.start_step

    policies = get_policies(is_static)
    # ray.init()

    # Simulation loop
    vis_env = InventoryManageEnv(env_config.copy())

    visualization(vis_env, policies, 0, policy_mode, basestock=True)
