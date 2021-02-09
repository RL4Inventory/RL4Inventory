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
from scheduler.inventory_eoq_policy import ConsumerEOQPolicy
from scheduler.inventory_minmax_policy import ConsumerMinMaxPolicy as ConsumerBaselinePolicy
from utility.tools import SimulationTracker
from config.inventory_config import env_config

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


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    env_config_for_rendering = env_config.copy()
    episod_duration = args.episod
    env_config_for_rendering['episod_duration'] = episod_duration
    env = InventoryManageEnv(env_config_for_rendering)

    policy_mode = "eoq"
    # Create the environment
    env.set_iteration(1, 1)
    print(f"Environment: Producer action space {env.action_space_producer}, Consumer action space {env.action_space_consumer}, Observation space {env.observation_space}")

    def load_policy(agent_id):
        if Utils.is_producer_agent(agent_id):
            return ProducerBaselinePolicy(env.observation_space, env.action_space_producer, BaselinePolicy.get_config_from_env(env))
        if agent_id.startswith('SKUStoreUnit') or agent_id.startswith('OuterSKUStoreUnit'):
            return ConsumerEOQPolicy(env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env))
        else:
            return ConsumerBaselinePolicy(env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env))

    policies = {}
    for agent_id in env.agent_ids():
        policies[agent_id] = load_policy(agent_id)

    # Simulation loop
    tracker = SimulationTracker(episod_duration, 1, env, policies)
    if args.pt:
        loc_path = f"{os.environ['PT_OUTPUT_DIR']}/{policy_mode}/"
    else:
        loc_path = 'output/%s/' % policy_mode
    tracker.run_and_render(loc_path)
