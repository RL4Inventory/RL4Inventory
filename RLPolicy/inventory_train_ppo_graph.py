import random
import numpy as np
import time
import os

import ray
from ray.tune.logger import pretty_print
from ray.rllib.utils import try_import_torch
import ray.rllib.agents.trainer_template as tt
# from ray.rllib.models.tf.tf_action_dist import MultiCategorical
from ray.rllib.models.torch.torch_action_dist import TorchMultiCategorical as MultiCategorical
from ray.rllib.models import ModelCatalog
from functools import partial

import ray.rllib.agents.ppo.ppo as ppo
# from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
import ray.rllib.agents.qmix.qmix as qmix
from ray.rllib.agents.qmix.qmix_policy import QMixTorchPolicy
import ray.rllib.env.multi_agent_env
import ray.rllib.models as models
from gym.spaces import Box, Tuple, MultiDiscrete, Discrete

import ray.rllib.agents.trainer_template as tt
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from render.inventory_renderer import AsciiWorldRenderer
from env.inventory_env import InventoryManageEnv
from env.inventory_utils import Utils
from scheduler.inventory_random_policy import ProducerBaselinePolicy, BaselinePolicy
# from scheduler.inventory_random_policy import ConsumerBaselinePolicy
from scheduler.inventory_minmax_policy import ConsumerMinMaxPolicy as ConsumerBaselinePolicy
# from scheduler.inventory_eoq_policy import ConsumerEOQPolicy as ConsumerBaselinePolicy
from utility.tools import SimulationTracker
# from scheduler.inventory_tf_model import FacilityNet
from scheduler.inventory_torch_model import SKUStoreBatchNormModel as SKUStoreDNN
from scheduler.inventory_torch_model import SKUWarehouseBatchNormModel as SKUWarehouseDNN
from config.inventory_config import env_config
from explorer.stochastic_sampling import StochasticSampling


# Configuration ===============================================================================


env_config_for_rendering = env_config.copy()
episod_duration = env_config_for_rendering['episod_duration']
env = InventoryManageEnv(env_config_for_rendering)


ppo_policy_config_producer = {
    "model": {
        "fcnet_hiddens": [128, 128],
        "custom_model": "facility_net"
    }
}

ppo_policy_config_store_consumer = {
    "model": {
        "fcnet_hiddens": [16, 16],
        "custom_model": "sku_store_net",
        # == LSTM ==
        "use_lstm": False,
        "max_seq_len": 14,
        "lstm_cell_size": 128, 
        "lstm_use_prev_action_reward": False
    }
}


ppo_policy_config_warehouse_consumer = {
    "model": {
        "fcnet_hiddens": [16, 16],
        "custom_model": "sku_warehouse_net",
        # == LSTM ==
        "use_lstm": False,
        "max_seq_len": 14,
        "lstm_cell_size": 128, 
        "lstm_use_prev_action_reward": False
    }
}

# Model Configuration ===============================================================================
models.ModelCatalog.register_custom_model("sku_store_net", SKUStoreDNN)
models.ModelCatalog.register_custom_model("sku_warehouse_net", SKUWarehouseDNN)

MyTorchPolicy = PPOTorchPolicy

policies = {
        'baseline_producer': (ProducerBaselinePolicy, env.observation_space, env.action_space_producer, BaselinePolicy.get_config_from_env(env)),
        'baseline_consumer': (ConsumerBaselinePolicy, env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env)),
        'ppo_producer': (MyTorchPolicy, env.observation_space, env.action_space_producer, ppo_policy_config_producer),
        'ppo_store_consumer': (MyTorchPolicy, env.observation_space, env.action_space_consumer, ppo_policy_config_store_consumer),
        'ppo_warehouse_consumer': (MyTorchPolicy, env.observation_space, env.action_space_consumer, ppo_policy_config_warehouse_consumer)
    }

def filter_keys(d, keys):
    return {k:v for k,v in d.items() if k in keys}

# Training Routines ===============================================================================

def print_training_results(result):
    keys = ['date', 'episode_len_mean', 'episodes_total', 'episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 
            'timesteps_total', 'policy_reward_max', 'policy_reward_mean', 'policy_reward_min']
    for k in keys:
        print(f"- {k}: {result[k]}")


def echelon_policy_map_fn(echelon, agent_id):
    facility_id = Utils.agentid_to_fid(agent_id)
    if Utils.is_producer_agent(agent_id):
        return 'baseline_producer'
    else:
        agent_echelon = env.world.agent_echelon[facility_id]
        if  agent_echelon == 0: # supplier
            return 'baseline_consumer'
        elif agent_echelon == env.world.total_echelon - 1: # retailer
            return 'ppo_store_consumer'
        elif agent_echelon >= echelon: # warehouse and current layer is trainning or has been trained.
            return 'ppo_warehouse_consumer'
        else: # warehouse on layers that haven't been trained yet
            return 'baseline_consumer'


def create_ppo_trainer(echelon):
    policy_map_fn = (lambda x: echelon_policy_map_fn(echelon, x))
    policies_to_train = (['ppo_store_consumer'] if echelon == env.world.total_echelon -1 else ['ppo_warehouse_consumer'])
    ext_conf = ppo.DEFAULT_CONFIG.copy()
    ext_conf.update({
            "env": InventoryManageEnv,
            "framework": "torch",
            "num_workers": 2,
            "vf_share_layers": True,
            "vf_loss_coeff": 1.00,   
            # estimated max value of vf, used to normalization   
            "vf_clip_param": 10.0,
            "clip_param": 0.1, 
            "use_critic": True,
            "use_gae": True,
            "lambda": 1.0,
            "gamma": 0.9,
            'env_config': env_config_for_rendering,
            # Number of steps after which the episode is forced to terminate. Defaults
            # to `env.spec.max_episode_steps` (if present) for Gym envs.
            "horizon": args.episod,
            # Calculate rewards but don't reset the environment when the horizon is
            # hit. This allows value estimation and RNN state to span across logical
            # episodes denoted by horizon. This only has an effect if horizon != inf.
            "soft_horizon": False,
            # Minimum env steps to optimize for per train call. This value does
            # not affect learning, only the length of train iterations.
            'timesteps_per_iteration': 1000,
            'batch_mode': 'complete_episodes',
            # Size of batches collected from each worker
            "rollout_fragment_length": args.rollout_fragment_length,
            # Number of timesteps collected for each SGD round. This defines the size
            # of each SGD epoch.
            "train_batch_size": args.rollout_fragment_length*args.batch_size,
            # Whether to shuffle sequences in the batch when training (recommended).
            "shuffle_sequences": True,
            # Total SGD batch size across all devices for SGD. This defines the
            # minibatch size within each epoch.
            "sgd_minibatch_size": args.rollout_fragment_length*args.min_batch_size,
            # Number of SGD iterations in each outer loop (i.e., number of epochs to
            # execute per train batch).
            "num_sgd_iter": 50,
            "lr": 1e-4,
            "_fake_gpus": True,
            "num_gpus": 0,
            "explore": True,
            "exploration_config": {
                "type": StochasticSampling,
                "random_timesteps": 10000, # args.rollout_fragment_length*args.batch_size*args.stop_iters // 2,
            },
            "multiagent": {
                "policies": filter_keys(policies, ['baseline_producer', 'baseline_consumer', 'ppo_store_consumer', 'ppo_warehouse_consumer']),
                "policy_mapping_fn": policy_map_fn,
                "policies_to_train": policies_to_train
            }
        })

    print(f"Environment: action space producer {env.action_space_producer}, action space consumer {env.action_space_consumer}, observation space {env.observation_space}")
    ppo_trainer = ppo.PPOTrainer(
        env = InventoryManageEnv,
        config = ext_conf)
    return ppo_trainer


def train_ppo(n_iterations, ppo_trainer):
    # ppo_trainer.restore('/root/ray_results/PPO_InventoryManageEnv_2020-11-02_13-23-229zjyl98i/checkpoint_10/checkpoint-10')
    for i in range(n_iterations):
        print("== Iteration", i, "==")
        print_training_results(ppo_trainer.train())
        if (i+1) % 10 == 0:
            checkpoint = ppo_trainer.save()
            print("checkpoint saved at", checkpoint)
    return ppo_trainer


import ray
from ray import tune
from tqdm import tqdm as tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument("--torch", action="store_true")
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--episod", type=int, default=episod_duration)
parser.add_argument("--rollout-fragment-length", type=int, default=14)
parser.add_argument("--batch-size", type=int, default=2560)
parser.add_argument("--min-batch-size", type=int, default=128)
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--use-prev-action-reward", action="store_true")
parser.add_argument("--stop-iters", type=int, default=20)
parser.add_argument("--stop-reward", type=float, default=100000000.0)
parser.add_argument("--echelon-to-train", type=int, default=2)
parser.add_argument("--pt", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()
    # Parameters of the tracing simulation
    # 'baseline' or 'trained'
    
    policy_mode = 'ppo_graph'
    episod_duration = args.episod
    total_echelon = env.world.total_echelon
    # number of echelon that to be trained, count from retailer
    # for instance, if echelon_to_be_trained == 2, it means that policies for products in stores and
    # warehouses closest to stores will be trained. 
    echelon_to_train = args.echelon_to_train
    
    retailer_ppo_trainer = create_ppo_trainer(total_echelon-1)
    retailer_ppo_trainer = train_ppo(args.stop_iters, retailer_ppo_trainer)

    pre_warehouse_trainer_weight = None
    for echelon in range(total_echelon-2, total_echelon-echelon_to_train-1, -1):
        ppo_trainer = create_ppo_trainer(echelon)
        ppo_trainer.set_weights(retailer_ppo_trainer.get_weights('ppo_store_consumer'))
        if pre_warehouse_trainer_weight is not None:
            ppo_trainer.set_weights(pre_warehouse_trainer_weight)
        ppo_trainer = train_ppo(args.stop_iters, ppo_trainer)
        pre_warehouse_trainer_weight = ppo_trainer.get_weights('ppo_warehouse_consumer')
        
    # Create the environment
    evaluation_epoch_len = 60
    env.set_iteration(1, 1)
    env.env_config.update({'episod_duration': evaluation_epoch_len, 'downsampling_rate': 1})
    print(f"Environment: Producer action space {env.action_space_producer}, Consumer action space {env.action_space_consumer}, Observation space {env.observation_space}")

    def load_policy(agent_id):
        agent_echelon = env.world.agent_echelon[Utils.agentid_to_fid(agent_id)]
        if Utils.is_producer_agent(agent_id):
            policy_name = 'baseline_producer'
        else:
            if agent_echelon == total_echelon - 1:
                policy_name = 'ppo_store_consumer'
            else:
                if agent_echelon >= total_echelon-echelon_to_train:
                    policy_name = 'ppo_warehouse_consumer'
                else:
                    policy_name = 'baseline_consumer'
        return ppo_trainer.get_policy(policy_name)

    policies = {}
    for agent_id in env.agent_ids():
        policies[agent_id] = load_policy(agent_id)
        
    # Simulation loop
    tracker = SimulationTracker(evaluation_epoch_len, 1, env, policies)
    
    if args.pt:
        loc_path = f"{os.environ['PT_OUTPUT_DIR']}/{policy_mode}/"
    else:
        loc_path = 'output/%s/' % policy_mode
    tracker.run_and_render(loc_path)