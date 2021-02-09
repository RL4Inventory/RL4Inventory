import random
import numpy as np
import time
import os

import ray
from ray import tune
from ray.tune.logger import pretty_print
from ray.rllib.utils import try_import_torch
import ray.rllib.agents.trainer_template as tt
# from ray.rllib.models.tf.tf_action_dist import MultiCategorical
from ray.rllib.models.torch.torch_action_dist import TorchMultiCategorical as MultiCategorical
from ray.rllib.models import ModelCatalog
from functools import partial

import ray.rllib.agents.ppo.ppo as ppo
import ray.rllib.agents.dqn.dqn as dqn
# from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.agents.dqn.dqn_torch_policy import DQNTorchPolicy

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
# from scheduler.inventory_minmax_policy import ConsumerMinMaxPolicy as ConsumerBaselinePolicy
from scheduler.inventory_eoq_policy import ConsumerEOQPolicy as ConsumerBaselinePolicy
from utility.tools import SimulationTracker
# from scheduler.inventory_tf_model import FacilityNet
# from scheduler.inventory_torch_model import SKUStoreGRU, SKUWarehouseNet, SKUStoreDNN
from scheduler.inventory_torch_model import SKUStoreBatchNormModel as SKUStoreDNN
from config.inventory_config import env_config
from explorer.stochastic_sampling import StochasticSampling
from utility.visualization import visluaization

# Configuration ===============================================================================


env_config_for_rendering = env_config.copy()
episod_duration = env_config_for_rendering['episod_duration']
env = InventoryManageEnv(env_config_for_rendering)


dqn_policy_config_store_consumer = {
    "model": {
        # === Built-in options ===
        # Number of hidden layers for fully connected net
        "fcnet_hiddens": [128, 128],
        # Nonlinearity for fully connected net (tanh, relu)
        "fcnet_activation": "relu",
        # Filter config. List of [out_channels, kernel, stride] for each filter
        "conv_filters": None,
        # Nonlinearity for built-in convnet
        "conv_activation": "relu",
        # For DiagGaussian action distributions, make the second half of the model
        # outputs floating bias variables instead of state-dependent. This only
        # has an effect is using the default fully connected net.
        "free_log_std": False,
        # Whether to skip the final linear layer used to resize the hidden layer
        # outputs to size `num_outputs`. If True, then the last hidden layer
        # should already match num_outputs.
        "no_final_linear": False,
        # Whether layers should be shared for the value function.
        "vf_share_layers": True,

        # == LSTM ==
        # Whether to wrap the model with an LSTM.
        "use_lstm": False,
        # Max seq len for training the LSTM, defaults to 20.
        "max_seq_len": 20,
        # Size of the LSTM cell.
        "lstm_cell_size": 256,
        # Whether to feed a_{t-1}, r_{t-1} to LSTM.
        "lstm_use_prev_action_reward": False,
        # Experimental (only works with `_use_trajectory_view_api`=True):
        # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
        "_time_major": False,


        # === Options for custom models ===
        # Name of a custom model to use
        "custom_model": None,
        # Extra options to pass to the custom classes. These will be available to
        # the Model's constructor in the model_config field. Also, they will be
        # attempted to be passed as **kwargs to ModelV2 models. For an example,
        # see rllib/models/[tf|torch]/attention_net.py.
        "custom_model_config": {},
        # Name of a custom action distribution to use.
        "custom_action_dist": None,
        # Custom preprocessors are deprecated. Please use a wrapper class around
        # your environment instead to preprocess observations.
        "custom_preprocessor": None,
    },
}


# Model Configuration ===============================================================================
models.ModelCatalog.register_custom_model("sku_store_net", SKUStoreDNN)
# models.ModelCatalog.register_custom_model("sku_store_net", SKUStoreGRU)


policies = {
        'baseline_producer': (ProducerBaselinePolicy, env.observation_space, env.action_space_producer, BaselinePolicy.get_config_from_env(env)),
        'baseline_consumer': (ConsumerBaselinePolicy, env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env)),
        'dqn_store_consumer': (DQNTorchPolicy, env.observation_space, env.action_space_consumer, dqn_policy_config_store_consumer),
    }

# Training Routines ===============================================================================

def print_training_results(result):
    keys = ['date', 'episode_len_mean', 'episodes_total', 'episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 
            'timesteps_total', 'policy_reward_max', 'policy_reward_mean', 'policy_reward_min']
    for k in keys:
        print(f"- {k}: {result[k]}")


def policy_map_fn(agent_id):
    if Utils.is_producer_agent(agent_id):
        return 'baseline_producer'
    else:
        if agent_id.startswith('SKUStoreUnit') or agent_id.startswith('OuterSKUStoreUnit'):
            return 'dqn_store_consumer'
        else:
            return 'baseline_consumer'

def get_policy(env, ppo_trainer):
    obss = env.reset()
    policies = {}

    for agent_id in obss.keys():
        policies[agent_id] = ppo_trainer.get_policy(policy_map_fn(agent_id))

    return policies

def train_dqn(args):
    ext_conf = dqn.DEFAULT_CONFIG.copy()
    ext_conf.update({
        "env": InventoryManageEnv,
        'env_config': env_config_for_rendering,

        "num_workers": 4,
        "train_batch_size": 1024,
        "learning_starts": 10000,
        "gamma": 0.9,
        "timesteps_per_iteration": 60,
        "target_network_update_freq": 500,

        # Whether to use dueling dqn
        "dueling": False,
        # Dense-layer setup for each the advantage branch and the value branch
        # in a dueling architecture.
        "hiddens": [256],
        # Whether to use double dqn
        "double_q": True,
        # N-step Q learning
        "n_step": 1,

        # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": "EpsilonGreedy",
            # Config for the Exploration class' constructor:
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 10000,  # Timesteps over which to anneal epsilon.

            # For soft_q, use:
            # "exploration_config" = {
            #   "type": "SoftQ"
            #   "temperature": [float, e.g. 1.0]
            # }
        },
        # Switch to greedy actions in evaluation workers.
        "evaluation_config": {
            "explore": False,
        },


        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": 50000,
        # If True prioritized replay buffer will be used.
        "prioritized_replay": False,
        # Alpha parameter for prioritized replay buffer.
        "prioritized_replay_alpha": 0.6,
        # Beta parameter for sampling from prioritized replay buffer.
        "prioritized_replay_beta": 0.4,
        # Final value of beta (by default, we use constant beta=0.4).
        "final_prioritized_replay_beta": 0.4,
        # Time steps over which the beta parameter is annealed.
        "prioritized_replay_beta_annealing_timesteps": 20000,
        # Epsilon to add to the TD errors when updating priorities.
        "prioritized_replay_eps": 1e-6,
        # Whether to LZ4 compress observations
        "compress_observations": False,
        # Callback to run before learning on a multi-agent batch of experiences.
        "before_learn_on_batch": None,
        # If set, this will fix the ratio of replayed from a buffer and learned on
        # timesteps to sampled from an environment and stored in the replay buffer
        # timesteps. Otherwise, the replay will proceed at the native ratio
        # determined by (train_batch_size / rollout_fragment_length).
        "training_intensity": None,

        # === Optimization ===
        # Learning rate for adam optimizer
        "lr": 3e-4,
        # Learning rate schedule
        "lr_schedule": None,
        # Adam epsilon hyper parameter
        "adam_epsilon": 1e-8,
        # If not None, clip gradients during optimization at this value
        "grad_clip": None, # 40

        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_map_fn,
            "policies_to_train": ['dqn_store_consumer']
        }
    })

    print(
        f"Environment: action space producer {env.action_space_producer}, action space consumer {env.action_space_consumer}, observation space {env.observation_space}")

    dqn_trainer = dqn.DQNTrainer(
        env=InventoryManageEnv,
        config=ext_conf)

    max_mean_reward = -100

    for i in range(args.stop_iters):
        print("== Iteration", i, "==")
        dqn_trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_iteration(i, args.stop_iters)))
        result = dqn_trainer.train()
        print_training_results(result)
        now_mean_reward = result['policy_reward_mean']['dqn_store_consumer']

        if (i + 1) % args.visualization_frequency == 0 or now_mean_reward > max_mean_reward:

            max_mean_reward = max(max_mean_reward, now_mean_reward)
            checkpoint = dqn_trainer.save()
            print("checkpoint saved at", checkpoint)
            visluaization(env, get_policy(env, dqn_trainer), i, args.run_name)
            # exit(0)



import ray
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
parser.add_argument("--stop-iters", type=int, default=1000)
parser.add_argument("--stop-reward", type=float, default=100000000.0)
parser.add_argument("--visualization-frequency", type=int, default=100)
parser.add_argument("--run-name", type=str, default='dqn_rllib_test1_rbaseline_debug')



if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()
    # Parameters of the tracing simulation
    # 'baseline' or 'trained'
    
    policy_mode = 'ppo'
    trainer = train_dqn(args)
