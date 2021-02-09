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
# from scheduler.inventory_minmax_policy import ConsumerMinMaxPolicy as ConsumerBaselinePolicy
from scheduler.inventory_eoq_policy import ConsumerEOQPolicy as ConsumerBaselinePolicy
from utility.tools import SimulationTracker
# from scheduler.inventory_tf_model import FacilityNet
# from scheduler.inventory_torch_model import SKUStoreGRU, SKUWarehouseNet, SKUStoreDNN
from scheduler.inventory_torch_model import SKUStoreBatchNormModel as SKUStoreDNN
from config.inventory_config import env_config
from explorer.stochastic_sampling import StochasticSampling
from utility.visualization import visualization
import torch

# Configuration ===============================================================================


env_config_for_rendering = env_config.copy()
episod_duration = env_config_for_rendering['episod_duration']
env = InventoryManageEnv(env_config_for_rendering)


ppo_policy_config_store_consumer = {
    "model": {
        "fcnet_hiddens": [32, 32],
        "custom_model": "sku_store_net",
        # == LSTM ==
        "use_lstm": False,
        "max_seq_len": 14,
        "lstm_cell_size": 8, 
        "lstm_use_prev_action_reward": False
    }
}


# Model Configuration ===============================================================================
models.ModelCatalog.register_custom_model("sku_store_net", SKUStoreDNN)
# models.ModelCatalog.register_custom_model("sku_store_net", SKUStoreGRU)

MyTFPolicy = PPOTorchPolicy

policies = {
        'baseline_producer': (ProducerBaselinePolicy, env.observation_space, env.action_space_producer, BaselinePolicy.get_config_from_env(env)),
        'baseline_consumer': (ConsumerBaselinePolicy, env.observation_space, env.action_space_consumer, BaselinePolicy.get_config_from_env(env)),
        'ppo_store_consumer': (MyTFPolicy, env.observation_space, env.action_space_consumer, ppo_policy_config_store_consumer),
    }

# Training Routines ===============================================================================

def print_training_results(result):
    keys = ['date', 'episode_len_mean', 'episodes_total', 'episode_reward_max', 'episode_reward_mean', 'episode_reward_min', 
            'timesteps_total', 'policy_reward_max', 'policy_reward_mean', 'policy_reward_min']
    for k in keys:
        print(f"- {k}: {result[k]}", flush=True)


def policy_map_fn(agent_id):
    if Utils.is_producer_agent(agent_id):
        return 'baseline_producer'
    else:
        if agent_id.startswith('SKUStoreUnit') or agent_id.startswith('OuterSKUStoreUnit'):
            return 'ppo_store_consumer'
        else:
            return 'baseline_consumer'

def get_policy(env, ppo_trainer):
    obss = env.reset()
    policies = {}

    for agent_id in obss.keys():
        policies[agent_id] = ppo_trainer.get_policy(policy_map_fn(agent_id))

    return policies

# def load_policy(plcs):
#     policies = {}
#     for key, val in plcs.items():
#         policies[key] = torch.load(val, map_location=
#                 torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#     return torch.load(policies

def train_ppo(args):
    env_config_for_rendering.update({'init':args.init})
    ext_conf = ppo.DEFAULT_CONFIG.copy()
    ext_conf.update({
            "env": InventoryManageEnv,
            "framework": "torch",
            "num_workers": 4,
            "vf_share_layers": True,
            "vf_loss_coeff": 1.00,   
            # estimated max value of vf, used to normalization   
            "vf_clip_param": 100.0,
            "clip_param": 0.2, 
            "use_critic": True,
            "use_gae": True,
            "lambda": 1.0,
            "gamma": 0.99,
            'env_config': env_config_for_rendering.copy(),
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
                "random_timesteps": 0, # args.rollout_fragment_length*args.batch_size*args.stop_iters // 2,
            },
            "multiagent": {
                "policies":policies,
                "policy_mapping_fn": policy_map_fn,
                "policies_to_train": ['ppo_store_consumer']
            }
        })

    print(f"Environment: action space producer {env.action_space_producer}, action space consumer {env.action_space_consumer}, observation space {env.observation_space}"
            , flush=True)

    if(args.is_pretrained):
        ext_conf.update(
            {
                'num_workers':0#, 'episod_duration':args.episod
            }
        )
        ppo_trainer = ppo.PPOTrainer(
                env = InventoryManageEnv,
                config = ext_conf)
        env.env_config.update({'episod_duration':args.episod})
        ppo_trainer.restore(args.premodel)
        visualization(InventoryManageEnv(env_config.copy()), get_policy(env, ppo_trainer), 1, args.run_name)
        return ppo_trainer
    
    # ppo_trainer.restore('/root/ray_results/PPO_InventoryManageEnv_2020-11-02_18-25-55cle_glgg/checkpoint_20/checkpoint-20')

    # stop = {
    #     "training_iteration": args.stop_iters,
    #     "timesteps_total": args.stop_timesteps,
    #     "episode_reward_min": args.stop_reward,
    # }

    # analysis = tune.run(args.run, config=ext_conf, stop=stop, mode='max', checkpoint_freq=1, verbose=1)
    # checkpoints = analysis.get_trial_checkpoints_paths(
    #                         trial=analysis.get_best_trial("episode_reward_max"),
    #                         metric="episode_reward_max")
    # ppo_trainer.restore(checkpoints[0][0])

    ext_conf['env_config'].update({
        'gamma':ext_conf['gamma'],
        'training':True,
        'policies':None
    })

    ppo_trainer = ppo.PPOTrainer(
            env = InventoryManageEnv,
            config = ext_conf)
    max_mean_reward = -100

    ppo_trainer.workers.foreach_worker(
        lambda ev: ev.foreach_env(
            lambda env: env.set_policies(get_policy(env, ev))))

    for i in range(args.stop_iters):
        print("== Iteration", i, "==", flush=True)
        
        ppo_trainer.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_iteration(i, args.stop_iters)))
        result = ppo_trainer.train()
        print_training_results(result)
        now_mean_reward = result['policy_reward_mean']['ppo_store_consumer']

        if (i+1) % args.visualization_frequency == 0 or now_mean_reward > max_mean_reward:
            max_mean_reward = max(max_mean_reward, now_mean_reward)
            checkpoint = ppo_trainer.save()
            print("checkpoint saved at", checkpoint, flush=True)
            visualization(InventoryManageEnv(env_config.copy()), get_policy(env, ppo_trainer),
                            i, args.run_name)
            # exit(0)

    return ppo_trainer


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
parser.add_argument("--min-batch-size", type=int, default=512)
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--use-prev-action-reward", action="store_true")
parser.add_argument("--stop-iters", type=int, default=1000)
parser.add_argument("--stop-reward", type=float, default=100000000.0)
parser.add_argument("--visualization-frequency", type=int, default=5)
parser.add_argument("--premodel", type=str, 
    default='/root/ray_results/PPO_InventoryManageEnv_2021-01-02_15-33-44wxfwo_0l/checkpoint_260/checkpoint-260')
parser.add_argument("--run-name", type=str, default='ppo_tree_online_test')
parser.add_argument("--init", type=str, default='rst') #'rst', 'rnd' or None

'''
Premodel:

<batch: 218, gamma: 0.9, without seller profit and unfulfilled cost>
/root/ray_results/PPO_InventoryManageEnv_2020-11-26_05-37-433lgvk2hw/checkpoint_153/checkpoint-153

<batch: 512, gamma: 0.999> 
/root/ray_results/PPO_InventoryManageEnv_2020-11-28_06-06-34opuw8gem/checkpoint_225/checkpoint-225

<batch: 256, gamma: 0.99>

<hrz:180, batch:512, gamma:0.999>
/root/ray_results/PPO_InventoryManageEnv_2020-12-02_08-52-40k_db4794/checkpoint_250/checkpoint-250

'''

if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()
    # Parameters of the tracing simulation
    # 'baseline' or 'trained'
    
    args.is_pretrained = (args.premodel and os.path.exists(args.premodel))
    # policy_mode = 'ppo', dummy param, use run-name instead
    trainer = train_ppo(args)

    # model = trainer.get_policy(policy_map_fn('SKUStoreUnit_8c')).model
    # torch.save(model.state_dict(), 'SKUStoreUnit_8c.model')

    # Create the environment
    '''
    renderer = AsciiWorldRenderer()
    frame_seq = []

    evaluation_epoch_len = 60
    env.set_iteration(1, 1)
    env.env_config.update({'episod_duration': evaluation_epoch_len, 'downsampling_rate': 1})
    print(f"Environment: Producer action space {env.action_space_producer}, Consumer action space {env.action_space_consumer}, Observation space {env.observation_space}")
    obss = env.reset()
    _, infos = env.state_calculator.world_to_state(env.world)


    def load_policy(agent_id):
        return trainer.get_policy(policy_map_fn(agent_id))

    policies = {}
    rnn_states = {}
    rewards = {}
    for agent_id in obss.keys():
        policies[agent_id] = load_policy(agent_id)
        rnn_states[agent_id] = policies[agent_id].get_initial_state()
        rewards[agent_id] = 0
        
    # Simulation loop
    tracker = SimulationTracker(evaluation_epoch_len, 1, env.agent_ids())
    for epoch in tqdm(range(evaluation_epoch_len)):
        print(f"{epoch}/{evaluation_epoch_len}")
        action_dict = {}
        for agent_id, obs in obss.items():
            policy = policies[agent_id]
            action, new_state, _ = policy.compute_single_action(obs, state=rnn_states[agent_id], info=infos[agent_id], explore=True ) 
            action_dict[agent_id] = action
            # if agent_id.startswith('SKUStoreUnit') and Utils.is_consumer_agent(agent_id):
            #     print(agent_id, action, rewards[agent_id])
            #     print(obs.tolist())
        obss, rewards, dones, infos = env.step(action_dict)
        step_balances = {}
        for agent_id in rewards.keys():
            step_balances[agent_id] = env.world.facilities[Utils.agentid_to_fid(agent_id)].economy.step_balance.total()

        tracker.add_sample(0, epoch, env.world.economy.global_balance().total(), step_balances)
        # some stats
        stock_status = env.get_stock_status()
        order_in_transit_status = env.get_order_in_transit_status()
        demand_status = env.get_demand_status()
        
        tracker.add_sku_status(0, epoch, stock_status, order_in_transit_status, demand_status)
        
        frame = renderer.render(env.world)
        frame_seq.append(np.asarray(frame))
    
    if not os.path.exists('output/%s' % policy_mode):
        os.mkdir('output/%s' % policy_mode)
    
    tracker.render("output/%s/plot.png" % policy_mode)
    tracker.render_sku(policy_mode)
    '''
    # if steps_to_render is not None:
    #     print('Rendering the animation...')
    #     AsciiWorldRenderer.plot_sequence_images(frame_seq, f"output/{policy_mode}/sim.mp4")
