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

from tqdm import tqdm as tqdm


def visualization(env, policies, iteration, policy_mode, basestock=False):

    policy_mode = policy_mode  # + f'_{iteration}'

    renderer = AsciiWorldRenderer()
    frame_seq = []

    evaluation_epoch_len = env.env_config['evaluation_len']
    starter_step = env.env_config['episod_duration']+env.env_config['tail_timesteps']
    env.set_iteration(1, 1)
    # env.env_config.update({'episod_duration': evaluation_epoch_len, 'downsampling_rate': 1})
    print(
        f"Environment: Producer action space {env.action_space_producer}, Consumer action space {env.action_space_consumer}, Observation space {env.observation_space}"
        , flush=True)
    obss = env.reset()
    if basestock:
        from scheduler.inventory_base_stock_policy import ConsumerBaseStockPolicy
        ConsumerBaseStockPolicy.facilities = env.world.facilities    

    if Utils.get_demand_sampler()=='ONLINE':
        env.set_retailer_step(starter_step)
    _, infos = env.state_calculator.world_to_state(env.world)


    # policies = {}
    rnn_states = {}
    rewards = {}
    for agent_id in obss.keys():
        # policies[agent_id] = load_policy(agent_id)
        rnn_states[agent_id] = policies[agent_id].get_initial_state()
        rewards[agent_id] = 0

    # Simulation loop
    tracker = SimulationTracker(evaluation_epoch_len, 1, env.agent_ids())
    print(f"  === evaluation length {evaluation_epoch_len}, it will take about 1 min ....", flush=True)

    for epoch in range(evaluation_epoch_len):
        action_dict = {}
        for agent_id, obs in obss.items():
            policy = policies[agent_id]
            action, new_state, _ = policy.compute_single_action(obs, state=rnn_states[agent_id], info=infos[agent_id],
                                                                explore=False)
            action_dict[agent_id] = action
            # if agent_id.startswith('SKUStoreUnit') and Utils.is_consumer_agent(agent_id):
            #     print(agent_id, action, rewards[agent_id])
            #     print(obs.tolist())
        obss, rewards, dones, infos = env.step(action_dict)
        step_balances = {}
        for agent_id in rewards.keys():
            step_balances[agent_id] = env.world.facilities[Utils.agentid_to_fid(agent_id)].economy.step_balance.total()
        # print(env.world.economy.global_balance().total(), step_balances, rewards)
        tracker.add_sample(0, epoch, env.world.economy.global_balance().total(), step_balances, rewards)
        # some stats
        stock_status = env.get_stock_status()
        order_in_transit_status = env.get_order_in_transit_status()
        demand_status = env.get_demand_status()

        tracker.add_sku_status(0, epoch, stock_status, order_in_transit_status, demand_status)

        frame = renderer.render(env.world)
        frame_seq.append(np.asarray(frame))

    print(tracker.get_retailer_profit())

    if not os.path.exists('output'):
        os.mkdir('output')

    if not os.path.exists('output/%s' % policy_mode):
        os.mkdir('output/%s' % policy_mode)

    if not os.path.exists(f'output/{policy_mode}/iter_{iteration}'):
        os.mkdir(f'output/{policy_mode}/iter_{iteration}')

    # tracker.render("output/%s/plot.png" % policy_mode)
    tracker.render(f'output/{policy_mode}/iter_{iteration}/plot.png')
    tracker.render_sku(policy_mode, iteration)
    print(f"  === evaluation length end ", flush=True)
