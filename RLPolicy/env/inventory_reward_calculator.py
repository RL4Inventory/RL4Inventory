from abc import ABC
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from collections import deque
import numpy as np
import random as rnd
import networkx as nx
import statistics

from enum import Enum, auto
from agents.base import Cell, Agent, BalanceSheet
from agents.inventory import *
from agents.inventory_order import OuterSKUStoreUnit
from env.inventory_utils import Utils


class RewardCalculator:
    
    def __init__(self, env_config):
        self.env_config = env_config
    
    def calculate_reward(self, world, step_outcome) -> dict:
        return self._retailer_profit(world, step_outcome)
    
    def _retailer_profit(self, env, step_outcome):
        # 终端（Retailer）营业额
        wc = self.env_config['global_reward_weight_consumer']
        parent_facility_balance = dict()
        # 计算SKU的Reward的时候，将其所属的Store的Reward也计算在内（以一定的权重wc）
        for facility in env.world.facilities.values():
            if isinstance(facility, ProductUnit):
                parent_facility_balance[facility.id] = step_outcome.facility_step_balance_sheets[facility.facility.id]
            else:
                parent_facility_balance[facility.id] = step_outcome.facility_step_balance_sheets[facility.id]
        
        consumer_reward_by_facility = { f_id: wc * parent_facility_balance[f_id] + (1 - wc) * reward for f_id, reward in step_outcome.facility_step_balance_sheets.items() }
        rewards_by_agent = {}

        for f_id, reward in step_outcome.facility_step_balance_sheets.items():
            rewards_by_agent[Utils.agentid_producer(f_id)] = reward / 1000000.0
        
        for f_id, reward in consumer_reward_by_facility.items():
            rewards_by_agent[Utils.agentid_consumer(f_id)] = reward / 1000000.0
        
        return rewards_by_agent