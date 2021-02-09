from abc import ABC
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from collections import deque
import numpy as np
import random as rnd
import networkx as nx
import time
from enum import Enum, auto
from ray.rllib.policy import Policy
from agents.base import Cell, Agent, BalanceSheet
from agents.inventory import *
from env.inventory_env import *
from scheduler.inventory_random_policy import BaselinePolicy


# Q = F^{-1}(1-c/p)
# Q optimal order quantity
# c cost of each unit
# p sale price of each unit
# F^{-1} generalized inverse cumulative distribution function of demand
class ConsumerNewsVendorPolicy(BaselinePolicy):
    
    def __init__(self, observation_space, action_space, config):
        BaselinePolicy.__init__(self, observation_space, action_space, config)
    
    def _find_source(self, f_state_info):
        # stop placing orders when the facility ran out of money 
        # if f_state_info['is_positive_balance'] <= 0:
        #     return (0, 0, 0)

        facility_type = None
        facility_type = type(f_state_info['facility'])
        
        if facility_type not in [SKUWarehouseUnit, SKUStoreUnit, OuterSKUStoreUnit]:
            return 0
        
        # consumer_source_inventory
        inputs = f_state_info['bom_inputs']
        available_inventory = np.array(f_state_info['storage_levels'])
        inflight_orders = np.array(f_state_info['consumer_in_transit_orders'])
        booked_inventory = available_inventory + inflight_orders
        
        # stop placing orders when the facilty runs out of capacity
        if np.sum(booked_inventory) > f_state_info['storage_capacity']:
            return 0
        
        most_neeed_product_id = None
        min_ratio = float('inf')
        for product_id, quantity in enumerate(inputs):
            if quantity > 0:
                fulfillment_ratio = booked_inventory[product_id] / quantity
                if fulfillment_ratio < min_ratio:
                    min_ratio = fulfillment_ratio
                    most_neeed_product_id = product_id
        
        exporting_sources = []
        if most_neeed_product_id is not None:
            for i in range(self.n_sources):
                for j in range(self.n_products):
                    if f_state_info['consumer_source_export_mask'][i*self.n_products + j] == 1:
                        exporting_sources.append(i)
        consumer_action_space_size = len(Utils.get_consumer_action_space())
        consumer_quantity = rnd.randint(0, consumer_action_space_size-1)
        return consumer_quantity