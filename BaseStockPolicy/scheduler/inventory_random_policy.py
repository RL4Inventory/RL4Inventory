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
from agents.inventory_order import *
from env.inventory_env import *


class BaselinePolicy(Policy):   
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.action_space_shape = action_space.shape
        self.n_products = config['number_of_products']
        self.n_sources = config['number_of_sources']


    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs): 
        
        if info_batch is None:
            action_dict = [self._action(f_state, None) for f_state in obs_batch ], [], {}  
        else:    
            action_dict = [self._action(f_state, f_state_info) for f_state, f_state_info in zip(obs_batch, info_batch)], [], {}
            
        return action_dict
    
    def learn_on_batch(self, samples):
        """No learning."""
        return {}

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
    
    def get_config_from_env(env):
        return {'facility_types': env.facility_types, 
                'number_of_products': env.n_products(),
                'number_of_sources': env.max_sources_per_facility}
    
    
class ProducerBaselinePolicy(BaselinePolicy):
    
    def __init__(self, observation_space, action_space, config):
        BaselinePolicy.__init__(self, observation_space, action_space, config)
        facility_types = config['facility_types']
        self.unit_prices = [0] * len(facility_types)
        unit_price_map = {
            SupplierCell.__name__: 0, # $500
            WarehouseCell.__name__: 0,  # $500
            RetailerCell.__name__: 0,  # $500
            SKUSupplierUnit.__name__: 0,  # $500
            SKUWarehouseUnit.__name__: 3, # $800
            SKUStoreUnit.__name__: 5,    # $1000
            OuterSKUStoreUnit.__name__: 5    # $1000
        }
        for f_class, f_id in facility_types.items():
            self.unit_prices[ f_id ] = unit_price_map[f_class]
            
    def _action(self, facility_state, facility_state_info):
        def default_facility_control(unit_price): 
            control = [
                unit_price,    # unit_price
                2,             # production_rate (level 2 -> 4 units)
            ]
            return control
                   
        action = default_facility_control(0)
        if facility_state_info is not None and len(facility_state_info) > 0:   
            unit_price = self.unit_prices[ np.flatnonzero( facility_state_info['facility_type'] )[0] ]
            action = default_facility_control(unit_price)
        
        return action
    
    
class ConsumerBaselinePolicy(BaselinePolicy):
    
    def __init__(self, observation_space, action_space, config):
        BaselinePolicy.__init__(self, observation_space, action_space, config)
            
    def _action(self, facility_state, facility_state_info):
        def default_facility_control(order_qty):  # (level 4 -> 8 units)
            control = order_qty
            return control
        action = default_facility_control(0)
        # print(facility_state, facility_state_info)
        if facility_state_info is not None and len(facility_state_info) > 0:
            if np.count_nonzero(facility_state_info['bom_inputs']) > 0:
                action = default_facility_control(self._find_source(facility_state_info))
            else:
                action = default_facility_control(0)
        return action
    
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

        if booked_inventory[most_neeed_product_id] / f_state_info['storage_capacity'] >= 1.0 / len(f_state_info['storage_levels']):
            return 0
        
        exporting_sources = []
        if most_neeed_product_id is not None:
            for i in range(self.n_sources):
                for j in range(self.n_products):
                    if f_state_info['consumer_source_export_mask'][i*self.n_products + j] == 1:
                        exporting_sources.append(i)
        consumer_action_space_size = len(Utils.get_consumer_action_space())
        consumer_quantity = rnd.randint(0, consumer_action_space_size-1)
        return consumer_quantity

