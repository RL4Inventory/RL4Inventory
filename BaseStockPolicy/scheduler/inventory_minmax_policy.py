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
from scheduler.inventory_random_policy import ConsumerBaselinePolicy
import scipy.stats as st


# parameters: (r, R), calculate according to VLT, demand variances, and service level
# replenish R - S units whenever the current stock is less than r
# S denotes the number of units in stock
class ConsumerMinMaxPolicy(ConsumerBaselinePolicy):
    
    def __init__(self, observation_space, action_space, config):
        ConsumerBaselinePolicy.__init__(self, observation_space, action_space, config)
       
    def _get_consumer_quantity(self, f_state_info):
        facility_info = f_state_info['facility_info']
        # sku_info = f_state_info['sku_info']
        assert 'order_cost' in facility_info, 'order_cost is needed for EOQ model'
        assert 'unit_storage_cost' in facility_info, 'unit_storage_cost is needed for EOQ model'
        # assert 'sale_gamma' in sku_info, 'sale_gamma is needed for EOQ model'
        order_cost = facility_info['order_cost']
        holding_cost = facility_info['unit_storage_cost']
        sale_gamma = f_state_info['sale_mean']
        consumer_quantity = int(np.sqrt(2*sale_gamma*order_cost / holding_cost) / sale_gamma)
        return Utils.get_consumer_quantity_action(consumer_quantity)
                
    
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
        
        most_needed_product_id = None
        min_ratio = float('inf')
        for product_id, quantity in enumerate(inputs):
            if quantity > 0:
                fulfillment_ratio = booked_inventory[product_id] / quantity
                if fulfillment_ratio < min_ratio:
                    min_ratio = fulfillment_ratio
                    most_needed_product_id = product_id

        exporting_sources = []
        if most_needed_product_id is not None:
            for i in range(self.n_sources):
                for j in range(self.n_products):
                    if f_state_info['consumer_source_export_mask'][i*self.n_products + j] == 1:
                        exporting_sources.append(i)
        # consumer_quantity = rnd.randint(0, consumer_action_space_size-1)
        source_id = rnd.choice(exporting_sources)
        # stop placing orders if no risk of out of stock
        vlt_buffer_days = 10 + 7 * ( Utils.get_env_config()['total_echelons'] - f_state_info['echelon_level'] )
        vlt = f_state_info['vlt'][source_id*self.n_products + most_needed_product_id] + vlt_buffer_days
        sale_mean, sale_std = f_state_info['sale_mean'], f_state_info['sale_std']
        service_level = f_state_info['service_level']
        r = (vlt*sale_mean + np.sqrt(vlt)*sale_std*st.norm.ppf(service_level))
        # print(vlt, r, f_state_info)
        # safty stock
        if booked_inventory[most_needed_product_id] > r:
            return 0

        R = 3*r
        consumer_quantity = Utils.get_consumer_quantity_action(int((R - r) / sale_mean))
        return consumer_quantity