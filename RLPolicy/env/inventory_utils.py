import numpy as np
import random as rnd
from enum import Enum, auto
import math
from config.inventory_config import sku_config, supplier_config, warehouse_config, store_config, demand_sampler, env_config


class Utils:

    @staticmethod
    def get_env_config():
        return env_config

    @staticmethod 
    def get_demand_sampler():
        return demand_sampler

    @staticmethod
    def get_reward_discount(inventory_level, is_retailer, echelon):
        """
        Given facility_type and the current inventory_level, return a reward discount. 
        The higher of the current inventory_level, the smaller of the reward discount.
        In this way, we discourage further replenishment if the current inventory is enough
        Args:
            inventory_level: expected days to sold out
            is_retailer: is the current facility a retailer
            echelon_level: the echelone level of the facility
        Return: a reward discount
        """
        inventory_level_bounds = [12, 7, 0]
        reward_discounts = [0.4, 0.9, 1.0]
        total_echelon = Utils.get_num_warehouse_echelon()
        if not is_retailer:
            inventory_level_bounds = [b+2*(total_echelon-echelon) for b in inventory_level_bounds]
        reward_discount = 1.05
        for i, b in enumerate(inventory_level_bounds):
            if inventory_level >= b:
                reward_discount = reward_discounts[i]
                break
        return math.pow(reward_discount, abs(inventory_level))
        
    @staticmethod
    def agentid_producer(facility_id):
        return facility_id + 'p'
    
    @staticmethod
    def agentid_consumer(facility_id):
        return facility_id + 'c'
    
    @staticmethod
    def is_producer_agent(agent_id):
        return agent_id[-1] == 'p'
    
    @staticmethod
    def is_consumer_agent(agent_id):
        return agent_id[-1] == 'c'
    
    @staticmethod
    def agentid_to_fid(agent_id):
        return agent_id[:-1]

    @staticmethod
    def get_consumer_action_space():
        # procurement space
        consumer_quantity_space = [0,1,2,3,4,5,7,9,12]
        consumer_quantity_space = [0, 0.33, 0.66, 1, 1.33, 1.66, 2, 2.5, 3, 4, 5, 6, 7, 9, 12]
        # consumer_quantity_space = [0, 5, 10, 15, 20, 40, 60, 80, 100, 150, 200, 250, 400]
        return consumer_quantity_space

    @staticmethod
    def get_consumer_quantity_action(consumer_quantity):
        consumer_action_space = Utils.get_consumer_action_space()
        for i, q in enumerate(consumer_action_space):
            if q > consumer_quantity:
                return i
        return len(consumer_action_space) - 1

    @staticmethod
    def get_sku_num():
        return sku_config['sku_num']

    @staticmethod
    def get_sku_name(sku_idx):
        return sku_config['sku_names'][sku_idx]

    @staticmethod
    def get_all_skus():
        return sku_config['sku_names']

    # get supplier config
    @staticmethod
    def get_supplier_num():
        return supplier_config['supplier_num']
    
    @staticmethod
    def get_supplier_name():
        return supplier_config['name']

    @staticmethod
    def get_supplier_short_name():
        return supplier_config['short_name']

    @staticmethod
    def get_supplier_info(supplier_idx):
        assert supplier_idx < Utils.get_supplier_num(), "supplier_idx must be less than total supplier number"
        keys = supplier_config.keys()
        supplier = {}
        for key in keys:
            if isinstance(supplier_config[key], list):
                supplier[key] = supplier_config[key][supplier_idx]
        return supplier

    @staticmethod
    def get_supplier_capacity(supplier_idx):
        return supplier_config['storage_capacity'][supplier_idx]

    @staticmethod
    def get_supplier_fleet_size(supplier_idx):
        assert supplier_idx < Utils.get_supplier_num(), "supplier_idx must be less than total supplier number"
        return supplier_config['fleet_size'][supplier_idx]

    @staticmethod
    def get_supplier_unit_storage_cost(supplier_idx):
        assert supplier_idx < Utils.get_supplier_num(), "supplier_idx must be less than total supplier number"
        return supplier_config['unit_storage_cost'][supplier_idx]

    @staticmethod
    def get_supplier_unit_transport_cost(supplier_idx):
        assert supplier_idx < Utils.get_supplier_num(), "supplier_idx must be less than total supplier number"
        return supplier_config['unit_transport_cost'][supplier_idx]

    @staticmethod
    def get_supplier_of_sku(sku_name):
        assert (sku_name in Utils.get_all_skus()), f"sku must be in {Utils.get_all_skus()}"
        supplier_list = []
        for supplier_idx in range(Utils.get_supplier_num()):
            sku_relation = supplier_config['sku_relation'][supplier_idx]
            for sku in sku_relation:
                if sku['sku_name'] == sku_name:
                    supplier_list.append(supplier_idx)
                    continue
        return supplier_list
    
    @staticmethod
    def get_sku_of_supplier(supplier_idx):
        assert supplier_idx < Utils.get_supplier_num(), "supplier_idx must be less than total supplier number"
        sku_relation = supplier_config['sku_relation'][supplier_idx]
        return sku_relation

    # get warehouse config
    @staticmethod
    def get_num_warehouse_echelon():
        return len(warehouse_config)

    @staticmethod
    def get_total_echelon():
        return len(warehouse_config) + 2

    @staticmethod
    def get_warehouse_num(i):
        return warehouse_config[i]['warehouse_num']

    @staticmethod
    def get_warehouse_name(i):
        return warehouse_config[i]['name']

    @staticmethod
    def get_warehouse_short_name(i):
        return warehouse_config[i]['short_name']

    @staticmethod
    def get_warehouse_info(i, warehouse_idx):
        assert warehouse_idx < Utils.get_warehouse_num(i), "warehouse_idx must be less than total warehouse number"
        keys = warehouse_config[i].keys()
        warehouse = {}
        for key in keys:
            if isinstance(warehouse_config[i][key], list):
                warehouse[key] = warehouse_config[i][key][warehouse_idx]
        return warehouse

    @staticmethod
    def get_warehouse_capacity(i, warehouse_idx):
        assert warehouse_idx < Utils.get_warehouse_num(i), "warehouse_idx must be less than total warehouse number"
        return warehouse_config[i]['storage_capacity'][warehouse_idx]

    @staticmethod
    def get_warehouse_fleet_size(i, warehouse_idx):
        assert warehouse_idx < Utils.get_warehouse_num(i), "warehouse_idx must be less than total warehouse number"
        return warehouse_config[i]['fleet_size'][warehouse_idx]

    @staticmethod
    def get_warehouse_unit_storage_cost(i, warehouse_idx):
        assert warehouse_idx < Utils.get_warehouse_num(i), "warehouse_idx must be less than total warehouse number"
        return warehouse_config[i]['unit_storage_cost'][warehouse_idx]

    @staticmethod
    def get_warehouse_unit_transport_cost(i, warehouse_idx):
        assert warehouse_idx < Utils.get_warehouse_num(i), "warehouse_idx must be less than total warehouse number"
        return warehouse_config[i]['unit_transport_cost'][warehouse_idx]

    @staticmethod
    def get_warehouse_of_sku(i, sku_name):
        assert (sku_name in Utils.get_all_skus()), f"sku must be in {Utils.get_all_skus()}"
        warehouse_list = []
        for warehouse_idx in range(Utils.get_warehouse_num(i)):
            sku_relation = warehouse_config[i]['sku_relation'][warehouse_idx]
            for sku in sku_relation:
                if sku['sku_name'] == sku_name:
                    warehouse_list.append(warehouse_idx)
                    continue
        return warehouse_list
    
    @staticmethod
    def get_sku_of_warehouse(i, warehouse_idx):
        assert warehouse_idx < Utils.get_warehouse_num(i), "warehouse_idx must be less than total warehouse number"
        sku_relation = warehouse_config[i]['sku_relation'][warehouse_idx]
        return sku_relation
    
    # get store config
    @staticmethod
    def get_store_num():
        return store_config['store_num']

    @staticmethod
    def get_store_name():
        return store_config['name']

    @staticmethod
    def get_store_short_name():
        return store_config['short_name']

    @staticmethod
    def get_store_info(store_idx):
        assert store_idx < Utils.get_store_num(), "store_idx must be less than total store number"
        keys = store_config.keys()
        store = {}
        for key in keys:
            if isinstance(store_config[key], list):
                store[key] = store_config[key][store_idx]
        return store

    @staticmethod
    def get_store_capacity(store_idx):
        assert store_idx < Utils.get_store_num(), "store_idx must be less than total store number"
        return store_config['storage_capacity'][store_idx]

    @staticmethod
    def get_store_unit_storage_cost(store_idx):
        assert store_idx < Utils.get_store_num(), "store_idx must be less than total store number"
        return store_config['unit_storage_cost'][store_idx]

    @staticmethod
    def get_store_unit_transport_cost(i):
        assert store_idx < Utils.get_store_num(), "store_idx must be less than total store number"
        return store_config['unit_transport_cost'][store_idx]

    @staticmethod
    def get_store_of_sku(sku_name):
        assert (sku_name in Utils.get_all_skus()), f"sku must be in {Utils.get_all_skus()}"
        store_list = []
        for store_idx in range(Utils.get_store_num()):
            sku_relation = store_config['sku_relation'][store_idx]
            for sku in sku_relation:
                if sku['sku_name'] == sku_name:
                    store_list.append(store_idx)
                    continue
        return store_list
    
    @staticmethod
    def get_sku_of_store(store_idx):
        assert store_idx < Utils.get_store_num(), "store_idx must be less than total store number"
        sku_relation = store_config['sku_relation'][store_idx]
        return sku_relation
