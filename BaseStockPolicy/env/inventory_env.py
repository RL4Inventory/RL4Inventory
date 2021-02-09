from abc import ABC
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from collections import deque
import numpy as np
import random as rnd
from copy import deepcopy

from numpy.lib.arraysetops import isin
import networkx as nx
from enum import Enum, auto
from agents.base import Cell, Agent, BalanceSheet
from agents.inventory import *
from agents.inventory_order import *
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Box, Tuple, MultiDiscrete, Discrete
from env.inventory_action_calculator import ActionCalculator
from env.inventory_reward_calculator import RewardCalculator
from env.inventory_state_calculator import StateCalculator
from env.inventory_utils import Utils
from agents.inventory import World
from env.gamma_sale_retailer_world import load_sale_sampler as gamma_sale_sampler
from env.online_retailer_world import load_sale_sampler as online_sale_sampler
from config.inventory_config import sku_config, supplier_config, warehouse_config, store_config

initial_balance = 1000000

class WorldBuilder():
    
    @staticmethod
    def create(x = 80, y = 32):
        world = World(x, y)
        world.grid = [[TerrainCell(xi, yi) for yi in range(y)] for xi in range(x)]

        def default_economy_config(order_cost=0, initial_balance = initial_balance):
            return ProductUnit.EconomyConfig(order_cost, initial_balance)
        
        # facility placement
        map_margin = 4
        size_y_margins = world.size_y - 2*map_margin

        supplier_x = 10
        retailer_x = 70
        
        n_supplies = Utils.get_supplier_num()
        suppliers = []
        supplier_skus = []
        supplier_sources = dict()
        for i in range(n_supplies):
            supplier_config = SupplierCell.Config(max_storage_capacity=Utils.get_supplier_capacity(i),
                                                  unit_storage_cost=Utils.get_supplier_unit_storage_cost(i),
                                                  fleet_size=Utils.get_supplier_fleet_size(i),
                                                  unit_transport_cost=Utils.get_supplier_unit_transport_cost(i))
            if n_supplies > 1:
                supplier_y = int(size_y_margins/(n_supplies - 1)*i + map_margin)
            else:
                supplier_y = int(size_y_margins/2 + map_margin)
            f = SupplierCell(supplier_x, supplier_y, 
                             world, supplier_config, 
                             default_economy_config() )
            f.idx_in_config = i
            f.facility_info = Utils.get_supplier_info(i)
            f.facility_short_name = Utils.get_supplier_short_name()
            world.agent_echelon[f.id] = 0
            world.place_cell(f) 
            suppliers.append(f)
            sku_info_list = Utils.get_sku_of_supplier(i)
            for _, sku_info in enumerate(sku_info_list):
                bom = BillOfMaterials({}, sku_info['sku_name'])
                supplier_sku_config = ProductUnit.Config(sources=None, 
                                                         unit_manufacturing_cost=sku_info['cost'], 
                                                         sale_gamma=sku_info.get('sale_gamma', 10), 
                                                         bill_of_materials=bom)
                sku = SKUSupplierUnit(f, supplier_sku_config, 
                                      default_economy_config(order_cost=f.facility_info['order_cost']) )
                sku.idx_in_config = sku_info['sku_name']
                f.sku_in_stock.append(sku)
                sku.distribution = f.distribution
                sku.storage = f.storage
                sku.sku_info = sku_info
                f.storage.try_add_units({sku_info['sku_name']: sku_info['init_stock']})
                supplier_skus.append(sku)
                if sku_info['sku_name'] not in supplier_sources:
                    supplier_sources[sku_info['sku_name']] = []
                supplier_sources[sku_info['sku_name']].append(sku)
                world.agent_echelon[sku.id] = 0

        # distribution  
        n_echelon = Utils.get_num_warehouse_echelon()
        
        pre_warehouses = suppliers
        all_warehouses = []
        warehouse_skus = []
        pre_warehouse_sources = supplier_sources
        for echelon in range(n_echelon):
            echelon_gap = (retailer_x-supplier_x)/(n_echelon+1)
            echelon_x = int(supplier_x+(echelon+1)*echelon_gap)
            n_warehouses = Utils.get_warehouse_num(echelon)
            warehouses = []
            warehouse_sources = dict()
            for i in range(n_warehouses):
                warehouse_config = WarehouseCell.Config(max_storage_capacity=Utils.get_warehouse_capacity(echelon, i), 
                                                        unit_storage_cost=Utils.get_warehouse_unit_storage_cost(echelon, i),
                                                        fleet_size=Utils.get_warehouse_fleet_size(echelon, i),
                                                        unit_transport_cost=Utils.get_warehouse_unit_transport_cost(echelon, i))
                if n_warehouses > 1:
                    warehouse_y = int(size_y_margins/(n_warehouses - 1)*i + map_margin)
                else:
                    warehouse_y = int(size_y_margins/2 + map_margin)
                w =  WarehouseCell(echelon_x, warehouse_y, 
                                world, warehouse_config, 
                                default_economy_config() )
                w.idx_in_config = i
                w.echelon_level = echelon
                w.facility_info = Utils.get_warehouse_info(echelon, i)
                w.facility_short_name = Utils.get_warehouse_short_name(echelon)
                world.agent_echelon[w.id] = 1+echelon
                world.place_cell(w) 
                warehouses.append(w)
                WorldBuilder.connect_cells(world, w, *pre_warehouses)
                sku_info_list = Utils.get_sku_of_warehouse(echelon, i)
                for _, sku_info in enumerate(sku_info_list):
                    candidate_upstream_suppliers = pre_warehouse_sources[sku_info['sku_name']]
                    upstream_suppliers = []
                    for s in candidate_upstream_suppliers:
                        if i in s.facility.facility_info['downstream_facilities']:
                            upstream_suppliers.append(s)
                    bom = BillOfMaterials({sku_info['sku_name']: 1}, sku_info['sku_name'])
                    warehouse_sku_config = ProductUnit.Config(sources=upstream_suppliers, 
                                                              unit_manufacturing_cost=sku_info.get('cost', 10), 
                                                              sale_gamma=sku_info.get('sale_gamma', 10), 
                                                              bill_of_materials=bom)
                    sku = SKUWarehouseUnit(w, warehouse_sku_config,
                                           default_economy_config(order_cost= w.facility_info['order_cost']) )
                    sku.idx_in_config = sku_info['sku_name']
                    w.sku_in_stock.append(sku)
                    sku.distribution = w.distribution
                    sku.storage = w.storage
                    sku.sku_info = sku_info
                    warehouse_skus.append(sku)
                    w.storage.try_add_units({sku_info['sku_name']: sku_info.get('init_stock', 0)})
                    if sku_info['sku_name'] not in warehouse_sources:
                        warehouse_sources[sku_info['sku_name']] = []
                    warehouse_sources[sku_info['sku_name']].append(sku)
                    world.agent_echelon[sku.id] = 1+echelon
                    # update downstreaming sku list in supplier_list
                    for s_sku in upstream_suppliers:
                        s_sku.downstream_skus.append(sku)

            all_warehouses.extend(warehouses)
            pre_warehouse_sources = warehouse_sources
            pre_warehouses = warehouses

        # final consumers
        n_stores = Utils.get_store_num()
        stores = []
        store_skus = []
        for i in range(n_stores):
            store_config = RetailerCell.Config(max_storage_capacity=Utils.get_store_capacity(i), 
                                               unit_storage_cost=Utils.get_store_unit_storage_cost(i),
                                               fleet_size=1000,
                                               unit_transport_cost=10)
            if n_stores > 1:
                retailer_y = int(size_y_margins/(n_stores - 1)*i + map_margin)
            else:
                retailer_y = int(size_y_margins/2 + map_margin)
            r = RetailerCell(retailer_x, retailer_y, 
                             world, store_config, 
                             default_economy_config() )
            r.idx_in_config = i
            r.facility_info = Utils.get_store_info(i)
            r.facility_short_name = Utils.get_store_short_name()
            world.agent_echelon[r.id] = 1+n_echelon
            world.place_cell(r)
            stores.append(r)
            WorldBuilder.connect_cells(world, r, *pre_warehouses)
            sku_info_list = Utils.get_sku_of_store(i)
            for _, sku_info in enumerate(sku_info_list):
                candidate_upstream_warehouses = pre_warehouse_sources[sku_info['sku_name']]
                upstream_warehouses = []
                for s in candidate_upstream_warehouses:
                    if i in s.facility.facility_info['downstream_facilities']:
                        upstream_warehouses.append(s)
                bom = BillOfMaterials({sku_info['sku_name']: 1}, sku_info['sku_name'])
                retail_sku_config = ProductUnit.Config(sources=upstream_warehouses, 
                                                       unit_manufacturing_cost=sku_info.get('cost', 10), 
                                                       sale_gamma=sku_info.get('sale_gamma', 10), 
                                                       bill_of_materials=bom)
                                
                if Utils.get_demand_sampler() == "DYNAMIC_GAMMA":
                    sku = SKUStoreUnit(r, retail_sku_config, default_economy_config(order_cost=r.facility_info['order_cost']) )
                elif Utils.get_demand_sampler() == "GAMMA":
                    sale_sampler = gamma_sale_sampler(i)
                    sku = OuterSKUStoreUnit(r, retail_sku_config, default_economy_config(order_cost=r.facility_info['order_cost']), sale_sampler )
                else:
                    sale_sampler = online_sale_sampler(f"data/OnlineRetail/store{i+1}_new.csv")
                    sku = OuterSKUStoreUnit(r, retail_sku_config, default_economy_config(order_cost=r.facility_info['order_cost']), sale_sampler )
                sku.idx_in_config = sku_info['sku_name']
                r.sku_in_stock.append(sku)
                sku.storage = r.storage
                sku.sku_info = sku_info
                r.storage.try_add_units({sku_info['sku_name']:  sku_info.get('init_stock', 0)})
                store_skus.append(sku)
                world.agent_echelon[sku.id] = 1+n_echelon

                # update downstreaming sku list in warehouse_list
                for w_sku in upstream_warehouses:
                    w_sku.downstream_skus.append(sku)
    
        for facility in suppliers + all_warehouses + stores:
            world.facilities[facility.id] = facility
        for sku in supplier_skus + warehouse_skus + store_skus:
            world.facilities[sku.id] = sku
            if sku.sku_info.get('price', 0) > world.max_price:
                world.max_price = sku.sku_info.get('price', 0)
        world.total_echelon = Utils.get_total_echelon()
        return world
        
    @staticmethod
    def connect_cells(world, source, *destinations):
        for dest_cell in destinations:
            WorldBuilder.build_railroad(world, source.x, source.y, dest_cell.x, dest_cell.y)

    @staticmethod    
    def build_railroad(world, x1, y1, x2, y2):
        step_x = np.sign(x2 - x1)
        step_y = np.sign(y2 - y1)

        # make several attempts to find a route non-adjacent to existing roads  
        for i in range(5):
            xi = min(x1, x2) + int(abs(x2 - x1) * rnd.uniform(0.15, 0.85))
            if not (world.is_railroad(xi-1, y1+step_y) or world.is_railroad(xi+1, y1+step_y)):
                break

        for x in range(x1 + step_x, xi, step_x):
            world.create_cell(x, y1, RailroadCell) 
        if step_y != 0:
            for y in range(y1, y2, step_y):
                world.create_cell(xi, y, RailroadCell) 
        for x in range(xi, x2, step_x):
            world.create_cell(x, y2, RailroadCell) 
    
    
class InventoryManageEnv(MultiAgentEnv):
    def __init__(self, env_config):
        self.env_config = env_config
        if(self.env_config['training'] and env_config['init']=='rnd'):
            self.copy_world = None
            self.world_idx = rnd.randint(1, env_config['episod_duration'])
        self.world = WorldBuilder.create(80, 16)
        self.current_iteration = 0
        self.n_iterations = 0
        self.policies = None
        # self.trainer = None
        
        self.product_ids = self._product_ids()
        # 存储当前最大的前置商品数量和车辆数量
        self.max_sources_per_facility = 0
        self.max_fleet_size = 0
        self.facility_types = {}
        facility_class_id = 0
        for f in self.world.facilities.values():
            if isinstance(f, FacilityCell):
                sources_num = 0
                for sku in f.sku_in_stock:
                    if sku.consumer is not None and sku.consumer.sources is not None:
                        sources_num = len(sku.consumer.sources)
                        if sources_num > self.max_sources_per_facility:
                            self.max_sources_per_facility = sources_num
                    
                if f.distribution is not None:      
                    if len(f.distribution.fleet) > self.max_fleet_size:
                        self.max_fleet_size = len(f.distribution.fleet)
                    
            facility_class = f.__class__.__name__
            if facility_class not in self.facility_types:
                self.facility_types[facility_class] = facility_class_id
                facility_class_id += 1
                
        self.state_calculator = StateCalculator(self)
        self.reward_calculator = RewardCalculator(env_config)
        self.action_calculator = ActionCalculator(self)
                         
        self.action_space_producer = MultiDiscrete([ 
            1,                             # unit price
            1,                             # production rate level
        ])
        
        
        # self.action_space_consumer = MultiDiscrete([ 
        #     self.max_sources_per_facility,               # consumer source id
        #     len(Utils.get_consumer_action_space())         # consumer_quantity
        # ])

        self.action_space_consumer = Discrete(len(Utils.get_consumer_action_space())) 
                
        example_state, _ = self.state_calculator.world_to_state(self.world)
        state_dim = len(list(example_state.values())[0])
        
        # 计算状态空间的大小，每个facility对应一个完整的状态
        self.observation_space = Box(low=-300.00, high=300.00, shape=(state_dim, ), dtype=np.float64)
    
    def tail_running(self, rewards, obss, infos):
        policies = self.policies
        gamma = self.env_config['gamma']
        rnn_states = {}
        for agent_id in obss.keys():
            rnn_states[agent_id] = policies[agent_id].get_initial_state()
        discount = 1.
        for epoch in range(self.env_config['tail_timesteps']):
            action_dict = {}
            for agent_id, obs in obss.items():
                policy = policies[agent_id]
                action, _, _ = policy.compute_single_action(obs, state=rnn_states[agent_id], info=infos[agent_id],
                                                                    explore=False)
                action_dict[agent_id] = action
            control = self.action_calculator.action_dictionary_to_control(action_dict, self.world)
            outcome = self.world.act(control)
            
            cur_rewards = self.reward_calculator.calculate_reward(self, outcome)
            obss, infos = self.state_calculator.world_to_state(self.world)
            discount *= gamma
            for agent_id in obss.keys():
                rewards[agent_id] += discount*cur_rewards[agent_id]
        return rewards
    
    def set_retailer_step(self, step):
        for sku in self.world.facilities.values():
            if isinstance(sku, OuterSKUStoreUnit):
                sku.seller.set_step(step)
                sku.seller._init_sale_pred()


    def reset(self):
        # print("Rst!")
        if(self.env_config['training'] and self.env_config['init']=='rnd' and self.copy_world):
            self.world = self.copy_world
        elif(self.env_config['training'] and self.env_config['init']=='rst'):
            self.world.reset()
        else:
            self.world = WorldBuilder.create(80, 16)
        state, _ = self.state_calculator.world_to_state(self.world)
        if(Utils.get_demand_sampler()=='ONLINE'):
            self.set_retailer_step(0)
        # print(state)
        return state
    
    def set_policies(self, plcs):
        # print("Policy installed")
        # print(plcs)
        self.policies = plcs
    
    def set_trainer(self, trainer):
        self.trainer = trainer

    def step(self, action_dict):
        control = self.action_calculator.action_dictionary_to_control(action_dict, self.world)
        outcome = self.world.act(control)
        
        # churn through no-action cycles 
        # 系统自然运行，无需决策
        for _ in range(self.env_config['downsampling_rate'] - 1): 
            nop_outcome = self.world.act(World.Control({}))
            
            balances = outcome.facility_step_balance_sheets
            for agent_id in balances.keys():
                # balances[agent_id] = balances[agent_id] + nop_outcome.facility_step_balance_sheets[agent_id]
                outcome.facility_step_balance_sheets[agent_id] += nop_outcome.facility_step_balance_sheets[agent_id]
            
        all_done = (self.world.time_step >= self.env_config['episod_duration'])
        # all_done = (self.world.time_step >= self.env_config['episod_duration']+20)
        # all_done = False
        if(self.env_config['training'] and self.env_config['init']=='rnd' and self.world.time_step==self.world_idx):
            self.copy_world = deepcopy(self.world)
            # self.copy_world.reset()
            self.copy_world.time_step = 0
            self.world_idx = rnd.randint(1, self.env_config['episod_duration']-1)
        rewards = self.reward_calculator.calculate_reward(self, outcome)
        seralized_states, info_states = self.state_calculator.world_to_state(self.world)

        # if self.policies:
        #     print(self.policies)

        if all_done and self.env_config['training'] and self.policies:
            rewards = self.tail_running(rewards, seralized_states, info_states)
            
        # if all_done:
        #   # update reward/cost for the last episode
        #   for agent_id in rewards.keys():
        #       facility = self.world.facilities[Utils.agentid_to_fid(agent_id)]
        #       if Utils.is_consumer_agent(agent_id) and isinstance(facility, ProductUnit):
        #           remaining_inventory_cost = ((info_states[agent_id]['inventory_in_transit']
        #                                         + info_states[agent_id]['inventory_in_stock']) * info_states[agent_id]['sku_cost'])
        #           rewards[agent_id] -= remaining_inventory_cost

        # for agent_id in seralized_states.keys():
        #     facility_id = Utils.agentid_to_fid(agent_id)
        #     if Utils.is_consumer_agent(agent_id) and isinstance(self.world.facilities[facility_id], RetailerCell):
        #         balance = self.world.facilities[facility_id].economy.total_balance.total()
        #         # print(self.timestep, agent_id, balance)
        #         if balance < 0 or balance >= 5*initial_balance:
        #             all_done = True
        
        dones = { agent_id: all_done for agent_id in seralized_states.keys() }
        dones['__all__'] = all_done
        # print(self.timestep, seralized_states['SKUStoreUnit_8c'], rewards['SKUStoreUnit_8c'])
        
        return seralized_states, rewards, dones, info_states
    
    def agent_ids(self):
        agents = []
        for f_id in self.world.facilities.keys():
            agents.append(Utils.agentid_producer(f_id))
        for f_id in self.world.facilities.keys():
            agents.append(Utils.agentid_consumer(f_id))
        return agents
    
    def set_iteration(self, iteration, n_iterations):
        self.current_iteration = iteration
        self.n_iterations = n_iterations
    
    def n_products(self):
        return len(self._product_ids())

    def get_stock_status(self):
        stock_info = dict()
        for facility in self.world.facilities.values():
            if isinstance(facility, ProductUnit):
                sku_name = facility.sku_info['sku_name']
                facility_key = f"{facility.id}_{sku_name}_{facility.facility.id}_{facility.facility.idx_in_config}"
                stock_info[facility_key] = facility.storage.stock_levels.get(sku_name, 0)
            elif isinstance(facility, FacilityCell):
                facility_key = f"{facility.id}_{facility.idx_in_config}"
                stock_info[facility_key] = np.sum(list(facility.storage.stock_levels.values()))
        return stock_info

    def get_demand_status(self):
        demand_info = dict()
        for facility in self.world.facilities.values():
            if isinstance(facility, ProductUnit):
                sku_name = facility.sku_info['sku_name']
                facility_key = f"{facility.id}_{sku_name}_{facility.facility.id}_{facility.facility.idx_in_config}"
                demand_info[facility_key] = facility.get_latest_sale()
            else:
                facility_key = f"{facility.id}_{facility.idx_in_config}"
                demand_info[facility_key] = 0
        return demand_info
    
    def get_reward_status(self):
        reward_info = dict()
        for facility in self.world.facilities.values():
            if isinstance(facility, ProductUnit):
                sku_name = facility.sku_info['sku_name']
                facility_key = f"{facility.id}_{sku_name}_{facility.facility.id}_{facility.facility.idx_in_config}"
                reward_info[facility_key] = facility.step_reward
            else:
                facility_key = f"{facility.id}_{facility.idx_in_config}"
                reward_info[facility_key] = 0
        return reward_info
        
    def get_order_in_transit_status(self):
        order_in_transit_info = dict()
        for facility in self.world.facilities.values():
            if isinstance(facility, ProductUnit):
                sku_name = facility.sku_info['sku_name']
                facility_key = f"{facility.id}_{sku_name}_{facility.facility.id}_{facility.facility.idx_in_config}"
                order_in_transit_info[facility_key] = 0
                if facility.consumer is not None and facility.consumer.sources is not None:
                    for source in facility.consumer.sources:
                        order_in_transit_info[facility_key] += facility.consumer.open_orders.get(source.id, {}).get(sku_name, 0)
            elif isinstance(facility, FacilityCell):
                facility_key = f"{facility.id}_{facility.idx_in_config}"
                order_in_transit_info[facility_key] = 0
                for sku in facility.sku_in_stock:
                    if sku.consumer is not None and sku.consumer.sources is not None:
                        for source in sku.consumer.sources:
                            order_in_transit_info[facility_key] += sku.consumer.open_orders.get(source.id, {}).get(sku.sku_info['sku_name'], 0)
        return order_in_transit_info

    # 获取所有商品ID
    def _product_ids(self):
        return Utils.get_all_skus()
