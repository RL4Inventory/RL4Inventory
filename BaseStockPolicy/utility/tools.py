from env.inventory_utils import Utils
import numpy as np
import matplotlib.pyplot as plt
import ray
import tensorflow as tf
from tensorflow.python.client import device_lib
import os
import seaborn as sns
import pickle
sns.set_style("darkgrid")

class SimulationTracker:
    def __init__(self, eposod_len, n_episods, facility_names):
        self.episod_len = eposod_len
        self.global_balances = np.zeros((n_episods, eposod_len))
        self.facility_names = list(facility_names)
        self.step_balances = np.zeros((n_episods, eposod_len, len(facility_names)))
        self.step_rewards = np.zeros((n_episods, eposod_len, len(facility_names)))
        self.n_episods = n_episods
        self.sku_to_track = None
        self.stock_status = None
        self.stock_in_transit_status = None
        self.demand_status = None
    
    def add_sample(self, episod, t, global_balance, balance, rewards):
        self.global_balances[episod, t] = global_balance
        for i, f in enumerate(self.facility_names):
            self.step_balances[episod, t, i] = balance[f]
            self.step_rewards[episod, t, i] = rewards[f]

    def add_sku_status(self, episod, t, stock, order_in_transit, demands):
        if self.sku_to_track is None:
            self.sku_to_track = set(list(stock.keys()) + list(order_in_transit.keys()) + list(demands.keys()))
            self.stock_status = np.zeros((self.n_episods, self.episod_len, len(self.sku_to_track)))
            self.stock_in_transit_status = np.zeros((self.n_episods, self.episod_len, len(self.sku_to_track)))
            self.demand_status = np.zeros((self.n_episods, self.episod_len, len(self.sku_to_track)))
        for i, sku_name in enumerate(self.sku_to_track):
            self.stock_status[episod, t, i] = stock[sku_name]
            self.stock_in_transit_status[episod, t, i] = order_in_transit[sku_name]
            self.demand_status[episod, t, i] = demands[sku_name]

    def render_sku_only_stock_status(self, train_mode, iteration, w=20, h=3):
        for i, sku_name in enumerate(self.sku_to_track):
            fig, ax = plt.subplots(1, 1, figsize=(w, h))

            x = np.linspace(0, self.episod_len, self.episod_len)

            stock = self.stock_status[0, :, i]
            order_in_transit = self.stock_in_transit_status[0, :, i]
            demand = self.demand_status[0, :, i]
            ax.set_title('SKU Stock Status by Episod')
            for y_label, y in [('stock', stock),
                               ('order_in_transit', order_in_transit),
                               ('demand', demand)]:
                ax.plot(x, y, label=y_label )
            fig.legend()
            fig.savefig(f"output/{train_mode}/iter_{iteration}/{sku_name}_only_stock_w{w}_h{h}.png")
            plt.close(fig=fig)
            fw = open(f"output/{train_mode}/iter_{iteration}/{sku_name}_only_stock.pkl", "wb")
            saved_info = {
                'name': sku_name,
                'stock': stock,
                'order_in_transit': order_in_transit,
                'demand': demand,
            }
            pickle.dump(saved_info, fw)
            fw.close()

    def render_sku(self, train_mode, iteration):
        self.render_sku_only_stock_status(train_mode, iteration, w=20, h=3)
        for i, sku_name in enumerate(self.sku_to_track):
            fig, ax = plt.subplots(3, 1, figsize=(25, 10))
            _idx = sku_name.find('_')
            _idx = sku_name.find('_', _idx+1)
            facility_consumer_name = sku_name[:_idx] + 'c'
            facility_consumer_idx = self.facility_names.index(facility_consumer_name)

            x = np.linspace(0, self.episod_len, self.episod_len)
            step_balance = self.step_balances[0, :, facility_consumer_idx]
            step_reward = self.step_rewards[0, :, facility_consumer_idx]

            ax[0].set_title('Cumulative Sum of reward')
            ax[0].plot(x, np.cumsum(step_reward))
            ax[1].set_title('Cumulative Sum of balance')
            ax[1].plot(x, np.cumsum(step_balance))

            stock = self.stock_status[0, :, i]
            order_in_transit = self.stock_in_transit_status[0, :, i]
            demand = self.demand_status[0, :, i] 
            ax[2].set_title('SKU Stock Status by Episod')
            for y_label, y in [('stock', stock), 
                               ('order_in_transit', order_in_transit), 
                               ('demand', demand)]:
                ax[2].plot(x, y, label=y_label )
            fig.legend()
            fig.savefig(f"output/{train_mode}/iter_{iteration}/{sku_name}.png")
            plt.close(fig=fig)

    def get_retailer_profit(self):
        _agent_list = []
        _step_balances_idx = []
        for i, f in enumerate(self.facility_names):
            if f.startswith('RetailerCell') and Utils.is_consumer_agent(f):
                _agent_list.append(f)
                _step_balances_idx.append(i)
        _step_balances = [self.step_balances[0, :, i] for i in _step_balances_idx]
        return np.sum(_step_balances)


    def render(self, file_name):
        fig, axs = plt.subplots(2, 1, figsize=(25, 10))
        x = np.linspace(0, self.episod_len, self.episod_len)
        
        _agent_list = []
        _step_balances_idx = []
        for i, f in enumerate(self.facility_names):
            if (f.startswith('SKUStoreUnit') or f.startswith('OuterSKUStoreUnit')) and Utils.is_consumer_agent(f):
                _agent_list.append(f)
                _step_balances_idx.append(i)
        _step_balances = [self.step_balances[0, :, i] for i in _step_balances_idx]

        # axs[0].set_title('Global balance')
        # axs[0].plot(x, self.global_balances.T)                                        

        axs[0].set_title('Cumulative Sum of Balance')
        axs[0].plot(x, np.cumsum(np.sum(_step_balances, axis = 0)) ) 
        
        
        axs[1].set_title('Reward Breakdown by Agent (One Episod)')
        axs[1].plot(x, np.cumsum(_step_balances, axis = 0).T)              
        axs[1].legend(_agent_list, loc='upper left')
        
        fig.savefig(file_name)
        # plt.show()
        
def print_hardware_status():
    
    import multiprocessing as mp
    print('Number of CPU cores:', mp.cpu_count())
    stream = os.popen('cat /proc/meminfo | grep Mem')
    print(f"Memory: {stream.read()}")
    
    stream = os.popen('lspci | grep -i nvidia ')
    print(f"GPU status: {stream.read()}")

    print(device_lib.list_local_devices())

    ray.shutdown()
    ray.init(num_gpus=1)
    print(f"ray.get_gpu_ids(): {ray.get_gpu_ids()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")