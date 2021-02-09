import sys
sys.path.append('/workspace/')
import cvxpy as cp
import random as rnd
import numpy as np
import matplotlib.pyplot as plt
from config.inventory_config import env_config
from agents.inventory import *
from env.inventory_env import *
from env.inventory_utils import Utils
from tqdm import tqdm
from scheduler.inventory_random_policy import ConsumerBaselinePolicy
from scheduler.inventory_random_policy import ProducerBaselinePolicy, BaselinePolicy
from scheduler.inventory_minmax_policy import ConsumerMinMaxPolicy



class ConsumerBaseStockPolicy(ConsumerBaselinePolicy):
    '''
    the base-stock policy
    '''
    # These fields serve for global param purposes.
    step = -1
    base_stocks = {}
    env_config = {}
    facilities = {}
    update_interval = 1
    start_step = 0
    time_hrz_len = 0
    oracle = False
    buyin_gap = 0
    stop_order_factor = 1
    def __init__(self, observation_space, action_space, config, is_static):
        ConsumerBaselinePolicy.__init__(self, observation_space, action_space, config)
        self.static = is_static
        self.step = -1

    @staticmethod
    def update_base_stocks():
        '''update the base-stock levels'''
        sku_base_stocks = {}
        time_hrz_len = ConsumerBaseStockPolicy.time_hrz_len
        skus = []
        for facility in ConsumerBaseStockPolicy.facilities.values():
            if isinstance(facility, ProductUnit):
                skus.append(facility)
        _sku_base_stocks = ConsumerBaseStockPolicy.get_base_stock(skus, time_hrz_len)
        sku_base_stocks.update(_sku_base_stocks)
        ConsumerBaseStockPolicy.base_stocks.update(sku_base_stocks)
    
    @staticmethod
    def get_real_demands(seller):
        return np.array(seller.get_future_sales(ConsumerBaseStockPolicy.time_hrz_len))

    @staticmethod
    def _base_stock_config(SKU):
        sale_mean = SKU.get_sale_mean()
        config = {'name': SKU.id,
                  'price': SKU.sku_info['price'], 
                  'vlt': SKU.sku_info.get('vlt', 0),
                  'T0': 0, 
                  'I0': SKU.sku_info['init_stock'],
                  'U0': 0,
                  'h': SKU.facility.facility_info['unit_storage_cost'],
                  'cost': SKU.sku_info['cost'],
                  'unit_transport_cost':SKU.facility.facility_info.get('unit_transport_cost', 0),
                  'order_cost':SKU.facility.facility_info['order_cost'],
                  'sale_mean':sale_mean if sale_mean else 1,
                  'is_store': isinstance(SKU, SKUStoreUnit),
                  'is_supp': isinstance(SKU, SKUSupplierUnit),
                  'ups': [], 'dwns': []}
        if config['is_store']:
            name = config['name']
        if SKU.consumer.sources is not None:
            for source in SKU.consumer.sources:
                config['ups'].append(source.id)
        for downstream_sku in SKU.downstream_skus:
            config['dwns'].append(downstream_sku.id)
        # print(config['name'], config['ups'], config['dwns'])
        if isinstance(SKU, SKUStoreUnit):
            config['k'] = SKU.sku_info.get('backlog_ratio', 0.1)
            if ConsumerBaseStockPolicy.oracle:
                config['D'] = ConsumerBaseStockPolicy.get_real_demands(SKU.seller)
            else:
                config['D'] = SKU.seller.sale_hist
        return config

    @staticmethod
    def _get_upstream_skus(SKU):
        _upstream_skus = [SKU]
        if isinstance(SKU, SKUSupplierUnit):
            return _upstream_skus
        else:
            for source in SKU.consumer.sources:
                _upstream_skus.extend(ConsumerBaseStockPolicy._get_upstream_skus(source))
            return _upstream_skus
    
    @staticmethod
    def _get_downstream_skus(SKU):
        _downstream_skus = [SKU]
        if isinstance(SKU, SKUStoreUnit):
            return _downstream_skus
        else:
            for source in SKU.downstream_skus:
                _downstream_skus.extend(ConsumerBaseStockPolicy._get_downstream_skus(source))
            return _downstream_skus

    @staticmethod
    def get_base_stock(SKUs, time_hrz_len):
        base_stock_config = []
        for SKU in SKUs:
            base_stock_config.append(ConsumerBaseStockPolicy._base_stock_config(SKU))
            # _upstream_skus = ConsumerBaseStockPolicy._get_upstream_skus(SKU)
            # _downstream_skus = ConsumerBaseStockPolicy._get_downstream_skus(SKU)
            # for _SKU in _upstream_skus + _downstream_skus:
            #     is_configged.add(_SKU.id)
            #     base_stock_config.append(ConsumerBaseStockPolicy._base_stock_config(_SKU))
        solver = Solver(Config(base_stock_config, time_hrz_len))
        solver.constrs_fml()
        solver.solve()
        unit_table = solver.unit_table
        _base_stock = {}
        buyin_gap = ConsumerBaseStockPolicy.buyin_gap
        for key in unit_table.keys():
            unit = unit_table[key]
            bs = unit.z.value
            rs = unit.buy_in.value
            # _base_stock[key] = [(b if int(r) else 0) for b, r in zip(bs, rs)]
            if buyin_gap > 1:
                _base_stock[key] = [bs[i] if i%buyin_gap==0 else 0 for i in range(time_hrz_len)]
            else:
                _base_stock[key] = bs
            # if unit.is_store:
            #     print(unit.R[unit.ups[0]].value[time_hrz_len:])
        return _base_stock


    def _find_source(self, f_state_info):
        # stop placing orders when the facility ran out of money 
        # if f_state_info['is_positive_balance'] <= 0:
        #     return (0, 0, 0)
        self.step += 1
        step = ConsumerBaseStockPolicy.step
        update_interval = ConsumerBaseStockPolicy.update_interval
        if(not self.static) and step!=self.step:
            ConsumerBaseStockPolicy.step += 1
            step = ConsumerBaseStockPolicy.step
            if step%update_interval==0:
                self.update_base_stocks()

        facility = f_state_info['facility']
        facility_type = type(facility)
        if facility_type not in [SKUWarehouseUnit, SKUStoreUnit, OuterSKUStoreUnit]:
            return 0
        
        # consumer_source_inventory
        inputs = f_state_info['bom_inputs']
        available_inventory = np.array(f_state_info['storage_levels'])
        inflight_orders = np.array(f_state_info['consumer_in_transit_orders'])
        booked_inventory = available_inventory + inflight_orders
        
        # stop placing orders when the facilty runs out of capacity
        # if np.sum(booked_inventory) > f_state_info['storage_capacity']:
        #     return 0
        
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

        start_step = ConsumerBaseStockPolicy.start_step
        shift = step%update_interval
        base = ConsumerBaseStockPolicy.base_stocks[facility.id][start_step+shift]
        reorder = base - booked_inventory[most_needed_product_id]
        
        # whether replenishment point is reached
        if reorder<=0:
            return 0
        
        factor = ConsumerBaseStockPolicy.stop_order_factor
        if reorder+np.sum(booked_inventory)>factor*f_state_info['storage_capacity']:
            reorder = factor*f_state_info['storage_capacity']-np.sum(booked_inventory)
        action = reorder/f_state_info['sale_mean']
        consumer_quantity = Utils.get_consumer_quantity_action(action)
        return consumer_quantity



class Config:
    '''
    Configuration that feeds the solver
    '''
    def __init__(self, config, time_hrz_len):
        '''
        Initialize a config object

        Attrs:
        | use_rnd_seed: to use seed or not
        | seed:         the random seed
        | time_hrz_len: the length of time horizon
        | dcs_step: the temporal interval that we span to renew the decisions
        | config: an attr that encodes the logistics relations
        | Keys in the config:
        | | name:       the unit name
        | | vlt:        the leading time the unit takes to transfer to a downstream unit
        | | price:      the price per SKU unit
        | | I0:         the initial inventory level
        | | T0:         the initial outstanding SKU level
        | | h:          the holding cost per unit
        | | is_supp:    True if the unit is a supplier
        | | is_store:   True if the unit is a store
        | | ups:        Upstream units
        | | dwns:       Downstream units
        | | k (store):  Cost per unfulfilled demands one timestep (lost sales)
        | |             or per timestep (backlogging)
        | | D (store):  Demands
        | | U0 (store): initial unfulfilled demand
        '''
        self.use_rnd_seed = True
        self.seed = 30
        if(self.use_rnd_seed):
            np.random.seed(self.seed)
        self.time_hrz_len = time_hrz_len
        '''The logistics 
        o   o
         \ /
          o
         / \
        o   o
        '''
        # self.config=[
        #     {
        #         "name":"U1", "vlt":5, "price":300, "I0":1000000000, "T0":0, "h":1, "is_supp":True,
        #         "is_store":False, "ups":[], "dwns":["U3"]
        #     },
        #     {
        #         "name":"U2", "vlt":4, "price":310, "I0":1000000000, "T0":0, "h":1, "is_supp":True,
        #         "is_store":False, "ups":[], "dwns":["U3"]
        #     },
        #     {
        #         "name":"U3", "vlt":3, "price":0, "I0":2000, "T0":0, "h":1, "is_supp":False,
        #         "is_store":False, "ups":["U1","U2"], "dwns":["U4","U5"]
        #     },
        #     {
        #         "name":"U4", "vlt":0, "price":430, "I0":2000, "T0":0, "h":1, "is_supp":False,
        #         "is_store":True, "ups":["U3"], "dwns":[], "k":200, "D":self.demands_generator(),
        #         "U0": 0
        #     },
        #     {
        #         "name":"U5", "vlt":0, "price":430, "I0":2000, "T0":0, "h":1, "is_supp":False,
        #         "is_store":True, "ups":["U3"], "dwns":[], "k":200, "D":self.demands_generator(),
        #         "U0": 0
        #     }
        # ]
        '''The logistics 
              o
           o<
          /   o
         /
        o
         \
          \   o
           o<
              o
        '''
        self.config = config
        self.timestep = 0
    
    def get_R_init(self):
        return np.zeros(self.time_hrz_len, dtype=int)
    
    def var_sum(self, vars:list):
        sumup = 0
        for var in vars:
            sumup += var
        return sumup
        
    @staticmethod
    def demands_generator(time_hrz_len):
        return np.random.randint(400,
                                700,
                                time_hrz_len,
                                dtype=int)
    
    def unpack(self, pack:list):
        res = []
        for p in pack:
            res += p
        return res


class Solver:
    '''
    Solver used to solve the replenishment model
    '''
    def __init__(self, config:Config):
        '''
        Attrs:
        | config:       the config object
        | time_hrz_len: the length of time horizon
        | unit_table:   the dict used for unit lookup
        | n_supps:      the # of supplier units
        | n_stores:     the # of store units
        | constrs:      all the constraints used to solve the model
        | P:            the var representing the overall profit
                        the target var that we would like to maximize
        '''
        self.config = config
        self.time_hrz_len = config.time_hrz_len
        self.unit_table = {}
        self.n_supps = 0
        self.n_stores = 0
        self.constrs = []
        self.config_parse()
        
        #vars
        self.P = cp.Variable(integer=True)
    
    def config_parse(self):
        '''
        config parser method
        '''
        for c in self.config.config:
            self.unit_table[c['name']]=Unit(c, self)
            if(c['is_supp']):
                self.n_supps+=1
            elif(c['is_store']):
                self.n_stores+=1

    def show_var(self, attr, unit_name=None, hrz_only=True):
        '''
        Plot indicators

        Params:
            attr: the attribute of units
            unit_name[optional]: unit specified by its name
            hrz_only[optional]: clip out the timestep preceding the horizon start
        '''
        print(f"==============={attr}==============")
        if(unit_name):
            print(unit_name, "within time horizon" if hrz_only
                  else "the whole axis (not limited to hrz)")
            unit = self.unit_table[unit_name]
            if(not hasattr(unit, attr)):
                print(f"{unit_name} doesn't have {attr}")
                return
            var = getattr(unit, attr)
            is_var = isinstance(var, cp.Variable)
            is_arr = isinstance(var, np.ndarray)
            is_dict = isinstance(var, dict)
            assert  is_var or is_arr or is_dict
            plt.figure()
            if not is_dict:
                if(is_var):
                    var = var.value
                res = var[-self.time_hrz_len:] if hrz_only else var
                plt.plot(res)
                print(res)
            else:
                for pu_name in unit.ups:
                    res = var[pu_name][-self.time_hrz_len:].value\
                          if hrz_only else var[pu_name].value
                    plt.plot(res)
                    print(res)
            plt.title(f"{attr} of {unit_name}")
        else:
            print("all units", "within time horizon" if hrz_only
                  else "the whole axis (not limited to hrz)")
            for uname in self.unit_table:
                print(uname)
                unit = self.unit_table[uname]
                if(not hasattr(unit, attr)):
                    print(f"{uname} doesn't have {attr}")
                    continue
                var = getattr(unit, attr)
                is_var = isinstance(var, cp.Variable)
                is_arr = isinstance(var, np.ndarray)
                is_dict = isinstance(var, dict)
                assert is_var or is_arr or is_dict
                plt.figure()
                if(not is_dict):
                    if(is_var):
                        var = var.value
                    res = var[-self.time_hrz_len:] if hrz_only else var
                    plt.plot(res)
                    print(res)
                else:
                    for pu_name in unit.ups:
                        res = var[pu_name][-self.time_hrz_len:].value\
                            if hrz_only else var[pu_name].value
                        plt.plot(res)
                        print(res)
                plt.title(f"{attr} of {uname}")

    def get_constrs(self):
        return self.constrs

    def append_constrs(self, constrs_new):
        if(not isinstance(constrs_new, list)):
            constrs_new = [constrs_new]
        constrs = self.get_constrs()
        constrs += constrs_new
    
    def constrs_fml(self):
        unit_table = self.unit_table
        # for retailer in ConsumerBaseStockPolicy.facilities.values():
        #     if isinstance(retailer, RetailerCell):
        #         storage_capacity = retailer.facility_info['storage_capacity']
        #         I_sumup = 0
        #         for sku in retailer.sku_in_stock:
        #             unit_name = sku.id
        #             unit = unit_table[unit_name]
        #             I_sumup += unit.I
        #         self.append_constrs(
        #             [I_sumup[1:]<=storage_capacity]
        #         )

        P_sumup = 0
        for unit_name in self.unit_table:
            unit = self.unit_table[unit_name]
            unit.constrs_fml()
            P_sumup += cp.sum(unit.P)
        P = self.P
        self.append_constrs(
            [P==P_sumup]
        )
    
    def obj_setup(self):
        return cp.Maximize(self.P)
    
    def solve(self, verbose=False):
        obj = self.obj_setup()
        prob = cp.Problem(obj, self.get_constrs())
        prob.solve(verbose=verbose)
        print("Status:", prob.status)
        print("The optimal value is", self.P.value)
        
class Unit:
    def __init__(self, config:dict, solver:Solver):
        '''
        Initialize the unit object

        Attrs:
        | config:           the config dict of the unit, NOT A CONFIG OBJECT
                            basically is part of the attr
                            `config` of the config object
        | solver:           the solver object
        | name:             the unit name
        | vlt:              leading time
        | price:            price of selling SKUs
        | I0:               the initial inventory
        | T0:               the initial outstanding order items
        | h:                the holding cost per unit
        | is_supp:          True if the unit is a supplier
        | is_store:         True if the unit is a store
        | ups:              upstream units
        | dwns:             downstream units
        | time_hrz_len:     length of the time horizon
        | pn_units:         the number of upstream units
        | k(store):         cost per unfulfilled demands one timestep (lost sales)
                            or per timestep (backlogging)
        | U0(store):        initial unfulfilled demand

        Attrs [Vars]:
        | D(store)[hrz_len]:    demands, temporal dim from 0~hrz_len
        | U(store)[hrz_len+1]:  unfulfilled demands, temporal dim from -1~hrz_len   
        | I[hrz_len+1]:         inventory
                                temporal dim from -1~hrz_len, requiring I0
        | T[hrz_len+1]:         outstanding items number
                                temporal dim from -1~hrz_len, requiring T0
        | S[hrz_len]:           sales, temporal dim from 0~hrz_len
        | R[pu_num, 2*hrz_len]: reorders,
                                specified by the previous upstream unit,
                                temporal dim from -hrz_len~hrz_len
                                to contain more preceding reorders
        | buy_in[hrz_len]:      SKU num that is reordered at each timestep
        | buy_arr[hrz_len]:     SKU num that arrived at each timestep
                                can be seen as buy_in shifted 
                                by corresponding leading times
        | inv_pos[hrz_len]:     inventory position, temporal dim from 0~hrz_len
        | z[hrz_len]:           base-stock level, temporal dim from 0~hrz_len
        | P[hrz_len]:           profit at each timestep,
                                temporal dim from 0~hrz_len
        '''
        self.config = config
        self.solver = solver
        self.name = config["name"]
        self.vlt = config["vlt"]
        self.price = config['price']
        self.proc_price = config['cost']
        # print(self.proc_price, self.price)
        self.I0 = config['I0']
        self.T0 = config['T0']
        self.h = config['h']
        self.is_supp = config['is_supp']
        self.is_store = config['is_store']
        self.ups = config['ups']
        self.dwns = config['dwns']
        self.unit_tc = config['unit_transport_cost']
        self.order_cost = config['order_cost']
        self.sale_mean = config['sale_mean']
        self.time_hrz_len = solver.time_hrz_len
        self.pn_units = len(self.ups)
        if(self.is_store):
            self.k = config['k']
            self.U0 = config['U0']
            self.D = config['D']
            # self.U = cp.Variable(self.time_hrz_len+1,integer=True)
        # Variables
        # 1d (temporal dim:|T|+1)
        self.I = cp.Variable(self.time_hrz_len+1, integer=True) 
        self.T = cp.Variable(self.time_hrz_len+1, integer=True)
        self.S = cp.Variable(self.time_hrz_len, integer=True)
        self.R = {pu_name: cp.Variable(2*self.time_hrz_len, integer=True)
        for pu_name in self.ups}
        self.buy_in = cp.Variable(self.time_hrz_len, integer=True)
        self.buy_arv = cp.Variable(self.time_hrz_len, integer=True)
        self.inv_pos = cp.Variable(self.time_hrz_len, integer=True)
        self.z = cp.Variable(self.time_hrz_len, integer=True)
        self.P = cp.Variable(integer=True)

        # constrs
        self.append_constrs([
            self.I>=0, 
            self.T>=0,
            self.S>=0
        ])
        if(self.is_store):
            self.append_constrs(
                [
                    # self.U>=0
                ]
            )
        for pu in self.ups:
            self.append_constrs([
                self.R[pu]>=0,
                self.R[pu][:self.time_hrz_len]==0
            ])
    
    def constrs_fml(self):
        I = self.I
        T = self.T
        S = self.S
        R = self.R
        buy_in = self.buy_in
        buy_arv = self.buy_arv
        P = self.P
        h = self.h
        price = self.price
        inv_pos = self.inv_pos
        z = self.z
        unit_tc = self.unit_tc
        if(self.is_store):
            # U = self.U
            D = self.D
            k = self.k
            proc_price = self.proc_price
            buyin_gap = ConsumerBaseStockPolicy.buyin_gap
            if buyin_gap > 1:
                self.append_constrs(
                    [buy_in[[i%buyin_gap!=0 for i in range(self.time_hrz_len)]]==0]
                )
        
        self.append_constrs(
            [
                # I[0]==self.I0,
                T[0]==self.T0,
                I[1:self.time_hrz_len+1]==I[:self.time_hrz_len]+buy_arv-S,
                T[1:self.time_hrz_len+1]==T[:self.time_hrz_len]-buy_arv+buy_in,
                S<=I[:self.time_hrz_len],
                (S<=D)if self.is_store else (S==self.get_sale_out()),
                buy_in == self.get_buy_in(),
                buy_arv == self.get_buy_arv(),
                inv_pos==I[:self.time_hrz_len]+T[:self.time_hrz_len],
                z==inv_pos+buy_in
            ]
        )
        if(self.is_store):
            self.append_constrs(
                [
                    # U[0]==self.U0,
                    # U[1:]==D-S,
                    P==cp.sum(
                            (price-unit_tc)*S-h*I[1:]-buy_in*proc_price
                            # (price-unit_tc)*S-k*U[1:]-h*I[1:]-buy_in*proc_price
                        ) - proc_price*I[0]
                ]
            )
        elif(self.is_supp):
            self.append_constrs(
                [
                    # P==cp.sum((-price-unit_tc)*S) #we don't count the holding cost here
                    P==0
                ]
            )
        else:
            self.append_constrs(
                [
                    P==0
                    # P == cp.sum(-h*I[1:])
                ]
            )

    def get_unit_table(self):
        return self.solver.unit_table

    def get_sale_out(self):
        name = self.name
        res = 0
        unit_table = self.get_unit_table()
        for nu_name in self.dwns:
            nu = unit_table[nu_name]
            res+=nu.R[name][self.time_hrz_len:]
        return res

    def get_buy_in(self):
        res = 0
        for pu_name in self.ups:
            res += self.R[pu_name][self.time_hrz_len:]
        return res

    def get_buy_arv(self):
        res = 0
        unit_table = self.get_unit_table()
        for pu_name in self.ups:
            pu = unit_table[pu_name]
            pvlt = pu.vlt
            res += self.R[pu_name][self.time_hrz_len-pvlt:2*self.time_hrz_len-pvlt]
        return res
    
    def get_next_IP(self):
        res = 0
        unit_table = self.get_unit_table()
        for nu_name in self.dwns:
            nu = unit_table[nu_name]
            next_IP = nu.inv_pos
            res += next_IP
        return res
    
    def get_constrs(self):
        return self.solver.get_constrs()
    
    def append_constrs(self, constrs_new):
        self.solver.append_constrs(constrs_new)
    

        
if __name__ == '__main__':
    time_hrz_len = 20
    env_config = [
            {
                "name":"U11", "vlt":3, "price":300, "I0":10000000, "T0":0, "h":1, "is_supp":True,
                "is_store":False, "ups":[], "dwns":["U21","U22"]
            },
            {
                "name":"U21", "vlt":2, "price":0, "I0":2000, "T0":0, "h":1, "is_supp":False,
                "is_store":False, "ups":["U11"], "dwns":["U31","U32"]
            },
            {
                "name":"U22", "vlt":2, "price":0, "I0":2000, "T0":0, "h":1, "is_supp":False,
                "is_store":False, "ups":["U11"], "dwns":["U33","U34"]
            },
            {
                "name":"U31", "vlt":1, "price":350, "I0":1000, "T0":0, "h":1, "is_supp":False,
                "is_store":True, "ups":["U21"], "dwns":[], "k":200, "D": Config.demands_generator(time_hrz_len),
                "U0": 0
            },
            {
                "name":"U32", "vlt":1, "price":340, "I0":1000, "T0":0, "h":1, "is_supp":False,
                "is_store":True, "ups":["U21"], "dwns":[], "k":200, "D": Config.demands_generator(time_hrz_len),
                "U0": 0
            },
            {
                "name":"U33", "vlt":1, "price":350, "I0":1000, "T0":0, "h":1, "is_supp":False,
                "is_store":True, "ups":["U22"], "dwns":[], "k":200, "D": Config.demands_generator(time_hrz_len),
                "U0": 0
            },
            {
                "name":"U34", "vlt":1, "price":360, "I0":1000, "T0":0, "h":1, "is_supp":False,
                "is_store":True, "ups":["U22"], "dwns":[], "k":200, "D": Config.demands_generator(time_hrz_len),
                "U0": 0
            }]
    solver = Solver(Config(env_config, time_hrz_len))
    solver.constrs_fml()
    solver.solve(verbose=True)
    # solver.show_var("I", hrz_only=False)
    # solver.show_var("R")
    # solver.show_var("U")
    # solver.show_var("D")
    # solver.show_var("inv_pos")
    # solver.show_var("z")
    # print(solver.unit_table['U34'].z.value)
    