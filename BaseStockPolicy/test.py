import cvxpy as cp
# from env.inventory_utils import Utils
# from config.inventory_config import basestock_policy_config_consumer
import random as rnd
import numpy as np
import matplotlib.pyplot as plt

class Utils:
    @staticmethod
    def get_consumer_action_space():
        action_space = [0,1,2,3,4,5,7,9,12]
        return action_space

class Config:
    '''
    Configuration that feeds the solver
    '''
    def __init__(self):
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
        self.time_hrz_len = 28
        self.dcs_step = 5
        self.temporal_len = 100
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
        self.config = [
            {
                "name":"U11", "vlt":3, "price":300, "I0":1000000000, "T0":0, "h":1, "is_supp":True,
                "is_store":False, "ups":[], "dwns":["U21","U22"]
            },
            {
                "name":"U21", "vlt":2, "price":0, "I0":0, "T0":0, "h":1, "is_supp":False,
                "is_store":False, "ups":["U11"], "dwns":["U31","U32"]
            },
            {
                "name":"U22", "vlt":2, "price":0, "I0":0, "T0":0, "h":1, "is_supp":False,
                "is_store":False, "ups":["U11"], "dwns":["U33","U34"]
            },
            {
                "name":"U31", "vlt":1, "price":350, "I0":0, "T0":0, "h":1, "is_supp":False,
                "is_store":True, "ups":["U21"], "dwns":[], "k":200, "D":self.demands_generator(),
                "U0": 0
            },
            {
                "name":"U32", "vlt":1, "price":340, "I0":0, "T0":0, "h":1, "is_supp":False,
                "is_store":True, "ups":["U21"], "dwns":[], "k":200, "D":self.demands_generator(),
                "U0": 0
            },
            {
                "name":"U33", "vlt":1, "price":350, "I0":0, "T0":0, "h":1, "is_supp":False,
                "is_store":True, "ups":["U22"], "dwns":[], "k":200, "D":self.demands_generator(),
                "U0": 0
            },
            {
                "name":"U34", "vlt":1, "price":360, "I0":0, "T0":0, "h":1, "is_supp":False,
                "is_store":True, "ups":["U22"], "dwns":[], "k":200, "D":self.demands_generator(),
                "U0": 0
            }
        ]
        self.config = [
            {'name': 'SKUSupplierUnit_2', 'price': 100, 'vlt': 5, 'T0': 0, 
            'I0': 1000, 'U0': 0, 'h': 1, 
            'unit_transport_cost': 2, 'order_cost': 200, 'sale_mean': 33.26069036579083, 
            'is_store': False, 'is_supp': True, 'ups': [], 'dwns': ['SKUWarehouseUnit_10']},
             {'name': 'SKUWarehouseUnit_10', 'price': 100, 'vlt': 2, 'T0': 0, 'I0': 1000, 
             'U0': 0, 'h': 1, 
             'unit_transport_cost': 1, 'order_cost': 200, 'sale_mean': 33.26069036579083, 
             'is_store': False, 'is_supp': False, 
             'ups': ['SKUSupplierUnit_2'], 'dwns': ['SKUWarehouseUnit_18']}, 
             {'name': 'SKUWarehouseUnit_18', 'price': 100, 'vlt': 1, 'T0': 0, 
             'I0': 10000, 'U0': 0, 'h': 1, 
             'unit_transport_cost': 1, 'order_cost': 200, 'sale_mean': 33.26069036579083, 
             'is_store': False, 'is_supp': False, 
             'ups': ['SKUWarehouseUnit_10'], 
             'dwns': ['OuterSKUStoreUnit_27', 'OuterSKUStoreUnit_34']}, 
             {'name': 'OuterSKUStoreUnit_27', 'price': 220, 'vlt': 0, 'T0': 0, 
             'I0': 50, 'U0': 0, 'h': 1, 'unit_transport_cost': 0, 'order_cost': 400, 
             'sale_mean': 14.985059247810407, 
             'is_store': True, 'is_supp': False, 
             'ups': ['SKUWarehouseUnit_18'], 'dwns': [], 'k': 0.1, 
             'D': np.array([ 8,  7,  6,  5,  7,  8,  6, 11,  7,  9,  7, 13, 12,  5,  6,  8,  6, 10, 11,  7,  5,  5,  8, 14, 15, 17,  4,  9])}, {'name': 'OuterSKUStoreUnit_34', 'price': 220, 'vlt': 0, 'T0': 0, 'I0': 40, 'U0': 0, 'h': 1, 'unit_transport_cost': 0, 'order_cost': 400, 'sale_mean': 18.27563111798042, 'is_store': True, 'is_supp': False, 'ups': ['SKUWarehouseUnit_18'], 'dwns': [], 'k': 0.1, 'D': np.array([ 8, 13, 20,  9,  9, 14, 24,  8,  6, 22,  9, 16, 11, 11, 17, 13, 19, 20,  8, 20, 16, 11, 12, 14,  8, 10, 14, 13])}] 
        self.timestep = 0
    
    def get_R_init(self):
        return np.zeros(self.time_hrz_len, dtype=int)
    
    def var_sum(self, vars:list):
        sumup = 0
        for var in vars:
            sumup += var
        return sumup
        
        
    def demands_generator(self, use_rnd_seed=True):
        return np.random.randint(400,
                                700,
                                self.time_hrz_len,
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
        self.min_p = 1000000000000
        self.max_p = 0
        self.avg_p = 0
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
        price_sum = 0
        for c in self.config.config:
            self.unit_table[c['name']]=Unit(c, self)
            if(c['is_supp']):
                self.n_supps+=1
                price = c['price']
                price_sum += price
                if(price<self.min_p):
                    self.min_p=price
                if(price>self.max_p):
                    self.max_p=price
            elif(c['is_store']):
                self.n_stores+=1
        self.avg_p = price_sum//self.n_supps

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
        P_sumup = 0
        for unit_name in self.unit_table:
            unit = self.unit_table[unit_name]
            unit.constrs_fml()
            P_sumup += cp.sum(unit.P)
        P = self.P
        self.append_constrs(
            [P==P_sumup-self.get_I0_sum()*self.max_p]
        )
    
    def get_I0_sum(self):
        res = 0
        for unit in self.unit_table.values():
            if not unit.is_supp:
                res += unit.I[0]
        return res

    def obj_setup(self):
        return cp.Maximize(self.P)
    
    def solve(self):
        obj = self.obj_setup()
        prob = cp.Problem(obj, self.get_constrs())
        prob.solve()
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
        self.I0 = config['I0']
        self.T0 = config['T0']
        self.h = config['h']
        self.is_supp = config['is_supp']
        self.is_store = config['is_store']
        self.ups = config['ups']
        self.dwns = config['dwns']
        self.time_hrz_len = solver.time_hrz_len
        self.pn_units = len(self.ups)
        if(self.is_store):
            self.k = config['k']
            self.U0 = config['U0']
            self.D = config['D']
            self.U = cp.Variable(self.time_hrz_len+1,integer=True)
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
        self.P = cp.Variable(self.time_hrz_len, integer=True)

        # constrs
        self.append_constrs([
            self.I>=0, 
            self.T>=0,
            self.S>=0
        ])
        if(self.is_store):
            self.append_constrs(
                [self.U>=0]
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
        if(self.is_store):
            U = self.U
            D = self.D
            k = self.k
        
        self.append_constrs(
            [
                I[0]==self.I0,
                T[0]==self.T0,
                I[1:self.time_hrz_len+1]==I[:self.time_hrz_len]+buy_arv-S,
                T[1:self.time_hrz_len+1]==T[:self.time_hrz_len]-buy_arv+buy_in,
                S<=I[:self.time_hrz_len],
                (S<=D)if self.is_store else (S==self.get_sale_out()),
                buy_in == self.get_buy_in(),
                buy_arv == self.get_buy_arv(),
                inv_pos == I[:self.time_hrz_len]+T[:self.time_hrz_len],
                z==inv_pos+buy_in
            ]
        )
        if(self.is_store):
            self.append_constrs(
                [
                    U[0]==self.U0,
                    U[1:]==D-S,
                    # inv_pos==I[1:]+T[1:]-U[1:],
                    P==price*S-k*U[1:]-h*I[1:]
                ]
            )
        elif(self.is_supp):
            self.append_constrs(
                [
                    # inv_pos==I[1:]+T[1:]+self.get_next_IP(),
                    P==-price*S #we don't count the holding cost here
                ]
            )
        else:
            self.append_constrs(
                [
                    # inv_pos==I[1:]+T[1:]+self.get_next_IP(),
                    P==-h*I[:self.time_hrz_len]
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
    for _ in range(6):
        config = Config()
        solver = Solver(config)
        solver.constrs_fml()
        solver.solve()
    solver.show_var("I", hrz_only=False)
    solver.show_var("R")
    solver.show_var("U")
    solver.show_var("D")
    solver.show_var("inv_pos")
    solver.show_var('z')