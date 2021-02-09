# Configuration ===============================================================================

# 配置仿真环境

# How to simulate demand, ['GAMMA', 'DYNAMIC_GAMMA', 'ONLINE']
# "GAMMA": generate demands according to predefined paramter, store them and apply them to all schedulers
# "DYNAMIC_GAMMA": Generate demands on-the-fly for all schedulers, which means different rounds may see different demands;
# "ONLINE": Read demand from files whose distribution is unknown.
demand_sampler = 'DYNAMIC_GAMMA'

#SKU CONFIG
sku_config = {
    "sku_num": 2,
    "sku_names": ['SKU%i' % i for i in range(1, 3)]
}

# SUPPLIER_CONFIG
supplier_config = {
    "name": 'SUPPLIER',
    "short_name": "M",
    "supplier_num": 1,
    "fleet_size": [200],
    "unit_storage_cost": [1],
    "unit_transport_cost": [1],
    "storage_capacity": [10000],
    "order_cost": [200],
    "delay_order_penalty": [1000],
    "downstream_facilities": [[0]], 
    "sku_relation": [
        [{"sku_name": "SKU1", "price": 300, "cost": 200, "service_level": .95, "vlt": 5, 'init_stock': 1000, "production_rate": 200},
         {"sku_name": "SKU2", "price": 400, "cost": 300, "service_level": .95, "vlt": 5, 'init_stock': 1000, "production_rate": 200}]
    ]
}

# WAREHOUSE_CONFIG
# The length of warehouse_config corresponds to number of intermedia echelons
warehouse_config = [
    {
        "name": "WAREHOUSE",
        "short_name": "R",
        "warehouse_num": 1,
        "fleet_size": [200],
        "unit_storage_cost": [5],
        "unit_transport_cost": [1],
        "storage_capacity": [6000],
        "order_cost": [500],
        "delay_order_penalty": [1000], 
        "downstream_facilities": [[0]],
        "sku_relation": [
            [{"sku_name": "SKU1", "service_level": .96, "vlt": 2, "price": 300, "cost": 300, 'init_stock': 1000},
             {"sku_name": "SKU2", "service_level": .98, "vlt": 2, "price": 400, "cost": 400, 'init_stock': 1000}]
        ]
    }
]

# store_CONFIG
store_config = {
    "name": "STORE",
    "short_name": "S",
    "store_num": 1,
    "storage_capacity": [3000],
    "unit_storage_cost": [10],
    "order_cost": [500],
    "sku_relation": [
        [{"sku_name": "SKU1", "price": 500, "service_level": 0.95, "cost": 300, "sale_gamma": 50, 'init_stock': 200, 'max_stock': 1000},
         {"sku_name": "SKU2", "price": 700, "service_level": 0.98, "cost": 400, "sale_gamma": 40, 'init_stock': 200, 'max_stock': 1000}]
    ]
}

# CONSUMER_NETWORK_CONFIG

env_config = {
    'global_reward_weight_producer': 0.50,
    'global_reward_weight_consumer': 0.50,
    'downsampling_rate': 1,
    'episod_duration': 60,
    "initial_balance": 100000,
    "consumption_hist_len": 21,
    "sale_hist_len": 21,
    "total_echelons": 3,
    "reward_normalization": 1e6
}

