# Configuration ===============================================================================

# 配置仿真环境

# How to simulate demand, ['GAMMA', 'DYNAMIC_GAMMA', 'ONLINE']
# "GAMMA": generate demands according to predefined paramter, store them and apply them to all schedulers
# "DYNAMIC_GAMMA": Generate demands on-the-fly for all schedulers, which means different rounds may see different demands;
# "ONLINE": Read demand from files whose distribution is unknown.
demand_sampler = 'DYNAMIC_GAMMA'

#SKU CONFIG
sku_config = {
    "sku_num": 5,
    "sku_names": ['SKU%i' % i for i in range(1, 6)]
}

# SUPPLIER_CONFIG
supplier_config = {
    "name": 'SUPPLIER',
    "short_name": "M",
    "supplier_num": 2,
    "fleet_size": [200, 200],
    "unit_storage_cost": [1, 1],
    "unit_transport_cost": [2, 2],
    "storage_capacity": [20000, 20000],
    "order_cost": [200, 200],
    "delay_order_penalty": [1000, 1000],
    "downstream_facilities": [[0], [1]], 
    "sku_relation": [
        [{"sku_name": "SKU1", "price": 100, "cost": 50, "service_level": 0.95, "vlt": 5, "penalty": 1000, 'init_stock': 1000, "production_rate": 200},
         {"sku_name": "SKU2", "price": 200, "cost": 100, "service_level": 0.90, "vlt": 7, "penalty": 1000, 'init_stock': 1000, "production_rate": 200},
         {"sku_name": "SKU3", "price": 300, "cost": 150, "service_level": 0.90, "vlt": 5, "penalty": 1000, 'init_stock': 1000, "production_rate": 200}
        ],
        [{"sku_name": "SKU4", "price": 400, "cost": 250, "service_level": 0.95, "vlt": 4, "penalty": 1000, 'init_stock': 1000, "production_rate": 200},
         {"sku_name": "SKU5", "price": 250, "cost": 150, "service_level": 0.95, "vlt": 5, "penalty": 1000, 'init_stock': 1000, "production_rate": 200}
        ]
    ]
}

# WAREHOUSE_CONFIG
# The length of warehouse_config corresponds to number of intermedia echelons
warehouse_config = [
    {
        "name": "WAREHOUSE",
        "short_name": "R",
        "warehouse_num": 2,
        "fleet_size": [200, 200],
        "unit_storage_cost": [1, 1],
        "unit_transport_cost": [1, 1],
        "storage_capacity": [10000, 10000],
        "order_cost": [200, 200],
        "delay_order_penalty": [1000, 1000], 
        "downstream_facilities": [[0, 1, 2], [1, 2]],
        "sku_relation": [
            [{"sku_name": "SKU1", "price": 100, "cost": 100, "service_level": 0.95, "vlt": 2, 'init_stock': 500},
             {"sku_name": "SKU2", "price": 200, "cost": 200, "service_level": 0.95, "vlt": 2, 'init_stock': 200},
             {"sku_name": "SKU3", "price": 300, "cost": 300, "service_level": 0.95, "vlt": 2, 'init_stock': 100}
            ],
            [{"sku_name": "SKU4", "price": 400, "cost": 400, "service_level": 0.95, "vlt": 1, 'init_stock': 200},
             {"sku_name": "SKU5", "price": 250, "cost": 250, "service_level": 0.95, "vlt": 3, 'init_stock': 200}
            ]
        ]
    },
    {
        "name": "WAREHOUSE",
        "short_name": "F",
        "warehouse_num": 3,
        "fleet_size": [200, 200, 200],
        "unit_storage_cost": [1, 1, 1],
        "unit_transport_cost": [1, 1, 1],
        "storage_capacity": [5000, 5000, 5000],
        "order_cost": [200, 200, 200],
        "delay_order_penalty": [1000, 1000, 1000], 
        "downstream_facilities": [[0], [0, 1], [1]],
        "sku_relation": [
            [{"sku_name": "SKU1", "price": 100, "cost": 100, "service_level": 0.95, "vlt": 2, 'init_stock': 500},
             {"sku_name": "SKU2", "price": 200, "cost": 200, "service_level": 0.95, "vlt": 2, 'init_stock': 200},
             {"sku_name": "SKU3", "price": 300, "cost": 300, "service_level": 0.95, "vlt": 2, 'init_stock': 100}
            ],
            [{"sku_name": "SKU2", "price": 200, "cost": 200, "service_level": 0.95, "vlt": 1, 'init_stock': 100},
             {"sku_name": "SKU3", "price": 300, "cost": 300, "service_level": 0.95, "vlt": 2, 'init_stock': 150},
             {"sku_name": "SKU4", "price": 400, "cost": 400, "service_level": 0.95, "vlt": 1, 'init_stock': 100}
            ],
            [{"sku_name": "SKU3", "price": 300, "cost": 300, "service_level": 0.95, "vlt": 2, 'init_stock': 200},
             {"sku_name": "SKU4", "price": 400, "cost": 400, "service_level": 0.95, "vlt": 1, 'init_stock': 200},
             {"sku_name": "SKU5", "price": 250, "cost": 250, "service_level": 0.95, "vlt": 2, 'init_stock': 200}
            ]
        ]
    }
]

# store_CONFIG
store_config = {
    "name": "STORE",
    "short_name": "S",
    "store_num": 2,
    "storage_capacity": [3000, 3000],
    "unit_storage_cost": [1, 1],
    "order_cost": [400, 400],
    "sku_relation": [
        [{"sku_name": "SKU1", "price": 220, "service_level": 0.95, "cost": 100, "sale_gamma": 50, 'init_stock': 500, 'max_stock': 400, 'backlog_ratio': 0.1},
         {"sku_name": "SKU2", "price": 350, "service_level": 0.95, "cost": 250, "sale_gamma": 20, 'init_stock': 200, 'max_stock': 400, 'backlog_ratio': 0.1},
         {"sku_name": "SKU3", "price": 430, "service_level": 0.95, "cost": 300, "sale_gamma": 8, 'init_stock': 80, 'max_stock': 400, 'backlog_ratio': 0.1},
         {"sku_name": "SKU4", "price": 560, "service_level": 0.95, "cost": 450, "sale_gamma": 9, 'init_stock': 90, 'max_stock': 400, 'backlog_ratio': 0.1}
        ],
        [{"sku_name": "SKU2", "price": 350, "service_level": 0.95, "cost": 250, "sale_gamma": 15, 'init_stock': 150, 'max_stock': 400, 'backlog_ratio': 0.1},
         {"sku_name": "SKU3", "price": 430, "service_level": 0.95, "cost": 300, "sale_gamma": 10, 'init_stock': 100, 'max_stock': 400, 'backlog_ratio': 0.1},
         {"sku_name": "SKU4", "price": 560, "service_level": 0.95, "cost": 450, "sale_gamma": 15, 'init_stock': 150, 'max_stock': 400, 'backlog_ratio': 0.1},
         {"sku_name": "SKU5", "price": 370, "service_level": 0.95, "cost": 200, "sale_gamma": 10, 'init_stock': 100, 'max_stock': 400, 'backlog_ratio': 0.1}
        ]
    ]
}

# CONSUMER_NETWORK_CONFIG

env_config = {
    'global_reward_weight_producer': 0.50,
    'global_reward_weight_consumer': 0.50,
    'downsampling_rate': 1,
    'episod_duration': 60,
    "initial_balance": 100000,
    "consumption_hist_len": 14,
    "sale_hist_len": 14,
    "total_echelons": 4,
    "reward_normalization": 1e6
}
