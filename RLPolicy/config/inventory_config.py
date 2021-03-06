# Configuration ===============================================================================

# 配置仿真环境

# How to simulate demand, ['GAMMA', 'DYNAMIC_GAMMA', 'ONLINE']
# "GAMMA": generate demands according to predefined paramter, store them and apply them to all schedulers
# "DYNAMIC_GAMMA": Generate demands on-the-fly for all schedulers, which means different rounds may see different demands;
# "ONLINE": Read demand from files whose distribution is unknown.
# demand_sampler = 'DYNAMIC_GAMMA'
demand_sampler = 'ONLINE'

#SKU CONFIG
sku_config = {
    "sku_num": 6,
    "sku_names": ['SKU%i' % i for i in range(1, 7)]
}

# SUPPLIER_CONFIG
supplier_config = {
    "name": 'SUPPLIER',
    "short_name": "M",
    "supplier_num": 2,
    "fleet_size": [50, 50],
    "unit_storage_cost": [1, 1],
    "unit_transport_cost": [2, 2],
    "storage_capacity": [30000, 30000],
    "order_cost": [200, 200],
    "delay_order_penalty": [1000, 1000],
    "downstream_facilities": [[0], [1]], 
    "sku_relation": [
        [{"sku_name": "SKU1", "price": 100, "cost": 50, "service_level": 0.95, "vlt": 5, "penalty": 1000, 'init_stock': 1000, "production_rate": 200},
         {"sku_name": "SKU2", "price": 200, "cost": 100, "service_level": 0.90, "vlt": 7, "penalty": 1000, 'init_stock': 1000, "production_rate": 200},
         {"sku_name": "SKU3", "price": 300, "cost": 150, "service_level": 0.90, "vlt": 5, "penalty": 1000, 'init_stock': 1000, "production_rate": 200}
        ],
        [{"sku_name": "SKU4", "price": 400, "cost": 250, "service_level": 0.95, "vlt": 8, "penalty": 1000, 'init_stock': 1000, "production_rate": 200},
         {"sku_name": "SKU5", "price": 250, "cost": 150, "service_level": 0.95, "vlt": 5, "penalty": 1000, 'init_stock': 1000, "production_rate": 200},
         {"sku_name": "SKU6", "price": 500, "cost": 300, "service_level": 0.95, "vlt": 5, "penalty": 1000, 'init_stock': 1000, "production_rate": 200}
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
        "fleet_size": [50, 50],
        "unit_storage_cost": [1, 1],
        "unit_transport_cost": [1, 1],
        "storage_capacity": [20000, 20000],
        "order_cost": [200, 200],
        "delay_order_penalty": [1000, 1000], 
        "downstream_facilities": [[0, 1], [1, 2]],
        "sku_relation": [
            [{"sku_name": "SKU1", "price": 100, "cost": 100, "service_level": 0.95, "vlt": 2, 'init_stock': 1000},
             {"sku_name": "SKU2", "price": 200, "cost": 200, "service_level": 0.95, "vlt": 2, 'init_stock': 1000},
             {"sku_name": "SKU3", "price": 300, "cost": 300, "service_level": 0.95, "vlt": 2, 'init_stock': 1000}
            ],
            [{"sku_name": "SKU4", "price": 400, "cost": 400, "service_level": 0.95, "vlt": 1, 'init_stock': 1000},
             {"sku_name": "SKU5", "price": 250, "cost": 250, "service_level": 0.95, "vlt": 3, 'init_stock': 1000},
             {"sku_name": "SKU6", "price": 500, "cost": 300, "service_level": 0.95, "vlt": 3, 'init_stock': 1000}
            ]
        ]
    },
    {
        "name": "WAREHOUSE",
        "short_name": "F",
        "warehouse_num": 3,
        "fleet_size": [50, 50, 50],
        "unit_storage_cost": [1, 1, 1],
        "unit_transport_cost": [1, 1, 1],
        "storage_capacity": [100000, 100000, 100000],
        "order_cost": [200, 200, 200],
        "delay_order_penalty": [1000, 1000, 1000], 
        "downstream_facilities": [[0,1], [0, 1], [0,1]],
        "sku_relation": [
            [{"sku_name": "SKU1", "price": 100, "cost": 100, "service_level": 0.95, "vlt": 1, 'init_stock': 10000},
             {"sku_name": "SKU2", "price": 200, "cost": 200, "service_level": 0.95, "vlt": 1, 'init_stock': 10000}
            ],
            [{"sku_name": "SKU3", "price": 300, "cost": 300, "service_level": 0.95, "vlt": 2, 'init_stock': 10000},
             {"sku_name": "SKU4", "price": 400, "cost": 400, "service_level": 0.95, "vlt": 2, 'init_stock': 10000}
            ],
            [{"sku_name": "SKU5", "price": 250, "cost": 250, "service_level": 0.95, "vlt": 3, 'init_stock': 10000},
             {"sku_name": "SKU6", "price": 500, "cost": 300, "service_level": 0.95, "vlt": 3, 'init_stock': 10000}
            ]
        ]
    }
]

# store_CONFIG
store_config = {
    "name": "STORE",
    "short_name": "S",
    "store_num": 2,
    "storage_capacity": [100, 100],
    "unit_storage_cost": [10, 10],
    "order_cost": [0, 0],
    "sku_relation": [
        [{"sku_name": "SKU1", "price": 220, "service_level": 0.95, "cost": 100, "sale_gamma": 50, 'init_stock': 50, 'max_stock': 400},
         {"sku_name": "SKU2", "price": 350, "service_level": 0.95, "cost": 250, "sale_gamma": 20, 'init_stock': 20, 'max_stock': 400},
         {"sku_name": "SKU3", "price": 430, "service_level": 0.95, "cost": 300, "sale_gamma": 8, 'init_stock': 80, 'max_stock': 400},
         {"sku_name": "SKU4", "price": 560, "service_level": 0.95, "cost": 450, "sale_gamma": 9, 'init_stock': 90, 'max_stock': 400},
         {"sku_name": "SKU5", "price": 370, "service_level": 0.95, "cost": 200, "sale_gamma": 10, 'init_stock': 10, 'max_stock': 400},
         {"sku_name": "SKU6", "price": 600, "service_level": 0.95, "cost": 500, "sale_gamma": 20, 'init_stock': 20, 'max_stock': 400}
        ],
        [{"sku_name": "SKU1", "price": 220, "service_level": 0.95, "cost": 100, "sale_gamma": 40, 'init_stock': 40, 'max_stock': 400},
         {"sku_name": "SKU2", "price": 350, "service_level": 0.95, "cost": 250, "sale_gamma": 10, 'init_stock': 10, 'max_stock': 400},
         {"sku_name": "SKU3", "price": 430, "service_level": 0.95, "cost": 300, "sale_gamma": 15, 'init_stock': 15, 'max_stock': 400},
         {"sku_name": "SKU4", "price": 560, "service_level": 0.95, "cost": 450, "sale_gamma": 10, 'init_stock': 10, 'max_stock': 400},
         {"sku_name": "SKU5", "price": 370, "service_level": 0.95, "cost": 200, "sale_gamma": 10, 'init_stock': 10, 'max_stock': 400},
         {"sku_name": "SKU6", "price": 600, "service_level": 0.95, "cost": 500, "sale_gamma": 30, 'init_stock': 30, 'max_stock': 400}
        ]
    ]
}

# store_config = {
#     "name": "STORE",
#     "short_name": "S",
#     "store_num": 2,
#     "storage_capacity": [6000, 6000],
#     "unit_storage_cost": [1, 1],
#     "order_cost": [400, 400],
#     "sku_relation": [
#         [{"sku_name": "SKU1", "price": 220, "service_level": 0.95, "cost": 100, "sale_gamma": 50, 'init_stock': 300, 'max_stock': 400},
#          {"sku_name": "SKU2", "price": 350, "service_level": 0.95, "cost": 250, "sale_gamma": 20, 'init_stock': 150, 'max_stock': 400},
#          {"sku_name": "SKU3", "price": 430, "service_level": 0.95, "cost": 300, "sale_gamma": 8, 'init_stock': 160, 'max_stock': 400},
#          {"sku_name": "SKU4", "price": 560, "service_level": 0.95, "cost": 450, "sale_gamma": 9, 'init_stock': 100, 'max_stock': 400},
#          {"sku_name": "SKU5", "price": 370, "service_level": 0.95, "cost": 200, "sale_gamma": 10, 'init_stock': 150, 'max_stock': 400},
#          {"sku_name": "SKU6", "price": 600, "service_level": 0.95, "cost": 500, "sale_gamma": 20, 'init_stock': 150, 'max_stock': 400}
#         ],
#         [{"sku_name": "SKU1", "price": 220, "service_level": 0.95, "cost": 100, "sale_gamma": 40, 'init_stock': 200, 'max_stock': 400},
#          {"sku_name": "SKU2", "price": 350, "service_level": 0.95, "cost": 250, "sale_gamma": 10, 'init_stock': 150, 'max_stock': 400},
#          {"sku_name": "SKU3", "price": 430, "service_level": 0.95, "cost": 300, "sale_gamma": 15, 'init_stock': 200, 'max_stock': 400},
#          {"sku_name": "SKU4", "price": 560, "service_level": 0.95, "cost": 450, "sale_gamma": 10, 'init_stock': 150, 'max_stock': 400},
#          {"sku_name": "SKU5", "price": 370, "service_level": 0.95, "cost": 200, "sale_gamma": 10, 'init_stock': 150, 'max_stock': 400},
#          {"sku_name": "SKU6", "price": 600, "service_level": 0.95, "cost": 500, "sale_gamma": 30, 'init_stock': 200, 'max_stock': 400}
#         ]
#     ]
# }

# CONSUMER_NETWORK_CONFIG

env_config = {
    'global_reward_weight_producer': 0.50,
    'global_reward_weight_consumer': 0.50,
    'downsampling_rate': 1,
    'episod_duration': 365*4, #365*4
    'evaluation_len': 365, #365
    "initial_balance": 100000,
    "consumption_hist_len": 21,
    "sale_hist_len": 21,
    "demand_prediction_len": 21,
    "uncontrollable_part_state_len": 21*7, #31
    "uncontrollable_part_pred_len": 6, # 3天具体，3天均值，7天均值，21天均值
    "total_echelons": 4,
    "init": None, # or `rnd`, None 'rst'
    "tail_timesteps": 0,
    "training": True
}
