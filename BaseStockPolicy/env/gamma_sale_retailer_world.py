from datetime import datetime, timedelta
from data_adapter.gamma_sale_adapter import GammaSaleAdapter

def load_sale_sampler(store_idx):
    config = dict()
    config['dt_col'] = 'DT'
    config['dt_format'] = '%Y-%m-%d'
    config['sale_col'] = 'Sales'
    config['id_col'] = 'SKU'
    config['total_span'] = 365*2
    config['store_idx'] = store_idx
    config['sale_price_col'] = 'Price'
    config['start_dt'] = datetime.today().date()
    sale_sampler = GammaSaleAdapter(config)
    return sale_sampler