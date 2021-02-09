from data_adapter.base_sale_adapter import BaseSaleAdapter


def load_sale_sampler(loc_path):
    config = dict()
    config['dt_col'] = 'DT'
    config['dt_format'] = '%Y-%m-%d'
    config['sale_col'] = 'Sales'
    config['id_col'] = 'SKU'
    config['sale_price_col'] = 'Price'
    config['sale_mean_col'] = 'SaleMean'
    config['file_loc'] = loc_path
    sale_sampler = BaseSaleAdapter(config)
    return sale_sampler