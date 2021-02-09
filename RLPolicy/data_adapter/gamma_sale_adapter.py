import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_adapter.base_sale_adapter import BaseSaleAdapter
from env.inventory_utils import Utils


class GammaSaleAdapter(BaseSaleAdapter):

    def __init__(self, config):
        self.config = config
        self.dt_col = self.config['dt_col']
        self.dt_format = self.config['dt_format']
        self.sale_col = self.config['sale_col']
        self.id_col = self.config['id_col']
        self.sale_price_col = self.config['sale_price_col']
        self.store_idx = self.config['store_idx']
        self.start_dt = self.config['start_dt']
        self.total_span = self.config['total_span']
        self.sale_ts_cache = dict()
        self.sale_price_ts_cache = dict()
        self.cache_data()

    def _transfer_to_daily_sale(self):
        pass

    def _transfer_to_original_sale(self):
        pass

    def sample_sale_and_price(self, id_val, gap):
        return (self.sale_ts_cache[id_val][gap], self.sale_price_ts_cache[id_val][gap])

    def cache_data(self):
        self.df = self._read_df()
        id_list = self.df[self.id_col].unique().tolist()
        dt_min, dt_max = self.df[self.dt_col].min(), self.df[self.dt_col].max()
        for id_val in id_list:
            df_tmp = self.df[self.df[self.id_col] == id_val]
            df_tmp[f"{self.dt_col}_str"] = df_tmp[self.dt_col].map(lambda x: x.strftime(self.dt_format))
            sale_cache_tmp = df_tmp.set_index(f"{self.dt_col}_str").to_dict('dict')[self.sale_col]
            sale_price_cache_tmp = df_tmp.set_index(f"{self.dt_col}_str").to_dict('dict')[self.sale_price_col]
            dt_tmp = dt_min
            self.sale_ts_cache[id_val] = []
            self.sale_price_ts_cache[id_val] = []
            sale_price_mean = df_tmp[self.sale_price_col].mean()
            while dt_tmp <= dt_max:
                dt_tmp_str = datetime.strftime(dt_tmp, self.dt_format)
                self.sale_ts_cache[id_val].append(sale_cache_tmp.get(dt_tmp_str, 0))
                self.sale_price_ts_cache[id_val].append(sale_price_cache_tmp.get(dt_tmp_str, sale_price_mean))
                dt_tmp = dt_tmp + timedelta(days=1)

    def _read_df(self):
        os.makedirs('data/GammaRetail/', exist_ok=True)
        file_name = f"data/GammaRetail/store{self.store_idx+1}.csv"
        if os.path.exists(file_name):
            return pd.read_csv(file_name, parse_dates=[self.dt_col])
        sku_info_list = Utils.get_sku_of_store(self.store_idx)
        data_list = []
        for sku_info in sku_info_list:
            sale_gamma = sku_info['sale_gamma']
            sku_name = sku_info['sku_name']
            sku_price = sku_info['price']
            for i in range(self.total_span):
                demand = int(np.random.gamma(sale_gamma))
                data_list.append([sku_name, self.start_dt+timedelta(i), demand, sku_price])
        df = pd.DataFrame(data_list, columns=[self.id_col, self.dt_col, self.sale_col, self.sale_price_col])
        df.to_csv(file_name)
        return df

        