import pandas as pd
from datetime import datetime, timedelta
from env.inventory_utils import Utils


class BaseSaleAdapter(object):

    def __init__(self, config):
        self.config = config
        self.dt_col = self.config['dt_col']
        self.dt_format = self.config['dt_format']
        self.sale_col = self.config['sale_col']
        self.id_col = self.config['id_col']
        self.sale_price_col = self.config['sale_price_col']
        self.file_format = self.config.get('file_format', 'CSV')
        self.encoding = self.config.get('encoding', 'utf-8')
        self.file_loc = self.config['file_loc']
        self.sale_mean_col = self.config['sale_mean_col']
        self.sale_ts_cache = dict()
        self.sale_price_ts_cache = dict()
        self.sale_mean = dict()
        self.sale_price_mean = dict()
        self.date_cache = dict()

        self.total_span = 0
        self.cache_data()

    def _transfer_to_daily_sale(self):
        pass

    def _transfer_to_original_sale(self):
        pass

    def sample_sale_and_price(self, id_val, gap):
        # if gap>=225: print(id_val, gap, self.sale_ts_cache[id_val][gap], self.sale_price_ts_cache[id_val][gap])
        return (self.sale_ts_cache[id_val][gap], self.sale_price_ts_cache[id_val][gap])

    def get_date_info(self, id_val, gap):
        date = self.date_cache[id_val][gap]
        date_info = {
            "isoweekday": date.isoweekday(),
            "year": date.year,
            "month": date.month,
            "day": date.day,
            "dayofyear": date.dayofyear,
            "isweekend": date.isoweekday() >= 6,
        }
        return date_info

    def get_sale_mean(self, id_val):
        return self.sale_mean[id_val]

    def cache_data(self):
        self.df = self._read_df()
        self._transfer_to_daily_sale()
        # id_list = self.df[self.id_col].unique().tolist()
        id_list = Utils.get_all_skus()
        dt_min, dt_max = self.df[self.dt_col].min(), self.df[self.dt_col].max()
        self.total_span = (dt_max - dt_min).days + 1

        for id_val in id_list:
            df_tmp = self.df[self.df[self.id_col] == id_val]
            df_tmp[f"{self.dt_col}_str"] = df_tmp[self.dt_col].map(lambda x: x.strftime(self.dt_format))
            sale_cache_tmp = df_tmp.set_index(f"{self.dt_col}_str").to_dict('dict')[self.sale_col]
            sale_price_cache_tmp = df_tmp.set_index(f"{self.dt_col}_str").to_dict('dict')[self.sale_price_col]
            date_cache_tmp = df_tmp.set_index(f"{self.dt_col}_str").to_dict('dict')[self.dt_col]
            dt_tmp = dt_min
            self.sale_ts_cache[id_val] = []
            self.sale_price_ts_cache[id_val] = []
            self.date_cache[id_val] = []
            self.sale_mean[id_val] = df_tmp[self.sale_col].mean()
            sale_price_mean = df_tmp[self.sale_price_col].mean()
            while dt_tmp <= dt_max:
                dt_tmp_str = datetime.strftime(dt_tmp, self.dt_format)
                if sale_cache_tmp.get(dt_tmp_str) == None:
                    print(f"this day is lose in dataset: {dt_tmp_str}")
                    print(f"press any key to continue ...")
                    input()
                self.sale_ts_cache[id_val].append(sale_cache_tmp.get(dt_tmp_str, 0))
                self.sale_price_ts_cache[id_val].append(sale_price_cache_tmp.get(dt_tmp_str, sale_price_mean))
                self.date_cache[id_val].append(date_cache_tmp.get(dt_tmp_str))
                dt_tmp = dt_tmp + timedelta(days=1)

    def _read_df(self):
        if self.file_format == 'CSV':
            self.df = pd.read_csv(self.file_loc, encoding=self.encoding, parse_dates=[self.dt_col])
        elif self.file_format == 'EXCEL':
            self.df = pd.read_excel(self.file_loc, encoding=self.encoding, parse_dates=[self.dt_col])
        else:
            raise BaseException('Not Implemented')
        return self.df