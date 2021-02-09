import pandas as pd
from datetime import datetime, timedelta
from data_adapter.base_sale_adapter import BaseSaleAdapter


class WeeklySaleAdapter(BaseSaleAdapter):

    def __init__(self, config):
        super(WeeklySaleAdapter).__init__(self, config)
        self.distribute_method = self.config.get('distribute_method', 'AVG')
        self.period = 7
        if self.distribute_method == 'AVG':
            self.distribute_proportion = [1.0/self.period] * self.period
        else:
            self.distribute_proportion = self.config.get('distribute_proportion', [])

    def _transfer_to_daily_sale(self):
        daily_record_list = []
        for _, row in self.df.iterrows():
            for i in range(self.period):
                dt = row[self.dt_col] + timedelta(i)
                daily_record_list.append([dt, row[self.id_col],
                                          row[self.sale_price_col],
                                          row[self.sale_col]*self.distribute_proportion[i]])
        self.df = pd.DataFrame(data=daily_record_list,
                               columns=[self.dt_col, self.id_col,
                                        self.sale_price_col, self.sale_col])

    def _transfer_to_original_sale(self):
        pass
