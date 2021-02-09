from dataclasses import dataclass
from agents.inventory import SellerUnit, BalanceSheet, SKUStoreUnit, ConsumerUnit
from ray.rllib.utils.annotations import override
from env.inventory_utils import Utils

    
class OuterSellerUnit(SellerUnit):
    @dataclass
    class Economy:
        unit_price: int = 0
        total_units_sold: int = 0
        latest_sale: int = 0
            
        def market_demand(self, sale_sampler, product_id, step):
            return sale_sampler.sample_sale_and_price(product_id, step)
        
        def profit(self, units_sold, unit_price):
            return units_sold * unit_price
        
        def step_balance_sheet(self, units_sold, unit_price, out_stock_demand, backlog_ratio):
            # （销售利润，缺货损失）
            # return BalanceSheet(self.profit(units_sold, unit_price), 0)# -0.5*self.profit(demand, unit_price))
            balance = BalanceSheet(self.profit(units_sold, unit_price),
                                    -self.profit(out_stock_demand, unit_price)*backlog_ratio)
            return balance
        
    @dataclass
    class Config:
        sale_gamma: int 
    
    @dataclass
    class Control:
        unit_price: int
    
    def get_future_sales(self, pred_len):
        sale_pred = []
        product_id = self.facility.bom.output_product_id
        for i in range(1+self.step, 1+pred_len+self.step):
            demand, _ = self.economy.market_demand(self.sale_sampler, product_id, i)
            sale_pred.append(demand)
        return sale_pred

    def _init_sale_pred(self):
        pred_len = Utils.get_env_config()['sale_oracle_len']
        return self.get_future_sales(pred_len)
            
    def __init__(self, facility, config, economy, sale_sampler):
        super(OuterSellerUnit, self).__init__(facility, config, economy)
        self.facility = facility
        self.economy = economy
        self.config = config
        self.sale_sampler = sale_sampler
        # self.sale_hist = []
        self.step = 0
        self.sale_pred = self._init_sale_pred()
        self.info = {
            "demand_hist": self.sale_hist,
            "demand_pred": self.pred_sale,
            "demand_mean": self.sale_mean(),
            "sku_num": 0,
            "isoweekday": 5,
            "year": 2012,
            "month": 1,
            "day": 28,
            "dayofyear": 28,
            "isweekend": 1,
        }
        self.info.update(self.sale_sampler.get_date_info(self.facility.bom.output_product_id, 0))

    def _update_backlog_demand_future(self):
        product_id = self.facility.bom.output_product_id
        demand, sale_price = self.economy.market_demand(self.sale_sampler, product_id, self.step + len(self.pred_sale))
        self.pred_sale.append(demand)
        self.pred_sale = self.pred_sale[1:]

    def _update_info(self):

        self.info.update({
            "demand_hist": self.sale_hist,
            "demand_pred": self.pred_sale,
        })
        self.info.update(self.sale_sampler.get_date_info(self.facility.bom.output_product_id, self.step))

    def set_step(self, t):
        self.step = t
        # TODO if t != 0 set the init info

    def reset(self):
        self.economy.total_units_sold = 0
        self.total_backlog_demand = 0
        self.step = 0
        self.info.update({
            "demand_hist": self.sale_hist,
            "demand_pred": self.pred_sale,
            "demand_mean": self.sale_mean(),
            "sku_num": 0,
            "isoweekday": 5,
            "year": 2012,
            "month": 1,
            "day": 28,
            "dayofyear": 28,
            "isweekend": 1,
        })

    @override(SellerUnit)
    def sale_mean(self):
        product_id = self.facility.bom.output_product_id
        return self.sale_sampler.get_sale_mean(product_id)
    
    def get_future_demand(self, product_id):
        f_step = (self.step+Utils.get_env_config()['sale_hist_len'])%self.sale_sampler.total_span
        f_demand, _ = self.economy.market_demand(self.sale_sampler, product_id, f_step)
        return f_demand


    def act(self, control):
        # if control is not None:
        # self.economy.unit_price = self.facility.sku_info['price']   # update the current unit price
        # print(self.economy.unit_price, self.economy.price_demand_intercept, self.economy.price_demand_slope)
        product_id = self.facility.bom.output_product_id
        self.step = (self.step + 1) % self.sale_sampler.total_span
        demand, sale_price = self.economy.market_demand(self.sale_sampler, product_id, self.step)
        
        ### This is new ###
        f_demand = self.get_future_demand(product_id)
        self._update_sale_pred(f_demand)
        
        self.economy.latest_sale = demand
        self.economy.unit_price = sale_price
        self._update_sale_hist(demand)
        sold_qty = self.facility.storage.take_available(product_id, demand)
        self.economy.total_units_sold += sold_qty
        # in_transit_orders = sum(self.facility.consumer.open_orders.values(), Counter())
        # out_stock_demand = max(0, demand - sold_qty - in_transit_orders[product_id])
        out_stock_demand = max(0, demand - sold_qty)
        self._update_backlog_demand_hist(out_stock_demand)
        self._update_backlog_demand_future()
        self._update_info()
        # balance = self.economy.step_balance_sheet( sold_qty, self.economy.unit_price, out_stock_demand )
        backlog_ratio = self.facility.sku_info.get('backlog_ratio', 0.1)
        balance = self.economy.step_balance_sheet( sold_qty, self.economy.unit_price, 0, backlog_ratio)
        return balance, 0


class OuterSKUStoreUnit(SKUStoreUnit):
    def __init__(self, facility, config, economy_config, sale_sampler):
        super(OuterSKUStoreUnit, self).__init__(facility, config, economy_config)
        self.consumer = ConsumerUnit(self, config.sources, ConsumerUnit.Economy(economy_config.order_cost))
        self.seller = OuterSellerUnit(self, OuterSellerUnit.Config(sale_gamma=config.sale_gamma), OuterSellerUnit.Economy(), sale_sampler)