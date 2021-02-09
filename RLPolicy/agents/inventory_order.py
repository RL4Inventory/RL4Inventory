from dataclasses import dataclass
from agents.inventory import SellerUnit, BalanceSheet, SKUStoreUnit, ConsumerUnit
from ray.rllib.utils.annotations import override

    
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
            
    def __init__(self, facility, config, economy, sale_sampler):
        super(OuterSellerUnit, self).__init__(facility, config, economy)
        self.facility = facility
        self.economy = economy
        self.config = config
        self.sale_sampler = sale_sampler
        # self.sale_hist = []
        self.step = 0
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
        today_date_info = self.sale_sampler.get_date_info(self.facility.bom.output_product_id, 0)
        self.info.update(today_date_info)
        self.date_info_hist = [today_date_info] * 21

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
        today_date_info = self.sale_sampler.get_date_info(self.facility.bom.output_product_id, self.step)
        self.info.update(today_date_info)
        self.date_info_hist.append(today_date_info)
        self.date_info_hist = self.date_info_hist[1:]


    def set_step(self, t):
        self.step = t
        # TODO if t != 0 set the init info
        if t != 0:
            product_id = self.facility.bom.output_product_id
            for i in range(len(self.sale_hist)):
                demand, sale_price = self.economy.market_demand(self.sale_sampler, product_id,
                                                                self.step - len(self.pred_sale) + i)
                self.sale_hist[i] = demand
                self.backlog_demand_hist[i] = 0

                self.date_info_hist[i] = self.sale_sampler.get_date_info(self.facility.bom.output_product_id,
                                                                         self.step - len(self.pred_sale) + i)
            self.info.update({
                "demand_hist": self.sale_hist,
                "demand_pred": self.pred_sale,
            })
            self.info.update(self.date_info_hist[-1])

            # sale history



    def reset(self):
        self.economy.total_units_sold = 0
        self.total_backlog_demand = 0
        self.step = 0
        self.backlog_demand_hist = [0] * len(self.backlog_demand_hist)
        self.sale_hist = [0] * len(self.sale_hist)
        self.pred_sale = [0] * len(self.pred_sale)
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
        today_date_info = self.sale_sampler.get_date_info(self.facility.bom.output_product_id, 0)
        self.info.update(today_date_info)
        self.date_info_hist = [today_date_info] * 21

    @override(SellerUnit)
    def sale_mean(self):
        product_id = self.facility.bom.output_product_id
        return self.sale_sampler.get_sale_mean(product_id)

    def act(self, control):
        # if control is not None:
        # self.economy.unit_price = self.facility.sku_info['price']   # update the current unit price
        # print(self.economy.unit_price, self.economy.price_demand_intercept, self.economy.price_demand_slope)
        product_id = self.facility.bom.output_product_id
        self.step = (self.step + 1) % self.sale_sampler.total_span
        demand, sale_price = self.economy.market_demand(self.sale_sampler, product_id, self.step)
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