import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
import random
from env.online_retailer_world import load_sale_sampler as online_sale_sampler
from utility.replay_memory import Dataset
from utility.tensorboard import TensorBoard

class forecasting_net(nn.Module):
    def __init__(self, num_hist=21, num_pred=3):

        super(forecasting_net, self).__init__()
        self.hidden_size = 32

        self.fc1 = nn.Linear(num_hist, self.hidden_size)
        #self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, num_pred)

        self.bn1 = nn.BatchNorm1d(self.hidden_size)
        self.bn2 = nn.BatchNorm1d(self.hidden_size)


    def forward(self, x):
        x = F.sigmoid(self.bn1(self.fc1(x)))

        x = self.fc3(x)
        return x

    def embedding(self, x):
        return self.fc1(x)

class Forecasting_model():
    def __init__(self, config):
        self.config = config
        self.num_hist = config['hist_len']
        self.num_pred = config['fore_len']
        self.num_input = self.num_hist + 22
        self.num_output = self.num_pred
        self.forecast = forecasting_net(self.num_input, self.num_output)

        self.train_dataset = Dataset(200000)
        self.validation_dataset = Dataset(200000)
        self.train_len = 1600
        self.validation_len = 300

        self.optimizer = torch.optim.Adam(self.forecast.parameters(), lr=(3e-4))
        self.loss_func = nn.SmoothL1Loss()
        # self.visualization()
        self.load_data()



    def push_to_dataset(self, data_info, dataset):
        data_in, data_out = self._serialize_data(data_info)

        data_in = torch.tensor(data_in.copy(), dtype=torch.float32)
        data_out = torch.tensor(data_out.copy(), dtype=torch.float32)

        dataset.push([data_in, data_out])

    def one_hot(self, sum, k):
        tmp = [0] * sum
        tmp[k-1] = 1
        return tmp

    def _serialize_data(self, data_info):
        data_out = [x / data_info['demand_mean'] for x in data_info['demand_pred']]
        # data_out = [x  for x in data_info['demand_pred']]

        data_in = [x / data_info['demand_mean'] for x in data_info['demand_hist']]
        # data_in = [x for x in data_info['demand_hist']]

        data_in.extend(self.one_hot(12, data_info['sku_num']))
        data_in.extend(self.one_hot(7, data_info['isoweekday']))
        data_in.extend([data_info['day'], data_info['month'], data_info['isweekend']])

        return data_in, data_out

    def load_data(self, pr=False, name='test'):
        if pr:
            writer_true = TensorBoard(f"{name}_true")
            writer_pred = TensorBoard(f"{name}_pred")

        price = {
            "SKU1": 220,
            "SKU2": 350,
            "SKU3": 430,
            "SKU4": 560,
            "SKU5": 370,
            "SKU6": 600,
        }
        sku_num = {
            "SKU1": 1,
            "SKU2": 2,
            "SKU3": 3,
            "SKU4": 4,
            "SKU5": 5,
            "SKU6": 6,
        }
        price_mean = sum(price.values()) / len(price)
        for i in range(2):
            # sale_sampler = online_sale_sampler(f"data/OnlineRetail/store{i+1}.csv")
            sale_sampler = online_sale_sampler(f"data/OnlineRetail/store{i+1}_new.csv")
            for key, demand in sale_sampler.sale_ts_cache.items():
                print(f"load demand curve of {key}, the length is {len(demand)}")
                demand_mean = sum(demand) / len(demand)

                demand_hist = [demand[0]] * self.num_hist
                demand_pred = demand[0:3].copy()
                sku_price = price[key] / price_mean

                for k in range(self.train_len + self.validation_len):
                    demand_hist.append(demand[k])
                    demand_hist = demand_hist[1:]
                    demand_pred.append(demand[k + self.num_pred])
                    demand_pred = demand_pred[1:]
                    data_info = {
                        "demand_hist": demand_hist,
                        "demand_pred": demand_pred,
                        "demand_mean": demand_mean,
                        "sku_price": sku_price,
                        "sku_num": i * 6 + sku_num[key],
                    }
                    data_info.update(sale_sampler.get_date_info(key, k))

                    # if k < self.train_len and data_info['year'] >= 2012:
                    if pr:
                        self.eval()
                        if data_info['year'] == 2016 and k < 1938:
                            data_in, data_out = self._serialize_data(data_info)
                            data_in = torch.tensor(data_in.copy(), dtype=torch.float32)
                            data_pred = self.forecast(data_in.unsqueeze(0)).squeeze()
                            _pred = float (data_pred[0]) * demand_mean
                            _true = data_out[0] * demand_mean
                            writer_pred.add_scalar(f'zzeval/demand_s{i}_{key}', _pred, data_info['dayofyear'])
                            writer_true.add_scalar(f'zzeval/demand_s{i}_{key}', _true, data_info['dayofyear'])

                    else:
                        if data_info['year'] != 2016 :

                            self.push_to_dataset(data_info, self.train_dataset)
                        elif data_info['year'] == 2016 and k < 1938:
                            self.push_to_dataset(data_info, self.validation_dataset)
        print(f'train dataset size: {len(self.train_dataset)}')
        print(f'validation dataset size: {len(self.validation_dataset)}')
        # exit(0)

    def learn(self, batch_size=128):
        self.train()
        demand_hist, demand_pred = self.train_dataset.sample(batch_size)

        forecast_demand = self.forecast(demand_hist)

        loss = self.loss_func(forecast_demand, demand_pred)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def test_eval(self, batch_size=128):
        self.eval()
        with torch.no_grad():
            demand_hist, demand_pred = self.validation_dataset.sample(batch_size)
            forecast_demand = self.forecast(demand_hist)
            loss = self.loss_func(forecast_demand, demand_pred)
        return loss.item()

    def eval_one_round(self, step):
        losss = []
        for i in range(step):
            loss = self.test_eval()
            losss.append(loss)
        return losss

    def eval_all(self, name):
        self.load_data(pr=True, name=name)

    def train_one_round(self, step):
        losss = []
        for i in range(step):
            loss = self.learn()
            losss.append(loss)
        return losss


    def save_param(self, name):
        if not os.path.exists('model'):
            os.mkdir('model')
        torch.save(self.forecast.state_dict(), f'model/{name}.pkl')

    def load_param(self, name):
        self.forecast.load_state_dict(torch.load(f'model/{name}.pkl'))

    def train(self):
        self.forecast.train()

    def eval(self):
        self.forecast.eval()

    def visualization(self):
        print('  start visualization ...')
        year = [2012, 2013, 2014, 2015]

        writer = [TensorBoard(f'train_log/zDemand/Demand_year{i}') for i in year]

        for i in range(2):
            sale_sampler = online_sale_sampler(f"data/OnlineRetail/store{i+1}_new.csv")
            for key, demand in sale_sampler.sale_ts_cache.items():
                for k in range(len(demand)):
                    date_info = sale_sampler.get_date_info(key, k)
                    if date_info['year'] >= 2012 and date_info['year'] <= 2015:
                        index = date_info['year'] - 2012
                        writer[index].add_scalar(f'zzz/demand_s{i}_{key}', demand[k], date_info['dayofyear'])

        print('  == end visualization ==')
        exit(0)






