import ray
import argparse

from env.inventory_env import InventoryManageEnv
from env.inventory_utils import Utils
from scheduler.inventory_random_policy import ProducerBaselinePolicy, BaselinePolicy

from scheduler.inventory_eoq_policy import ConsumerEOQPolicy as ConsumerBaselinePolicy

from config.inventory_config import env_config
from utility.visualization import visualization


from scheduler.inventory_dqn_baseline import ConsumerDQNTorchPolicy
from scheduler.Trainer import Trainer
import numpy as np
import os
from utility.tensorboard import TensorBoard
from scheduler.forecasting_model import Forecasting_model

# Configuration ===============================================================================

forecast_mode_config_default = {
    "hist_len": 21,
    "fore_len": 3,
    "batch_size": 128,
    "training_steps": 10000,
    "evaluation_steps": 200,
    "train_round": 50,

}

def train_forecasting_model(args):
    if not os.path.exists('train_log'):
        os.mkdir('train_log')
    writer = TensorBoard(f'train_log/{args.run_name}')
    print(" == start training forecasting model ==")
    forecast_config = forecast_mode_config_default.copy()
    forecast_model = Forecasting_model(forecast_config)
    for i in range(forecast_config['train_round']):
        eval_loss = forecast_model.eval_one_round(forecast_config['evaluation_steps'])
        train_loss = forecast_model.train_one_round(forecast_config['training_steps'])

        print(f"round {i}")
        print(f"train_loss: max: {np.max(train_loss):13.6f} mean: {np.mean(train_loss):13.6f} min: {np.min(train_loss):13.6f}")
        print(f"eval_loss:  max: {np.max(eval_loss):13.6f} mean: {np.mean(eval_loss):13.6f} min: {np.min(eval_loss):13.6f}")
        writer.add_scalar('ztrain/train_loss', np.mean(train_loss), i)
        writer.add_scalar('ztrain/eval_loss', np.mean(eval_loss), i)
    forecast_model.eval_all(f'train_log/{args.run_name}')
    return forecast_model


parser = argparse.ArgumentParser()
parser.add_argument("--torch", action="store_true")
parser.add_argument("--batch-size", type=int, default=2048)
parser.add_argument("--use-prev-action-reward", action="store_true")
parser.add_argument("--num-iterations", type=int, default=1000)
parser.add_argument("--visualization-frequency", type=int, default=100)
parser.add_argument("--run-name", type=str, default='1223_forecasting_hid32_sigmoid_2layer_train_more')

if __name__ == "__main__":
    args = parser.parse_args()
    train_forecasting_model(args)

