import torch
import numpy as np
# for i in range(21):
#     state['uncontrollable_part_state_cnn'][0][i] = demand_history[i]
#     state['uncontrollable_part_state_cnn'][1][i] = np.sin(date_info_hist[i]['isoweekday'] / 7 * 2 * np.pi)
#     state['uncontrollable_part_state_cnn'][2][i] = np.cos(date_info_hist[i]['isoweekday'] / 7 * 2 * np.pi)
#     state['uncontrollable_part_state_cnn'][3][i] = np.sin(date_info_hist[i]['month'] / 12 * 2 * np.pi)
#     state['uncontrollable_part_state_cnn'][4][i] = np.cos(date_info_hist[i]['month'] / 12 * 2 * np.pi)
#     state['uncontrollable_part_state_cnn'][5][i] = np.sin(date_info_hist[i]['day'] / 31 * 2 * np.pi)
#     state['uncontrollable_part_state_cnn'][6][i] = np.cos(date_info_hist[i]['day'] / 31 * 2 * np.pi)
# state['uncontrollable_part_state'] = state['uncontrollable_part_state_cnn'].copy().flatten()

class demand_augmentation():
    def __init__(self, noise_type='none', noise_scale=0.2, sparse_scale=0.1):
        self.noise_base = torch.cat([torch.ones([21]), torch.zeros([21*6])])
        self.noise_scale = noise_scale
        self.noise_type = noise_type # dense or sparse
        self.sparse_scale = sparse_scale

    def augment(self, data_in):
        if self.noise_type == 'none':
            return data_in
        elif self.noise_type == 'dense':
            noise = self.dense_noise(data_in)
        elif self.noise_type == 'sparse':
            noise = self.sparse_noise(data_in)
        else:
            raise Exception("only support noise_type 'none', 'dense' or 'sparse' ")
        return data_in + noise

    def dense_noise(self, data_in):
        noise = torch.rand(data_in.shape)
        noise = self.noise_scale * (2 * noise - 1)
        noise = noise * self.noise_base
        return noise

    def sparse_noise(self, data_in):
        noise = np.random.chisquare(2, data_in.shape)
        noise = torch.tensor(noise, dtype=torch.float32)
        mask = torch.rand(data_in.shape)
        mask = mask.lt(self.sparse_scale).to(dtype=torch.float32)
        noise = noise - data_in
        noise = noise * mask * self.noise_base

        return noise


