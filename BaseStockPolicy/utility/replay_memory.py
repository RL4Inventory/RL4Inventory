import numpy as np
import random
import torch
import pickle as pickle

class replay_memory(object):
    # replay memory
    def __init__(self, buffer_size, rcsize=40):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0

    def clear(self):
        self.buffer = []
        self.index = 0

    def push(self, obj):
        if len(self.buffer) == self.buffer_size:
            self.buffer[self.index] = obj
        else:
            self.buffer.append(obj)
        self.index = (self.index + 1) % self.buffer_size

    def sample(self, batch_size, device=torch.device("cpu")):
        batch = random.sample(self.buffer, batch_size)

        res = []
        for i in range(8):
            k = torch.stack(tuple(item[i] for item in batch), dim=0)
            res.append(k.to(device))

        return res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7]

    def __len__(self):
        return len(self.buffer)



class Dataset(object):
    def __init__(self, buffer_size, ):
        self.buffer_size = buffer_size
        self.buffer = []
        self.index = 0

    def push(self, obj):
        if len(self.buffer) == self.buffer_size:
            self.buffer[self.index] = obj
        else:
            self.buffer.append(obj)
        self.index = (self.index + 1) % self.buffer_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        res = []
        for i in range(2):
            k = torch.stack(tuple(item[i] for item in batch), dim=0)
            res.append(k)

        return res[0], res[1]

    def __len__(self):
        return len(self.buffer)