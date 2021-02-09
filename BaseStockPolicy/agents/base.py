from abc import ABC
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from collections import deque
import numpy as np
import random as rnd
import networkx as nx
from enum import Enum, auto

class Cell(ABC):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"{self.__class__.__name__} ({self.x}, {self.y})"
    
class Agent(ABC):
    def act(self, control):
        pass
    
@dataclass
class BalanceSheet:
    profit: int = 0
    loss: int = 0
        
    def total(self) -> int:
        return self.profit + self.loss
        
    def __add__(self, other):
        return BalanceSheet(self.profit + other.profit, self.loss + other.loss)
    
    def __sub__(self, other):
        return BalanceSheet(self.profit - other.profit, self.loss - other.loss)
    
    def __repr__(self):
        return f"{round(self.profit+self.loss, 0)} ({round(self.profit, 0)} {round(self.loss, 0)})"
    
    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)