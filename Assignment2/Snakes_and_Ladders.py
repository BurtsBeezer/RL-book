from dataclasses import dataclass
from typing import Optional, Mapping
import numpy as np
import itertools
from rl.distribution import Distribution
from rl.markov_process import MarkovProcess, NonTerminal, State

import random

@dataclass(frozen=True)
class Die(Distribution):
    def __init__(self, sides):
        self.sides = sides
    def sample(self):
        return random.randint(1, self.sides)
six_sided = Die(6)
def roll_dice():
    return six_sided.sample()

@dataclass(frozen=True)
class StateMP1:
    position: int

@dataclass
class MP1:
    def next_state(
            self,
            state: NonTerminal[StateMP1]
    ) -> Categorical[State[StateMP1]]:
        
