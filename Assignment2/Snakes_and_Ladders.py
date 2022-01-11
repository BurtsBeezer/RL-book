from dataclasses import dataclass
from rl.distribution import Categorical
from rl.markov_process import FiniteMarkovProcess, NonTerminal
import itertools
from matplotlib import pyplot as plt

@dataclass(frozen=True)
class State_SandL:
    position: int

class MP_SandL(FiniteMarkovProcess[State_SandL]):
    def __init__(self):
        self.snakes_dict = {
            16:6,
            49:11,
            47:26,
            62:19,
            64:60,
            56:53,
            93:73,
            95:75,
            98:78,
            87:24
            }

        self.ladders_dict = {
            1:38,
            4:14,
            9:31,
            28:84,
            21:42,
            36:44,
            51:67,
            71:91,
            80:100
            }
        
        self.dice_roll_dict = dict()
        total_prob = 1
        for roll in range(1,7):
            self.dice_roll_dict[roll] = total_prob/6
        super().__init__(self.get_transition_map())
         
    def get_transition_map(self):
        transition_dict = dict()
        for pos in range(100):
            state_prob_map = self.get_state_prob_map(pos)
            transition_dict[State_SandL(pos)] = Categorical(state_prob_map)
        return transition_dict

    def get_state_prob_map(self, position):
        state_prob_dict = dict()
        for roll, prob in self.dice_roll_dict.items():
            new_position = position
            new_position += roll
            new_position = self.ladders_dict.get(new_position,new_position)
            new_position = self.snakes_dict.get(new_position,new_position)
            new_position = min(new_position,100)
            if State_SandL(new_position) in state_prob_dict:
                state_prob_dict[State_SandL(new_position)] += prob
            else:
                state_prob_dict[State_SandL(new_position)] = prob
        return state_prob_dict

if __name__ == '__main__':
    sl_mp = MP_SandL()

    print("Transition Map")
    print(sl_mp)

    ls = []
    start_state_distribution = Categorical({NonTerminal(State_SandL(1)): 1})
    trace_iter = sl_mp.traces(start_state_distribution)
    for k in range(1000):
        ls.append(len(list(next(trace_iter))))
    plt.hist(ls,bins=range(200))
    plt.savefig('Snakes_and_Ladders_distribution.png')
    print(min(ls))
