import random
import numpy as np

from tichu_env import Player

class AutonomousPlayer(Player):
    def action_probs(self):
        '''Returns list of actions and their odds.'''
        raise NotImplementedError
    
    def sample_action(self):
        '''Samples action based on odds calculated from action_probs'''
        actions, odds = zip(*self.action_probs())
        rand_idx = np.random.choice(len(actions), 1, p=odds)[0]
        return actions[rand_idx]
    
    def choose_exchange(self):
        '''Provides list of (user index, card index to send)'''
        raise NotImplementedError

class RandomPlayer(AutonomousPlayer):
    '''Simple player that performs random actions.'''
    
    def action_probs(self):
        my_options = self.possible_actions()
        action_num = len(my_options)
        any_option_odds = 1./action_num
        return [(action, any_option_odds) for action in my_options]
    
    def choose_exchange(self):
        other_idxs = list(filter(lambda x: x != self.player_id, range(4)))
        picked_cards = random.sample(range(len(self.hand)), 3)
        return list(zip(other_idxs, picked_cards))

def test_player():
    from tichu_env import Game
    game = Game(0, [RandomPlayer for _ in range(4)])
    game.run_game(upto='firstRound')
    print(game)
    
if __name__ == '__main__':
    test_player()