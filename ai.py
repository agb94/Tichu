import random
import numpy as np

from tichu_env import Player

class RandomPlayer(Player):
    '''Simple player that performs random actions.
    Mostly provided as a reference point for AI API.'''
    
    def sample_action(self):
        '''Samples uniformly from all available actions.
        Returns an instance of the Combination class,
        or an empty list if there is no action possible.'''
        my_options = self.possible_actions()
        if len(my_options) == 0:
            return []
        else:
            return random.sample(my_options, 1)[0]

def test_player():
    from tichu_env import Game
    game = Game(0, [RandomPlayer for _ in range(4)])
    print(game)
    game.mark_exchange(0, 1, 0)
    game.mark_exchange(0, 2, 7)
    game.mark_exchange(0, 3, 1)
    game.mark_exchange(1, 0, 0)
    game.mark_exchange(1, 2, 1)
    game.mark_exchange(1, 3, 7)
    game.mark_exchange(2, 0, 7)
    game.mark_exchange(2, 1, 0)
    game.mark_exchange(2, 3, 1)
    game.mark_exchange(3, 0, 0)
    game.mark_exchange(3, 1, 7)
    game.mark_exchange(3, 2, 1)
    game.exchange()
    print(game)
    
    for turn_idx in range(8):
        player = game.players[game.turn]
        a = player.sample_action()
        print(game.turn, a)
        game.play(game.turn, a)
    
if __name__ == '__main__':
    test_player()