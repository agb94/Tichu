import random
import numpy as np

from tichu_env import *

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
    
    def call_big_tichu(self):
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
    
    def call_big_tichu(self):
        baseline_odds = 0.1
        if random.random() < baseline_odds:
            return True
        else:
            return False

class NeuralPlayer(AutonomousPlayer):
    '''Player that relies on value network to compute next move'''
    def __init__(self, game, player_id, network):
        super().__init__(game, player_id)
        self.network = network
    
    def state_rep(self, modified_hand):
        '''Returns game state in a tensor format.'''
        norm_card_states = np.zeros((4, 13))
        spec_card_states = np.zeros((4, 1))
        obtain_states = np.zeros((4, 13))
        current_top = -1
        tichu_calls = []
        cards_in_hand = []
        
        curr_cards = set(sum([c.cards for c in self.game.current], []))
        used_cards = set(self.game.used)
        # card state key: 0 in my hand; 1 in current play; 2 played; 
        # 3-5 known but unused; 6 unknown
        for card in self.game.deck.cards:
            # normies
            if card in modified_hand: # not sure if ok
                state_val = 0
            elif card in curr_cards:
                state_val = 1
            elif card in used_cards:
                state_val = 2
            elif card in self.card_locs.keys():
                raise NotImplementedError # todo
            elif card in self.game.unused_cards:
                state_val = 3
            else:
                raise ValueError(f'{card} is of unknown state')
            
            if card.number is None:
                suite_idx = Card.SPECIALS.index(card.suite)
                spec_card_states[suite_idx, 0] = state_val
            else:
                suite_idx = Card.COLORS.index(card.suite)
                number_idx = Card.NUMBERS[card.number] - 2
                norm_card_states[suite_idx, number_idx] = state_val
        
        print(norm_card_states)
        print('a-ok')
        exit(0)
    
    def action_probs(self):
        my_options = self.possible_actions()
        action_num = len(my_options)
        any_option_odds = 1./action_num
        return [(action, any_option_odds) for action in my_options]
        pass
                
    def call_big_tichu(self):
        baseline_odds = 0.1
        if random.random() < baseline_odds:
            return True
        else:
            return False
    
    def choose_exchange(self):
        other_idxs = list(filter(lambda x: x != self.player_id, range(4)))
        picked_cards = random.sample(range(len(self.hand)), 3)
        return list(zip(other_idxs, picked_cards))
        
def test_player():
    from tichu_env import Game
    game = Game(0, [lambda x, y: NeuralPlayer(x, y, None) for _ in range(4)])
    game.run_game(upto='firstRound', verbose=True)
    print(game)
    print(game.players[0].state_rep(game.players[0].hand))
    
if __name__ == '__main__':
    test_player()