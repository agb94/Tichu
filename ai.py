import random
import numpy as np

from tichu_env import *
import torch

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
    def __init__(self, game, player_id, network, device='cuda'):
        super().__init__(game, player_id)
        self.network = network
        self.device = device
    
    def normalized_playerid(self, id):
        return (id - self.player_id) % NUM_PLAYERS

    def card_rep(self, action):
        '''Returns game state in a tensor format.'''
        # card information
        norm_card_states = np.zeros((4, 13))
        spec_card_states = np.zeros((4, 1))
        
        modified_hand = set(self.hand) - set(action.cards)
        curr_cards = set(sum([c.cards for c in self.game.current], [])) | set(action.cards)
        used_cards = set(self.game.used)
        known_cards = set(self.card_locs.keys())
        # card state key: 0 in my hand; 1 in current play; 2 played; 
        # 3-5 known but unused; 6 unknown
        for card in self.game.deck.cards:
            if card in modified_hand: # not sure if ok
                state_val = 0
            elif card in curr_cards:
                state_val = 1
            elif card in used_cards:
                state_val = 2
            elif card in known_cards:
                card_owner = self.card_locs[card]
                state_val = self.normalized_playerid(card_owner) + 3
            elif card in self.game.unused_cards:
                state_val = 6
            else:
                raise ValueError(f'{card} is of unknown state')
            
            if card.number is None:
                suite_idx = Card.SPECIALS.index(card.suite)
                spec_card_states[suite_idx, 0] = state_val
            else:
                suite_idx = Card.COLORS.index(card.suite)
                number_idx = Card.NUMBERS[card.number] - 2
                norm_card_states[suite_idx, number_idx] = state_val
        
        return norm_card_states, spec_card_states
    
    def noncard_rep(self, action):
        if action is None:
            curr_owner = (self.game.turn - (self.game.pass_count-1)) % 4
            owner_rep = self.normalized_playerid(curr_owner)
            call_value = 0
        else:
            owner_rep = 0 # if not skip, any action incurs my owning pile (I think).
            if isinstance(action, MahJongSingle):
                call_value = action.call_value - 1
            else:
                call_value = 0
        return owner_rep, call_value

    @classmethod
    def state2Tensor(cls, card_rep, noncard_rep, device='cuda'):
        norm_cr, spec_cr = card_rep
        owner_idx, call_idx = noncard_rep
        norm_cr = torch.LongTensor(norm_cr).unsqueeze(0).to(device)
        spec_cr = torch.LongTensor(spec_cr).unsqueeze(0).to(device)

        owner_idx = torch.LongTensor([owner_idx]).unsqueeze(0).to(device)
        call_idx = torch.LongTensor([call_idx]).unsqueeze(0).to(device)

        return (norm_cr, spec_cr), (owner_idx, call_idx)

    def action_value(self, action):
        action_card_rep = self.card_rep(action)
        action_noncard_rep = self.noncard_rep(action)
        tensor_cr, tensor_ncr = self.__class__.state2Tensor(
            action_card_rep, action_noncard_rep, device=self.device
        )
        estimated_value = self.network(tensor_cr, tensor_ncr)
        return estimated_value
    
    def action_probs(self):
        my_options = self.possible_actions()
        
        action_num = len(my_options)
        any_option_odds = 1./action_num
        return [(action, any_option_odds) for action in my_options]
                
    def call_big_tichu(self):
        baseline_odds = 0.0
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
    from model import TichuNet1
    device = 'cuda'
    tn1 = TichuNet1()
    tn1.to(device)
    game = Game(0, [lambda x, y: NeuralPlayer(x, y, tn1, device=device) 
                    for _ in range(4)])
    game.run_game(upto='firstRound', verbose=True)
    print(game)
    test_idx = 0
    print(game.players[test_idx].state_rep(game.players[test_idx].hand))
    
if __name__ == '__main__':
    test_player()